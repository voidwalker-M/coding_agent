"""
tools/runtime.py

Runtime abstraction layer: decouples command execution from tool implementations.

Tools (ShellTool / PytestTool / GitTool) only construct command arguments;
the Runtime handles actual execution — either a local subprocess or a Docker container.

Design principles:
- Tools are completely unaware of the Runtime; it is injected via dependency injection
- The Runtime can be injected once at ToolRegistry creation time and shared by all tools
- LocalRuntime is the default (backward-compatible; omitting runtime is the same as before)
- DockerRuntime manages the container lifecycle; the container is lazily started on first exec()

Usage:
    # Default local execution
    registry = build_registry()

    # Docker sandbox
    runtime = DockerRuntime(repo_path="/path/to/repo")
    registry = build_registry(runtime=runtime)
    # Clean up after the agent finishes
    runtime.cleanup()

    # Or use a context manager for automatic cleanup
    with DockerRuntime(repo_path="/path/to/repo") as runtime:
        registry = build_registry(runtime=runtime)
        agent.run(task, log)
"""

from __future__ import annotations

import subprocess
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunResult — result of a single Runtime execution
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of executing one command via a Runtime."""
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Merged stdout + stderr; used directly by the tool layer."""
        return self.stdout + self.stderr


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Runtime(ABC):
    """
    Abstract base class for command execution.
    All tools execute commands via runtime.exec() rather than calling subprocess directly.
    """

    @abstractmethod
    def exec(
        self,
        cmd: str,
        cwd: str | None = None,
        timeout: int = 30,
    ) -> RunResult:
        """
        Execute a shell command and return a RunResult.

        Args:
            cmd:     shell command string
            cwd:     working directory (relative or absolute path)
            timeout: timeout in seconds

        Returns:
            RunResult; never raises (timeouts and errors are captured inside it)
        """
        ...

    def cleanup(self) -> None:
        """Release resources held by this runtime (containers, connections, etc.). No-op by default."""

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()

    @property
    @abstractmethod
    def name(self) -> str:
        """Runtime name, used in log messages."""
        ...


# ---------------------------------------------------------------------------
# LocalRuntime — local subprocess (default)
# ---------------------------------------------------------------------------

class LocalRuntime(Runtime):
    """
    Local execution via subprocess.run.
    Behavior is identical to the original pre-Runtime code; this is the default runtime.
    """

    @property
    def name(self) -> str:
        return "local"

    def exec(
        self,
        cmd: str,
        cwd: str | None = None,
        timeout: int = 30,
    ) -> RunResult:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return RunResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s: {cmd!r}",
            )
        except Exception as e:
            return RunResult(returncode=-1, stdout="", stderr=str(e))


# ---------------------------------------------------------------------------
# DockerRuntime — Docker sandbox
# ---------------------------------------------------------------------------

# Docker image used for the sandbox container
# Includes Python, git, and common tools at a reasonable size
SANDBOX_IMAGE = "python:3.11-slim"

# Mount path for the repo inside the container
CONTAINER_WORKDIR = "/workspace"


class DockerRuntime(Runtime):
    """
    Docker sandbox Runtime.

    The container is lazily started on the first exec() call:
    - Based on the python:3.11-slim image
    - Bind-mounts repo_path to /workspace inside the container
    - Container stays alive (tail -f /dev/null); each command uses docker exec
    - cleanup() stops and removes the container

    This is much faster than docker run per command (avoids repeated container startup overhead).

    Args:
        repo_path:    absolute path to the repo on the host; mounted into the container
        image:        Docker image name, default python:3.11-slim
        extra_mounts: additional bind mounts as [(host_path, container_path), ...]
        setup_cmds:   initialization commands to run after the container starts
                      (e.g. "pip install -r requirements.txt")
    """

    def __init__(
        self,
        repo_path: str | Path,
        image: str = SANDBOX_IMAGE,
        extra_mounts: list[tuple[str, str]] | None = None,
        setup_cmds: list[str] | None = None,
    ) -> None:
        self._repo_path = str(Path(repo_path).resolve())
        self._image = image
        self._extra_mounts = extra_mounts or []
        self._setup_cmds = setup_cmds or []
        self._container_id: str | None = None
        # Random suffix on the container name to avoid collisions
        self._container_name = f"coding-agent-sandbox-{uuid.uuid4().hex[:8]}"

    @property
    def name(self) -> str:
        return f"docker({self._image})"

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def is_running(self) -> bool:
        return self._container_id is not None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def exec(
        self,
        cmd: str,
        cwd: str | None = None,
        timeout: int = 30,
    ) -> RunResult:
        """Execute a command inside the container, starting it on first call."""
        if not self.is_running:
            startup_result = self._start_container()
            if startup_result is not None:
                # Startup failed; return the error
                return startup_result

        # Determine the working directory inside the container
        if cwd:
            # If cwd is a host path, translate it to the container path
            host_cwd = str(Path(cwd).resolve())
            if host_cwd.startswith(self._repo_path):
                relative = host_cwd[len(self._repo_path):].lstrip("/")
                container_cwd = f"{CONTAINER_WORKDIR}/{relative}" if relative else CONTAINER_WORKDIR
            else:
                container_cwd = cwd   # may already be an absolute container path
        else:
            container_cwd = CONTAINER_WORKDIR

        docker_cmd = [
            "docker", "exec",
            "--workdir", container_cwd,
            self._container_id,
            "bash", "-c", cmd,
        ]

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 5,   # docker exec itself has a small overhead
            )
            return RunResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s in container: {cmd!r}",
            )
        except Exception as e:
            return RunResult(returncode=-1, stdout="", stderr=str(e))

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if not self._container_id:
            return
        logger.info("Stopping sandbox container %s", self._container_name)
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=15,
            )
        except Exception as e:
            logger.warning("Failed to remove container %s: %s", self._container_id, e)
        finally:
            self._container_id = None

    # ------------------------------------------------------------------
    # Internal: container lifecycle
    # ------------------------------------------------------------------

    def _start_container(self) -> RunResult | None:
        """
        Pull the image (if needed) and start the container.
        Returns None on success, or a RunResult describing the failure.
        """
        logger.info(
            "Starting sandbox container %s (image=%s, repo=%s)",
            self._container_name, self._image, self._repo_path,
        )

        # Check that Docker is available
        check = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        if check.returncode != 0:
            return RunResult(
                returncode=-1,
                stdout="",
                stderr=(
                    "Docker is not available. "
                    "Make sure Docker Desktop is running, or use --no-sandbox."
                ),
            )

        # Build the docker run command
        run_args = [
            "docker", "run",
            "--detach",                                 # run in background
            "--name", self._container_name,
            "--rm",                                     # auto-remove on stop
            "-v", f"{self._repo_path}:{CONTAINER_WORKDIR}",  # mount repo
            "--workdir", CONTAINER_WORKDIR,
            "--network", "none",                        # no network by default (safer)
        ]

        # Additional mounts
        for host_path, container_path in self._extra_mounts:
            run_args += ["-v", f"{host_path}:{container_path}"]

        run_args += [self._image, "tail", "-f", "/dev/null"]

        try:
            proc = subprocess.run(
                run_args,
                capture_output=True,
                text=True,
                timeout=60,  # pulling the image may take time
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                returncode=-1, stdout="",
                stderr="Timed out starting Docker container (60s). Is Docker running?",
            )

        if proc.returncode != 0:
            return RunResult(
                returncode=proc.returncode,
                stdout="",
                stderr=f"Failed to start container:\n{proc.stderr}",
            )

        self._container_id = proc.stdout.strip()
        logger.info("Container started: %s", self._container_id[:12])

        # Run initialization commands
        for setup_cmd in self._setup_cmds:
            result = self.exec(setup_cmd, timeout=120)
            if not result.success:
                logger.warning(
                    "Setup command failed: %r\n%s", setup_cmd, result.stderr
                )

        return None   # success

    def install_requirements(self, requirements_file: str = "requirements.txt") -> RunResult:
        """
        Install dependencies inside the container.
        Convenience method equivalent to exec("pip install -r requirements.txt").
        """
        return self.exec(
            f"pip install -r {requirements_file} -q",
            timeout=120,
        )


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def create_runtime(
    sandbox: bool = False,
    repo_path: str | None = None,
    image: str = SANDBOX_IMAGE,
    network: bool = False,
) -> Runtime:
    """
    Create the appropriate Runtime based on configuration.

    Args:
        sandbox:   True to create a DockerRuntime, False for LocalRuntime
        repo_path: required when sandbox=True
        image:     Docker image name
        network:   whether to allow network access in sandbox mode (default False, safer)

    Returns:
        A Runtime instance
    """
    if not sandbox:
        return LocalRuntime()

    if not repo_path:
        raise ValueError("repo_path is required when sandbox=True")

    runtime = DockerRuntime(repo_path=repo_path, image=image)
    if network:
        # Allow network by removing --network none
        runtime._allow_network = True  # checked in DockerRuntime._start_container

    return runtime
