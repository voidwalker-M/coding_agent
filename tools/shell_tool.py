"""
tools/shell_tool.py

Shell command execution tool. Four-layer protection:
1. Blacklist:  reject obviously destructive commands (hard block, cannot be bypassed)
2. Whitelist:  read-only commands execute immediately without confirmation
3. Confirm:    write operations wait for user y/n (injected via confirm_callback)
4. Timeout + output truncation: prevent hangs and context explosion

Confirmation design:
- confirm_callback is a Callable[[str], bool] that returns True to allow execution
- Default None (no confirmation, execute directly) — used in run mode
- Chat / interactive mode passes in a real terminal confirmation function
- Tests pass in a mock, no real terminal needed
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable

from tools.base import BaseTool, ToolResult
from tools.runtime import LocalRuntime, Runtime


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MAX_OUTPUT_CHARS = 8_000

# Hard-block blacklist (never execute, never ask the user)
_BLOCKED_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",       # fork bomb
    "chmod -R 777 /",
    "chown -R",
    "> /dev/sda",
)

# Read-only command prefix whitelist (execute without asking)
_READONLY_PREFIXES: tuple[str, ...] = (
    "ls", "ll", "la",
    "cat", "head", "tail", "less", "more",
    "echo", "printf",
    "pwd", "whoami", "which", "type",
    "find", "locate",
    "grep", "egrep", "fgrep", "rg", "ag",
    "wc", "sort", "uniq", "cut", "awk", "sed -n",
    "diff", "diff3",
    "file", "stat",
    "python -c", "python3 -c",
    "python -m pytest", "python3 -m pytest", "pytest",
    "git status", "git diff", "git log", "git show",
    "git branch", "git tag", "git remote",
    "git stash list",
    "tree",
    "env", "printenv",
    "ps", "top", "htop",
    "df", "du",
    "uname", "hostname",
    "date", "cal",
    "man", "help",
)

# Dangerous command keywords that require confirmation (when not on the whitelist)
_CONFIRM_KEYWORDS: tuple[str, ...] = (
    "rm ", "rmdir",
    "mv ",
    "cp -r", "cp -f",
    "chmod", "chown",
    "pip install", "pip uninstall",
    "npm install", "npm uninstall",
    "git commit", "git push", "git reset",
    "git checkout", "git merge", "git rebase",
    "git clean",
    "sudo",
    "curl", "wget",            # network requests
    "kill", "pkill", "killall",
    "shutdown", "reboot",
    "docker", "kubectl",
    "make", "make install",
    "> ",                      # write redirect (>> append is not blocked)
    "| tee ",
)

# Confirm callback type: receives command string, returns True=allow / False=deny
ConfirmCallback = Callable[[str], bool]


# ---------------------------------------------------------------------------
# ShellTool
# ---------------------------------------------------------------------------

class ShellTool(BaseTool):
    """
    Execute a shell command and return stdout + stderr.

    params:
        cmd (str):     shell command string
        timeout (int): timeout in seconds (default 30)
        cwd (str):     working directory (default current directory)

    Constructor args:
        confirm_callback: called when confirmation is needed; returns True to allow.
                          None means skip confirmation (default for run mode).
    """

    def __init__(
        self,
        confirm_callback: ConfirmCallback | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        self._confirm_callback = confirm_callback
        self._runtime = runtime or LocalRuntime()

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output (stdout + stderr combined). "
            "Timeout is 30s by default. Avoid long-running commands; "
            "prefer targeted commands like 'grep', 'pytest tests/foo.py', 'git diff'."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional)",
                },
            },
            "required": ["cmd"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cmd: str = params.get("cmd", "").strip()
        timeout: int = int(params.get("timeout", 30))
        cwd: str | None = params.get("cwd", None)

        if not cmd:
            return ToolResult(success=False, output="", error="cmd is required")

        # Layer 1: hard blacklist block
        blocked = _check_blocked(cmd)
        if blocked:
            return ToolResult(
                success=False,
                output="",
                error=f"Command blocked for safety: matched '{blocked}'",
            )

        # Layer 2: whitelist — execute without confirmation
        if not _needs_confirm(cmd):
            return self._run(cmd, timeout, cwd)

        # Layer 3: confirmation gate
        if self._confirm_callback is not None:
            allowed = self._confirm_callback(cmd)
            if not allowed:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command rejected by user: {cmd!r}",
                )
        # confirm_callback is None → skip confirmation and execute (run mode)

        return self._run(cmd, timeout, cwd)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, cmd: str, timeout: int, cwd: str | None) -> ToolResult:
        """Execute the command via runtime (local or Docker sandbox)."""
        result = self._runtime.exec(cmd, cwd=cwd, timeout=timeout)
        output = _truncate(result.output, MAX_OUTPUT_CHARS)
        if not result.success:
            # Distinguish timeout from ordinary errors
            if "timed out" in result.stderr.lower():
                error = result.stderr.strip()
            else:
                error = f"Exit code: {result.returncode}"
        else:
            error = None
        return ToolResult(success=result.success, output=output, error=error)


# ---------------------------------------------------------------------------
# Helper functions (exported for testing)
# ---------------------------------------------------------------------------

def _check_blocked(cmd: str) -> str | None:
    """Return the matching blacklist pattern, or None if no match."""
    cmd_lower = cmd.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in cmd_lower:
            return pattern
    return None


def _is_readonly(cmd: str) -> bool:
    """
    Check whether a command is on the read-only whitelist.
    Commands containing a write redirect (>) are not considered read-only
    even if the command name itself is on the whitelist.
    """
    # Write redirect: >[^>] matches ">" but not ">>"; ">>" (append) is relatively safe
    import re as _re
    if _re.search(r'(?<![>])>(?![>])', cmd):
        return False
    stripped = cmd.strip().lower()
    for prefix in _READONLY_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + " "):
            return True
    return False


def _needs_confirm(cmd: str) -> bool:
    """
    Check whether a command requires user confirmation.
    Not on the whitelist AND contains a dangerous keyword → needs confirmation.
    """
    if _is_readonly(cmd):
        return False
    cmd_lower = cmd.lower()
    return any(kw in cmd_lower for kw in _CONFIRM_KEYWORDS)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate long output: keep 60% from the head and 40% from the tail."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n... [{omitted} characters truncated] ...\n"
        + text[-tail:]
    )


# ---------------------------------------------------------------------------
# Terminal confirmation function (used directly in cli/chat)
# ---------------------------------------------------------------------------

def terminal_confirm(cmd: str) -> bool:
    """
    Display the command in the terminal and wait for user confirmation.
    Returns True to allow, False to deny.

    Display format:
        ⚠  Agent wants to run:
           $ git commit -m "fix parser"
        Allow? [y/N] _
    """
    import sys

    if not sys.stdin.isatty():
        # Non-interactive (pipe / CI): deny by default to avoid accidental execution
        print(f"\n[confirm] Non-interactive terminal, rejecting: {cmd!r}", flush=True)
        return False

    print(f"\n\033[33m  ⚠  Agent wants to run:\033[0m")
    print(f"     \033[1m$ {cmd}\033[0m")

    while True:
        try:
            ans = input("  Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            print("  \033[31m✗ Rejected\033[0m")
            return False
        print("  Please enter y or n.")


def always_allow(cmd: str) -> bool:
    """Skip confirmation and always allow (used for --no-confirm mode)."""
    return True


def always_deny(cmd: str) -> bool:
    """Skip confirmation and always deny (used for tests or CI mode)."""
    return False
