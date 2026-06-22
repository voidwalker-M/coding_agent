"""
eval/verifiers.py

Independent verifiers: the "graders" for evaluation tasks.

Each Verifier is a callable:
    verifier(repo_path: str) -> (passed: bool, detail: str)

Key design: verifiers are completely independent of the agent's tool layer and
self-reported status. Even if the agent calls FINISH claiming success, the
verifier's objective re-run result is the ground truth.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class Verifier(ABC):
    """Abstract base class for evaluation verifiers."""

    @abstractmethod
    def __call__(self, repo_path: str) -> tuple[bool, str]:
        """Return (passed, human-readable detail). Never raise — exceptions become failures."""
        ...

    @property
    def description(self) -> str:
        return self.__class__.__name__


class PytestVerifier(Verifier):
    """Re-run pytest; exit code 0 means passed. The most common grading method."""

    def __init__(self, path: str = ".", timeout: int = 120) -> None:
        self._path = path
        self._timeout = timeout

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", self._path, "-q", "--no-header", "--tb=line"],
                cwd=repo_path, capture_output=True, text=True, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {self._timeout}s"
        except Exception as exc:
            return False, f"pytest could not run: {exc}"

        tail = (proc.stdout + proc.stderr).strip().splitlines()
        summary = tail[-1] if tail else "(no output)"
        return proc.returncode == 0, summary


class FileExistsVerifier(Verifier):
    """Passes if the target file exists."""

    def __init__(self, relpath: str) -> None:
        self._rel = relpath

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        p = Path(repo_path) / self._rel
        if p.exists():
            return True, f"{self._rel} exists"
        return False, f"{self._rel} not found"


class FileContainsVerifier(Verifier):
    """Passes if the target file exists and contains the specified substring."""

    def __init__(self, relpath: str, substring: str) -> None:
        self._rel = relpath
        self._sub = substring

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        p = Path(repo_path) / self._rel
        if not p.exists():
            return False, f"{self._rel} not found"
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, f"cannot read {self._rel}: {exc}"
        if self._sub in content:
            return True, f"{self._rel} contains expected content"
        return False, f"{self._rel} missing {self._sub!r}"


class CommandVerifier(Verifier):
    """
    Run a command and grade by exit code / output substring.
    Example: CommandVerifier("python hello.py", expect_substring="hello world")
    """

    def __init__(
        self,
        cmd: str,
        expect_returncode: int | None = 0,
        expect_substring: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._cmd = cmd
        self._rc = expect_returncode
        self._sub = expect_substring
        self._timeout = timeout

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                self._cmd, shell=True, cwd=repo_path,
                capture_output=True, text=True, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"command timed out: {self._cmd!r}"
        except Exception as exc:
            return False, f"command failed to run: {exc}"

        out = proc.stdout + proc.stderr
        if self._rc is not None and proc.returncode != self._rc:
            return False, f"exit {proc.returncode} (expected {self._rc})"
        if self._sub is not None and self._sub not in out:
            return False, f"output missing {self._sub!r}"
        return True, f"command ok: {self._cmd!r}"


class AllOfVerifier(Verifier):
    """Composite verifier: passes only when all sub-verifiers pass."""

    def __init__(self, *verifiers: Verifier) -> None:
        self._verifiers = verifiers

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        details = []
        for v in self._verifiers:
            ok, detail = v(repo_path)
            details.append(("✓" if ok else "✗") + " " + detail)
            if not ok:
                return False, " | ".join(details)
        return True, " | ".join(details)


class UnmodifiedFilesVerifier(Verifier):
    """
    Hallucination / over-edit guard: passes only if the listed files were NOT
    changed from the eval baseline commit.

    The harness makes a baseline `git commit` before the agent runs, so a simple
    `git diff` against HEAD detects any edit (including deletion). Compose this
    with PytestVerifier via AllOfVerifier to grade "fixed the bug AND did not
    touch files it had no business touching" — a direct proxy for the
    edit-the-wrong-file failure mode (the doc's 幻觉率 / hallucination rate).
    """

    def __init__(self, *relpaths: str) -> None:
        if not relpaths:
            raise ValueError("UnmodifiedFilesVerifier needs at least one path")
        self._paths = list(relpaths)

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-only", "HEAD", "--", *self._paths],
                cwd=repo_path, capture_output=True, text=True, timeout=20,
            )
        except Exception as exc:
            return False, f"git diff could not run: {exc}"
        if proc.returncode != 0:
            return False, f"git diff failed: {proc.stderr.strip() or 'no baseline commit?'}"
        changed = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if changed:
            return False, f"protected files were modified: {', '.join(changed)}"
        return True, f"{len(self._paths)} protected file(s) unchanged"


class NoRegressionVerifier(Verifier):
    """
    Regression guard: re-run a set of tests that must stay green.

    PytestVerifier checks that the *target* tests now pass; this checks that a
    designated pre-existing test path was not broken as a side effect — the
    analog of SWE-bench's PASS_TO_PASS set. Defaults to the whole repo.
    """

    def __init__(self, path: str = ".", timeout: int = 120) -> None:
        self._inner = PytestVerifier(path=path, timeout=timeout)

    def __call__(self, repo_path: str) -> tuple[bool, str]:
        ok, detail = self._inner(repo_path)
        return ok, ("no regressions: " + detail) if ok else ("regression: " + detail)
