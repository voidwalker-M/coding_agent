"""
tools/git_tool.py

Git operation tools providing four actions:
- git_status:  show working tree status (equivalent to git status --short)
- git_diff:    show changes (equivalent to git diff or git diff HEAD)
- git_add:     stage files (equivalent to git add)
- git_commit:  commit (equivalent to git commit -m)

Design decisions:
- Does not wrap git push / PR creation; those are handled by entry/github_issue.py
- git_diff truncates output; diffs from large refactors can be very long
- All operations call the git CLI via subprocess rather than using gitpython
  (fewer dependencies; CLI output is easier for the agent to understand)
"""

from __future__ import annotations

import subprocess
from typing import Any

from tools.base import BaseTool, ToolResult
from tools.runtime import LocalRuntime, Runtime


MAX_DIFF_CHARS = 8_000


def _run_git(
    args: list[str],
    cwd: str | None = None,
    runtime: "Runtime | None" = None,
) -> tuple[bool, str]:
    """
    Run a git command and return (success, output).
    Falls back to subprocess when runtime is None (backward compatible).
    """
    from tools.runtime import LocalRuntime
    rt = runtime or LocalRuntime()
    cmd = "git " + " ".join(
        f'"{a}"' if " " in a else a for a in args
    )
    result = rt.exec(cmd, cwd=cwd, timeout=30)
    output = result.output.strip()
    return result.success, output


class GitStatusTool(BaseTool):
    """
    (see class docstring below)
    """

    def __init__(self, runtime: Runtime | None = None) -> None:
        from tools.runtime import LocalRuntime
        self._runtime = runtime or LocalRuntime()

    """
    Show working tree status.

    params:
        cwd (str): repository root directory (default: current directory)
    """

    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return (
            "Show the working tree status (modified, untracked, staged files). "
            "Run this before committing to see what has changed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cwd": {"type": "string", "description": "Repository root directory"},
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cwd = params.get("cwd")
        success, output = _run_git(["status", "--short", "--branch"], cwd=cwd, runtime=self._runtime)
        if not output:
            output = "Nothing to commit, working tree clean"
        return ToolResult(success=success, output=output, error=None if success else output)


class GitDiffTool(BaseTool):
    """
    (see class docstring below)
    """

    def __init__(self, runtime: Runtime | None = None) -> None:
        from tools.runtime import LocalRuntime
        self._runtime = runtime or LocalRuntime()

    """
    Show change diff.

    params:
        staged (bool): True to show staged diff (git diff --cached), default False
        path (str):    only diff a specific file
        cwd (str):     repository root directory
    """

    @property
    def name(self) -> str:
        return "git_diff"

    @property
    def description(self) -> str:
        return (
            "Show changes in the working tree or staging area. "
            "Use staged=true to see what will be committed. "
            "Use path to diff a specific file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes (git diff --cached). Default false.",
                },
                "path": {
                    "type": "string",
                    "description": "Specific file to diff (optional)",
                },
                "cwd": {"type": "string", "description": "Repository root directory"},
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cwd = params.get("cwd")
        staged = params.get("staged", False)
        path = params.get("path")

        args = ["diff"]
        if staged:
            args.append("--cached")
        if path:
            args += ["--", path]

        success, output = _run_git(args, cwd=cwd, runtime=self._runtime)

        if not output:
            label = "staged" if staged else "unstaged"
            return ToolResult(success=True, output=f"No {label} changes.")

        # Truncate excessively long diffs
        if len(output) > MAX_DIFF_CHARS:
            kept = MAX_DIFF_CHARS
            omitted = len(output) - kept
            output = output[:kept] + f"\n... [{omitted} chars truncated]"

        return ToolResult(success=success, output=output, error=None if success else output)


class GitAddTool(BaseTool):
    """
    (see class docstring below)
    """

    def __init__(self, runtime: Runtime | None = None) -> None:
        from tools.runtime import LocalRuntime
        self._runtime = runtime or LocalRuntime()

    """
    Stage files.

    params:
        paths (list[str]): file paths to stage; default ["."] (stage all)
        cwd (str):         repository root directory
    """

    @property
    def name(self) -> str:
        return "git_add"

    @property
    def description(self) -> str:
        return (
            "Stage files for commit. "
            "Pass a list of paths, or omit to stage all changes (git add .)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage. Default: ['.'] (all changes)",
                },
                "cwd": {"type": "string", "description": "Repository root directory"},
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cwd = params.get("cwd")
        paths: list[str] = params.get("paths", ["."])
        if not paths:
            paths = ["."]

        success, output = _run_git(["add"] + paths, cwd=cwd, runtime=self._runtime)
        if success:
            return ToolResult(success=True, output=f"Staged: {', '.join(paths)}")
        return ToolResult(success=False, output=output, error=output)


class GitCommitTool(BaseTool):
    """
    (see class docstring below)
    """

    def __init__(self, runtime: Runtime | None = None) -> None:
        from tools.runtime import LocalRuntime
        self._runtime = runtime or LocalRuntime()

    """
    Commit staged changes.

    params:
        message (str): commit message (required)
        cwd (str):     repository root directory
    """

    @property
    def name(self) -> str:
        return "git_commit"

    @property
    def description(self) -> str:
        return (
            "Commit staged changes with a message. "
            "Always run git_add before git_commit. "
            "Write a clear, descriptive commit message."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message (be descriptive)",
                },
                "cwd": {"type": "string", "description": "Repository root directory"},
            },
            "required": ["message"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cwd = params.get("cwd")
        message = params.get("message", "").strip()

        if not message:
            return ToolResult(
                success=False, output="", error="commit message is required"
            )

        success, output = _run_git(["commit", "-m", message], cwd=cwd, runtime=self._runtime)
        return ToolResult(
            success=success,
            output=output,
            error=None if success else output,
        )
