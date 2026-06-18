"""
tools/test_tool.py

pytest execution tool; returns structured test results.

Key design decisions:
- Returns parsed failure information rather than raw stdout
- On failure, output contains a compact failure summary to avoid flooding context with full tracebacks
- Success/failure determined by exit code, not string matching
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult
from tools.runtime import LocalRuntime, Runtime


PYTEST_TIMEOUT = 120        # longer than the shell tool default; tests can take time
MAX_OUTPUT_CHARS = 6_000    # test output tends to be long


class PytestTool(BaseTool):
    """
    Run pytest and return a structured result.

    params:
        path (str):  test file or directory (default "tests/", falls back to "." if not found)
        args (str):  extra pytest arguments (e.g. "-x -v --tb=short")
        cwd (str):   working directory (default current directory)
    """

    def __init__(self, runtime: Runtime | None = None) -> None:
        self._runtime = runtime or LocalRuntime()

    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return (
            "Run pytest and return a structured summary of results. "
            "Shows which tests failed and their error messages. "
            "Use path to run specific test files or directories."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Test file or directory to run (default: 'tests/' or '.')",
                },
                "args": {
                    "type": "string",
                    "description": "Extra pytest arguments (e.g. '-x -v --tb=short')",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory",
                },
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        cwd = params.get("cwd", None)
        cwd_path = Path(cwd) if cwd else Path.cwd()

        test_path = params.get("path", "")
        if not test_path:
            if (cwd_path / "tests").exists():
                test_path = "tests/"
            else:
                test_path = "."

        extra_args = params.get("args", "")

        # --tb=short gives enough detail for the agent; --no-header reduces noise
        cmd_parts = [
            "python", "-m", "pytest",
            test_path,
            "--tb=short",
            "--no-header",
            "-q",               # quiet: only show failures and final summary
        ]
        if extra_args:
            cmd_parts.extend(extra_args.split())

        cmd_str = " ".join(cmd_parts)
        run_result = self._runtime.exec(cmd_str, cwd=cwd, timeout=PYTEST_TIMEOUT)
        if "timed out" in run_result.stderr.lower():
            return ToolResult(
                success=False,
                output="",
                error=f"pytest timed out after {PYTEST_TIMEOUT}s",
            )
        raw = run_result.output
        success = run_result.returncode == 0

        output = _format_pytest_output(raw, success)

        return ToolResult(
            success=success,
            output=output,
            error=None if success else f"pytest exited with code {run_result.returncode}",
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_pytest_output(raw: str, success: bool) -> str:
    """
    Format raw pytest output into an agent-friendly summary.

    On success: return the passing statistics line (e.g. "5 passed in 0.12s").
    On failure: extract the FAILED test list and each failure's short traceback.
    """
    if len(raw) > MAX_OUTPUT_CHARS:
        # On failure the agent needs the tail most (error summary); head (collection info) is less important
        raw = "...[output truncated]...\n" + raw[-MAX_OUTPUT_CHARS:]

    if success:
        lines = raw.strip().splitlines()
        summary_lines = [l for l in lines if re.search(r"passed|no tests", l)]
        if summary_lines:
            return summary_lines[-1]
        return raw.strip()

    # On failure: extract FAILED lines
    failed_lines = [l for l in raw.splitlines() if l.startswith("FAILED")]
    failed_section = "\n".join(failed_lines) if failed_lines else ""

    # Extract "short test summary info" block (printed by pytest -q)
    short_summary_match = re.search(
        r"=+ short test summary info =+(.*?)(?:=+|\Z)",
        raw,
        re.DOTALL,
    )
    short_summary = short_summary_match.group(1).strip() if short_summary_match else ""

    # Final statistics line (e.g. "2 failed, 3 passed in 0.45s")
    stat_match = re.search(r"\d+ (failed|error).*in \d+\.\d+s", raw)
    stat_line = stat_match.group(0) if stat_match else ""

    parts = []
    if failed_section:
        parts.append(f"Failed tests:\n{failed_section}")
    if short_summary and short_summary != failed_section:
        parts.append(f"Summary:\n{short_summary}")
    if stat_line:
        parts.append(stat_line)

    return "\n\n".join(parts) if parts else raw.strip()
