"""
tools/search_tool.py

Code search tools providing three actions:
- search_text:   search file contents for a string (grep-style)
- find_files:    find files by filename pattern
- find_symbol:   find function/class definitions in Python files (regex, no tree-sitter dependency)

Design notes:
- No dependency on external tools (grep may not be available); implemented in pure Python
- find_symbol uses regex to match def/class; can be replaced with tree-sitter for precision
- Result counts are capped to prevent blowing up the context window
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult


MAX_RESULTS = 50        # maximum results returned per search
MAX_LINE_LENGTH = 200   # truncate display of lines that exceed this length

# Directories to skip during search
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build", "*.egg-info",
})


class SearchTextTool(BaseTool):
    """
    Search for text in repo files; returns matching lines with context.

    params:
        pattern (str):       search string (regex supported)
        path (str):          search scope (file or directory, default current directory)
        file_pattern (str):  only search files matching this name pattern (e.g. "*.py")
        case_sensitive (bool): whether the search is case-sensitive (default True)
    """

    @property
    def name(self) -> str:
        return "search_text"

    @property
    def description(self) -> str:
        return (
            "Search for a text pattern (regex supported) in files. "
            "Returns matching lines with file path and line number. "
            f"Returns at most {MAX_RESULTS} matches."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: current directory)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py'). Default: all files",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search (default true)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        raw_pattern = params.get("pattern", "")
        search_path = Path(params.get("path", "."))
        file_pattern = params.get("file_pattern", "*")
        case_sensitive = params.get("case_sensitive", True)

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(raw_pattern, flags)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")

        if not search_path.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {search_path}"
            )

        matches: list[str] = []
        files = _iter_files(search_path, file_pattern)

        for filepath in files:
            if len(matches) >= MAX_RESULTS:
                break
            try:
                for lineno, line in enumerate(
                    filepath.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if regex.search(line):
                        display_line = line[:MAX_LINE_LENGTH]
                        if len(line) > MAX_LINE_LENGTH:
                            display_line += " ..."
                        matches.append(f"{filepath}:{lineno}: {display_line}")
                        if len(matches) >= MAX_RESULTS:
                            break
            except OSError:
                continue

        if not matches:
            return ToolResult(
                success=True,
                output=f"No matches found for '{raw_pattern}'",
            )

        suffix = f"\n[Showing {len(matches)} matches]"
        if len(matches) == MAX_RESULTS:
            suffix = f"\n[Showing first {MAX_RESULTS} matches, there may be more]"

        return ToolResult(success=True, output="\n".join(matches) + suffix)


class FindFilesTool(BaseTool):
    """
    Find files by filename pattern.

    params:
        pattern (str): glob-style filename pattern (e.g. "*.py", "test_*.py")
        path (str):    root directory to search (default current directory)
    """

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find files by name pattern (glob style). "
            "Example: pattern='test_*.py' finds all test files. "
            f"Returns at most {MAX_RESULTS} results."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g. '*.py', 'conftest.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in (default: current directory)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern", "")
        search_path = Path(params.get("path", "."))

        if not search_path.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {search_path}"
            )

        results: list[str] = []
        for filepath in _iter_files(search_path, pattern):
            results.append(str(filepath))
            if len(results) >= MAX_RESULTS:
                break

        if not results:
            return ToolResult(
                success=True,
                output=f"No files found matching '{pattern}' in {search_path}",
            )

        suffix = ""
        if len(results) == MAX_RESULTS:
            suffix = f"\n[Showing first {MAX_RESULTS} results]"

        return ToolResult(
            success=True,
            output="\n".join(results) + suffix,
        )


class FindSymbolTool(BaseTool):
    """
    Find function/class definitions in Python files.
    Uses regex to match def / class statements; can be replaced with tree-sitter for precision.

    params:
        symbol (str): function or class name (partial match supported)
        path (str):   root directory to search (default current directory)
    """

    @property
    def name(self) -> str:
        return "find_symbol"

    @property
    def description(self) -> str:
        return (
            "Find function or class definitions in Python files. "
            "Searches for 'def symbol' or 'class symbol' patterns. "
            "Supports partial name matching."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Function or class name to find (partial match supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in (default: current directory)",
                },
            },
            "required": ["symbol"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        symbol = params.get("symbol", "")
        search_path = Path(params.get("path", "."))

        if not symbol:
            return ToolResult(success=False, output="", error="symbol is required")

        # Match "def foo" / "class Foo" (including indented forms for methods)
        pattern = re.compile(
            rf"^(\s*)(def|class)\s+({re.escape(symbol)}\w*)\s*[:(]",
            re.MULTILINE,
        )

        matches: list[str] = []
        for filepath in _iter_files(search_path, "*.py"):
            if len(matches) >= MAX_RESULTS:
                break
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                for m in pattern.finditer(content):
                    lineno = content[: m.start()].count("\n") + 1
                    kind = m.group(2)   # def / class
                    name = m.group(3)
                    indent = len(m.group(1))
                    scope = "method" if indent > 0 else "top-level"
                    matches.append(
                        f"{filepath}:{lineno}: {kind} {name} ({scope})"
                    )
                    if len(matches) >= MAX_RESULTS:
                        break
            except OSError:
                continue

        if not matches:
            return ToolResult(
                success=True,
                output=f"No definition found for '{symbol}'",
            )

        return ToolResult(success=True, output="\n".join(matches))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _iter_files(root: Path, glob_pattern: str):
    """Recursively walk a directory, skip _SKIP_DIRS, and filter by glob_pattern."""
    if root.is_file():
        yield root
        return

    for filepath in sorted(root.rglob(glob_pattern)):
        if any(part in _SKIP_DIRS for part in filepath.parts):
            continue
        if filepath.is_file():
            yield filepath
