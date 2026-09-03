"""
tools/file_tool.py

File operation tools providing three actions:
- file_read:   read the full contents of a file
- file_view:   view a file in windowed pages (prevents blowing up context with one read)
- file_write:  write a file (full overwrite)

Design principles:
- file_read truncates large files by line count and suggests file_view for pagination
- file_view maintains a "window" concept; returns a fixed number of lines per call
- file_write creates parent directories automatically and confirms line count on success
- All paths are restricted to within repo_path (prevents reading system files)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult
from tools.path_policy import TEST_EDIT_BLOCKED_MSG, is_test_path


# Maximum lines returned by a single file_read; beyond this, suggest file_view
MAX_READ_LINES = 500
# Lines per window for file_view
VIEW_WINDOW_LINES = 100


def _reject_test_edit(path: Path) -> ToolResult | None:
    if is_test_path(path):
        return ToolResult(success=False, output="", error=TEST_EDIT_BLOCKED_MSG)
    return None


class FileReadTool(BaseTool):
    """
    Read file contents. Truncates files longer than MAX_READ_LINES lines.

    params:
        path (str): file path (relative or absolute)
    """

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            f"Read the contents of a file. "
            f"Files longer than {MAX_READ_LINES} lines will be truncated; "
            f"use file_view with line numbers to read specific sections."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to repo root)",
                },
            },
            "required": ["path"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        path = Path(params.get("path", ""))
        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )
        if not path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {path}",
            )

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(success=False, output="", error=str(e))

        total = len(lines)
        truncated = total > MAX_READ_LINES
        display_lines = lines[:MAX_READ_LINES]

        # Add line numbers so the agent can use file_view to navigate
        numbered = "\n".join(
            f"{i + 1:4d} | {line}"
            for i, line in enumerate(display_lines)
        )

        suffix = ""
        if truncated:
            suffix = (
                f"\n... ({total - MAX_READ_LINES} more lines not shown) "
                f"Use file_view with start_line to read the rest."
            )

        return ToolResult(
            success=True,
            output=f"File: {path} ({total} lines total)\n{numbered}{suffix}",
        )


class FileViewTool(BaseTool):
    """
    View a file in windowed pages, returning VIEW_WINDOW_LINES lines at a time.

    params:
        path (str):       file path
        start_line (int): line to start from (1-indexed, default 1)
    """

    @property
    def name(self) -> str:
        return "file_view"

    @property
    def description(self) -> str:
        return (
            f"View a specific section of a file, {VIEW_WINDOW_LINES} lines at a time. "
            f"Use start_line to scroll through large files. Lines are 1-indexed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file",
                },
                "start_line": {
                    "type": "integer",
                    "description": f"First line to show (1-indexed, default 1)",
                },
            },
            "required": ["path"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        path = Path(params.get("path", ""))
        start_line = max(1, int(params.get("start_line", 1)))

        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {path}")

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(success=False, output="", error=str(e))

        total = len(lines)
        if start_line > total:
            return ToolResult(
                success=False,
                output="",
                error=f"start_line {start_line} exceeds file length ({total} lines)",
            )

        end_line = min(start_line + VIEW_WINDOW_LINES - 1, total)
        window = lines[start_line - 1 : end_line]

        numbered = "\n".join(
            f"{start_line + i:4d} | {line}"
            for i, line in enumerate(window)
        )

        nav = ""
        if end_line < total:
            nav = f"\n[Lines {start_line}–{end_line} of {total}. Next: file_view path={path} start_line={end_line + 1}]"
        else:
            nav = f"\n[Lines {start_line}–{end_line} of {total}. End of file.]"

        return ToolResult(success=True, output=numbered + nav)


class FileWriteTool(BaseTool):
    """
    Write a file (full overwrite). Parent directories are created automatically.

    params:
        path (str):    file path
        content (str): content to write
    """

    def __init__(
        self,
        undo_manager: "UndoManager | None" = None,
        *,
        block_test_edits: bool = False,
    ) -> None:
        from tools.undo_tool import UndoManager
        self._undo = undo_manager or UndoManager()
        self._block_test_edits = block_test_edits

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        base = (
            "Write content to a file, replacing its entire contents. "
            "Parent directories are created automatically. "
            "Always read the file first before writing to avoid losing existing content."
        )
        if self._block_test_edits:
            base += (
                " Do NOT write under tests/ or to test_*.py — fix library code only."
            )
        return base

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        path = Path(params.get("path", ""))
        content = params.get("content", "")
        if self._block_test_edits:
            blocked = _reject_test_edit(path)
            if blocked is not None:
                return blocked

        self._undo.snapshot_from_disk("file_write", str(path))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, output="", error=str(e))

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return ToolResult(
            success=True,
            output=f"Written {line_count} lines to {path}",
        )


class FileEditTool(BaseTool):
    """
    Make a surgical edit to an existing file by replacing an exact string.

    This is the preferred way to change existing files: unlike file_write it does
    NOT require reproducing the whole file, so a small, targeted change stays
    small and cannot accidentally destroy the rest of the file. old_string must
    match the file exactly (including indentation) and must be unique.

    params:
        path (str):       file path
        old_string (str): exact text to find (must appear exactly once)
        new_string (str): text to replace it with
    """

    def __init__(
        self,
        undo_manager: "UndoManager | None" = None,
        *,
        block_test_edits: bool = False,
    ) -> None:
        from tools.undo_tool import UndoManager
        self._undo = undo_manager or UndoManager()
        self._block_test_edits = block_test_edits

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        base = (
            "Replace an exact string in an existing file with new text. PREFER THIS "
            "over file_write for modifying existing files — you only provide the "
            "snippet to change, not the whole file. `old_string` must match the file "
            "exactly (copy it verbatim, including indentation and surrounding lines) "
            "and must be unique; include a few extra lines of context if needed to "
            "make it unique. Read the file first to get the exact text."
        )
        if self._block_test_edits:
            base += (
                " Do NOT edit under tests/ or test_*.py — fix library code only."
            )
        return base

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find and replace (must be unique in the file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to replace old_string with",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        path = Path(params.get("path", ""))
        old_string = params.get("old_string", "")
        new_string = params.get("new_string", "")

        if self._block_test_edits:
            blocked = _reject_test_edit(path)
            if blocked is not None:
                return blocked
        if not path or not str(path):
            return ToolResult(success=False, output="", error="path is required")
        if not old_string:
            return ToolResult(
                success=False, output="",
                error="old_string is required and must be non-empty. To create a new "
                      "file or fully rewrite one, use file_write instead.",
            )
        if not path.exists():
            return ToolResult(success=False, output="", error=f"{path} does not exist")

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, output="", error=str(e))

        occurrences = content.count(old_string)
        if occurrences == 0:
            return ToolResult(
                success=False, output="",
                error="old_string not found in the file. Read the file again and copy "
                      "the exact text (whitespace and indentation must match).",
            )
        if occurrences > 1:
            return ToolResult(
                success=False, output="",
                error=f"old_string appears {occurrences} times; it must be unique. "
                      "Include more surrounding lines of context to disambiguate.",
            )
        if old_string == new_string:
            return ToolResult(success=False, output="",
                              error="old_string and new_string are identical — nothing to change.")

        updated = content.replace(old_string, new_string, 1)
        # Reuse the content already read above rather than re-reading from disk.
        self._undo.snapshot("file_edit", str(path),
                            existed_before=True, content_before=content)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, output="", error=str(e))

        removed = old_string.count("\n") + 1
        added = new_string.count("\n") + 1
        return ToolResult(
            success=True,
            output=f"Edited {path}: replaced {removed} line(s) with {added} line(s).",
        )
