"""
context/workspace_sync.py

Policy for keeping prompt caches aligned with the working tree after edits.

Disk is the source of truth. Repo-map / RAG prompt caches live for a whole
task unless a successful write lands; then the next LLM call drops those
caches and runs incremental RagRetriever.build / SymbolIndex.build (content
hash) plus a cheap RepoMap.build from disk.

This module only records *which* paths changed. The loops own the flush.
"""

from __future__ import annotations

from typing import Any, Iterable

# Tools that mutate the working tree. agent.decision and eval.trajectory import this.
EDIT_TOOLS = frozenset({"file_write", "file_edit", "edit", "undo"})

# Undo without a path can restore several files; treat the whole tree as dirty.
ANY_PATH = "*"

_PATH_KEYS = ("path", "file", "filename")


def path_from_edit(tool_name: str, params: dict[str, Any] | None) -> str | None:
    """Return the dirty path for an edit tool, or None if this is not an edit."""
    if tool_name not in EDIT_TOOLS:
        return None
    params = params or {}
    for key in _PATH_KEYS:
        value = params.get(key)
        if value:
            return str(value)
    if tool_name == "undo":
        return ANY_PATH
    return None


def record_edit(
    dirty: set[str],
    tool_name: str,
    params: dict[str, Any] | None,
    *,
    succeeded: bool,
) -> None:
    """Mark a path dirty after a successful edit. Failed tools do not count."""
    if not succeeded:
        return
    path = path_from_edit(tool_name, params)
    if path:
        dirty.add(path)


def note_for_prompt(paths: Iterable[str]) -> str:
    """Short STM note so the model knows indexes and conversation copies are stale."""
    items = sorted({p for p in paths if p})
    if not items:
        return ""
    if ANY_PATH in items:
        return (
            "Indexes refreshed after workspace edits (undo without a specific path). "
            "Re-read files from disk; copies in this conversation are stale."
        )
    listed = ", ".join(items)
    return (
        f"Indexes refreshed after edits to {listed}. "
        "Re-read those files from disk; copies in this conversation are stale."
    )
