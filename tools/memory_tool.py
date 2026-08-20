"""
tools/memory_tool.py

Agent-facing memory tools (feature #2), bound to a LongTermMemory (and optionally
a ShortTermMemory scratchpad):

    remember — persist a fact/preference/insight for future sessions
    recall   — search past memories for anything relevant to a query

These mirror how Claude Code exposes memory to itself: an explicit "remember this"
path plus on-demand recall. Automatic episodic capture at end-of-run is handled in
agent/core.py; these tools are the model's *deliberate* memory controls.
"""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool, ToolResult

# Kinds the model is allowed to write (a subset of context.memory.KINDS that make
# sense for self-authored notes; episodic is reserved for automatic capture).
_WRITABLE_KINDS = ("semantic", "reference", "project", "user", "feedback")


class RememberTool(BaseTool):
    """Persist a durable memory (fact, preference, gotcha) for future sessions."""

    def __init__(self, long_term, short_term=None) -> None:
        self._ltm = long_term
        self._stm = short_term

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Save a durable memory for future sessions — a repo convention, a "
            "user preference, or a hard-won insight worth not rediscovering. "
            "Use sparingly for facts that will still matter next time; do NOT "
            "use it as a scratchpad for the current task."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory to store (one clear fact/insight)."},
                "kind": {
                    "type": "string",
                    "enum": list(_WRITABLE_KINDS),
                    "description": "reference=repo navigation/convention, project=background, "
                                   "user=preference, feedback=working constraint, semantic=general fact.",
                },
                "description": {"type": "string", "description": "Optional one-line summary for the memory index."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional topic tags."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Optional related file paths."},
            },
            "required": ["text"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        text = (params.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, output="", error="`text` is required")
        kind = params.get("kind") or "semantic"
        if kind not in _WRITABLE_KINDS:
            kind = "semantic"
        try:
            rec = self._ltm.remember(
                text,
                kind=kind,
                description=(params.get("description") or "").strip(),
                tags=list(params.get("tags") or []),
                files=list(params.get("files") or []),
                source="agent",
            )
        except Exception as exc:                       # never break the agent loop
            return ToolResult(success=False, output="", error=f"memory write failed: {exc}")
        if self._stm is not None:
            self._stm.add_note(f"Remembered ({kind}): {rec.description or text[:60]}")
        return ToolResult(success=True, output=f"Stored memory '{rec.name}' ({kind}).")


class RecallTool(BaseTool):
    """Search long-term memory for records relevant to a query."""

    def __init__(self, long_term) -> None:
        self._ltm = long_term

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return (
            "Search memory from past sessions for anything relevant to a query "
            "(repo conventions, prior solutions, user preferences). Returns the "
            "most relevant stored memories."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up in memory."},
                "k": {"type": "integer", "description": "How many memories to return (default 5)."},
            },
            "required": ["query"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        query = (params.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output="", error="`query` is required")
        k = int(params.get("k") or 5)
        try:
            block = self._ltm.recall(query, k=max(1, min(k, 10)))
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"memory recall failed: {exc}")
        if not block:
            return ToolResult(success=True, output="No relevant memories found.")
        return ToolResult(success=True, output=block)
