"""
tools/undo_tool.py

Snapshot-based, git-independent undo for agent file mutations.

- UndoManager: keeps a stack of file snapshots taken immediately *before* every
  file_write / file_edit, mirrored to an append-only JSONL sidecar file
  (same write-then-flush discipline as agent/event_log.py) so a crashed run can
  still be rolled back afterwards.
- UndoTool: the LLM-facing tool. Pops the most recent mutation(s) and restores
  them, letting the agent walk back a bad edit mid-task.

Never shells out to git, so this works on any directory — the `run` and `chat`
commands accept a --repo that is not necessarily a git work tree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutation — one snapshot
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A file's on-disk state captured immediately before a mutating tool call."""
    seq: int
    tool: str                     # "file_write" | "file_edit"
    path: str
    existed_before: bool
    content_before: str | None    # None when the file did not exist
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# UndoManager
# ---------------------------------------------------------------------------

class UndoManager:
    """
    Records file snapshots and restores them on demand.

    One instance is shared by FileWriteTool, FileEditTool and UndoTool, and lives
    as long as the tool registry does: one process for `run`, one whole session
    for `chat` (where the registry is reused across rounds).
    """

    def __init__(self, sidecar_path: Path | str | None = None) -> None:
        self._stack: list[Mutation] = []
        self._seq = 0
        self._sidecar_path = Path(sidecar_path) if sidecar_path else None
        self._file = None
        if self._sidecar_path is not None:
            self._sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._sidecar_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Write side — called by the file tools before they mutate
    # ------------------------------------------------------------------

    def snapshot(self, tool: str, path: str, *, existed_before: bool,
                 content_before: str | None) -> None:
        """Record one snapshot. Sidecar write failures degrade to in-memory only."""
        self._seq += 1
        mutation = Mutation(
            seq=self._seq,
            tool=tool,
            path=path,
            existed_before=existed_before,
            content_before=content_before,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._stack.append(mutation)
        if self._file is not None:
            try:
                self._file.write(json.dumps(mutation.to_dict(), ensure_ascii=False) + "\n")
                self._file.flush()
            except OSError as exc:
                logger.warning("undo sidecar write failed, continuing in memory: %s", exc)

    def snapshot_from_disk(self, tool: str, path: str) -> None:
        """Read the current content and snapshot it.

        Never raises: failing to snapshot must not block the write it protects.
        """
        try:
            p = Path(path)
            existed = p.exists()
            content = p.read_text(encoding="utf-8") if existed else None
        except OSError as exc:
            logger.warning("undo snapshot failed for %s, continuing: %s", path, exc)
            return
        self.snapshot(tool, path, existed_before=existed, content_before=content)

    # ------------------------------------------------------------------
    # Read / restore side
    # ------------------------------------------------------------------

    @property
    def stack(self) -> list[Mutation]:
        return list(self._stack)

    def pop(self, steps: int = 1, path: str | None = None) -> list[Mutation]:
        """
        Restore and discard the `steps` most recent mutations (newest first),
        optionally limited to one path. Returns what was actually reverted,
        which may be shorter than `steps` when the stack runs out.
        """
        reverted: list[Mutation] = []
        kept: list[Mutation] = []
        for mutation in reversed(self._stack):
            if len(reverted) < steps and (path is None or mutation.path == path):
                self._restore(mutation)
                reverted.append(mutation)
            else:
                kept.append(mutation)
        self._stack = list(reversed(kept))
        return reverted

    @staticmethod
    def _restore(mutation: Mutation) -> None:
        p = Path(mutation.path)
        if mutation.existed_before:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(mutation.content_before or "", encoding="utf-8")
        else:
            p.unlink(missing_ok=True)

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "UndoManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Static helpers — used by the CLI, which has no live manager
    # ------------------------------------------------------------------

    @staticmethod
    def load_sidecar(path: Path | str) -> list[Mutation]:
        """Read a .undo.jsonl file back into Mutations."""
        mutations: list[Mutation] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    mutations.append(Mutation(**json.loads(line)))
        return mutations

    @staticmethod
    def find_sidecar(log_dir: Path | str, task_id: str) -> Path | None:
        """Newest sidecar for a task_id, or None."""
        matches = sorted(
            Path(log_dir).glob(f"{task_id}_*.undo.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return matches[0] if matches else None

    @staticmethod
    def sidecar_for_log_path(jsonl_path: Path | str) -> Path | None:
        """Locate the sidecar belonging to an event-log .jsonl path."""
        p = Path(jsonl_path)
        return UndoManager.find_sidecar(p.parent, p.stem.split("_")[0])

    @staticmethod
    def rollback_plan(mutations: list[Mutation],
                      path: str | None = None) -> list[Mutation]:
        """
        One Mutation per distinct path: the *earliest* snapshot for that path.

        This is what "roll back the run" means — restore each file to how it was
        before the run first touched it, not merely undo its final edit. Differs
        deliberately from UndoTool's last-in-first-out behaviour.
        """
        earliest: dict[str, Mutation] = {}
        for m in mutations:
            if path is not None and m.path != path:
                continue
            if m.path not in earliest or m.seq < earliest[m.path].seq:
                earliest[m.path] = m
        return sorted(earliest.values(), key=lambda m: m.path)


# ---------------------------------------------------------------------------
# UndoTool — LLM-facing
# ---------------------------------------------------------------------------

class UndoTool(BaseTool):
    """
    Revert the agent's own most recent file change(s).

    params:
        steps (int): how many recent mutations to undo (default 1)
        path (str):  only undo mutations to this file (optional)
    """

    def __init__(self, manager: UndoManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "undo"

    @property
    def description(self) -> str:
        return (
            "Revert your most recent file_write/file_edit change(s), restoring the "
            "previous content (or deleting the file if it did not exist before). Use "
            "this when a change you just made turns out to be wrong — for example "
            "right after a test fails because of your last edit. Undoes the single "
            "most recent change by default; pass steps to undo more, or path to undo "
            "only changes to one file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of recent changes to undo (default 1)",
                },
                "path": {
                    "type": "string",
                    "description": "Only undo changes to this file (optional)",
                },
            },
            "required": [],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            steps = int(params.get("steps", 1))
        except (TypeError, ValueError):
            return ToolResult(success=False, output="", error="steps must be an integer")
        path = params.get("path")

        if steps < 1:
            return ToolResult(success=False, output="", error="steps must be >= 1")

        reverted = self._manager.pop(steps=steps, path=path)
        if not reverted:
            return ToolResult(
                success=False, output="",
                error=("Nothing to undo" + (f" for {path}" if path else "")
                       + " — no file changes have been made yet."),
            )

        lines = [
            f"Reverted {m.tool} on {m.path} "
            + ("(restored previous content)" if m.existed_before
               else "(deleted — it did not exist before)")
            for m in reverted
        ]
        return ToolResult(success=True, output="\n".join(lines))
