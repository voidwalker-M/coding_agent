"""
agent/checkpoint.py

Checkpoint / resume for long agent runs.

Each checkpoint is a JSON file capturing conversation history and run counters so
a task can continue after process crash, API timeout, or explicit interrupt.
Checkpoints are written after every completed step (action + observation).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.task import Task

CHECKPOINT_VERSION = 1


@dataclass
class RunCheckpoint:
    """Serializable agent run state for resume."""

    version: int
    task: dict[str, Any]
    step: int
    total_tokens: int
    llm_time_total: float
    tool_time_total: float
    steps_without_edit: int
    noop_finish_rejections: int
    history: list[dict[str, str]]
    short_term: dict[str, Any] | None
    status: str                         # "running" | "success" | "failed" | "interrupted"
    log_path: str
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        step: int,
        total_tokens: int,
        llm_time_total: float,
        tool_time_total: float,
        steps_without_edit: int,
        noop_finish_rejections: int,
        history: list[dict[str, str]],
        short_term: dict[str, Any] | None,
        log_path: str | Path,
        status: str = "running",
    ) -> "RunCheckpoint":
        return cls(
            version=CHECKPOINT_VERSION,
            task=task.to_dict(),
            step=step,
            total_tokens=total_tokens,
            llm_time_total=llm_time_total,
            tool_time_total=tool_time_total,
            steps_without_edit=steps_without_edit,
            noop_finish_rejections=noop_finish_rejections,
            history=history,
            short_term=short_term,
            status=status,
            log_path=str(log_path),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunCheckpoint":
        return cls(
            version=int(data.get("version", 1)),
            task=data["task"],
            step=int(data["step"]),
            total_tokens=int(data.get("total_tokens", 0)),
            llm_time_total=float(data.get("llm_time_total", 0.0)),
            tool_time_total=float(data.get("tool_time_total", 0.0)),
            steps_without_edit=int(data.get("steps_without_edit", 0)),
            noop_finish_rejections=int(data.get("noop_finish_rejections", 0)),
            history=list(data.get("history", [])),
            short_term=data.get("short_term"),
            status=str(data.get("status", "running")),
            log_path=str(data["log_path"]),
            updated_at=str(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
        )


def checkpoint_path_for(task_id: str, checkpoint_dir: str | Path) -> Path:
    """Stable checkpoint filename for a task (overwritten each step)."""
    base = Path(checkpoint_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{task_id}.checkpoint.json"


def save_checkpoint(cp: RunCheckpoint, checkpoint_dir: str | Path) -> Path:
    """Atomically persist a checkpoint (write temp + rename)."""
    path = checkpoint_path_for(cp.task["task_id"], checkpoint_dir)
    tmp = path.with_suffix(".tmp")
    cp.updated_at = datetime.now(timezone.utc).isoformat()
    tmp.write_text(json.dumps(cp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path) -> RunCheckpoint:
    """Load a checkpoint file; raises FileNotFoundError / ValueError on bad input."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version", 1) != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {data.get('version')}")
    return RunCheckpoint.from_dict(data)


def short_term_to_dict(stm) -> dict[str, Any] | None:
    """Serialize ShortTermMemory when present."""
    if stm is None:
        return None
    if hasattr(stm, "to_state"):
        return stm.to_state()
    return {
        "notes": list(getattr(stm, "_notes", [])),
        "files": list(getattr(stm, "_files", [])),
        "summary": getattr(stm, "_summary", ""),
    }


def short_term_from_dict(data: dict[str, Any] | None):
    """Restore ShortTermMemory from checkpoint payload."""
    if not data:
        return None
    from context.memory import ShortTermMemory
    stm = ShortTermMemory(window_queries=int(data.get("window_queries") or 10))
    if hasattr(stm, "load_state"):
        stm.load_state(data)
        return stm
    stm._notes = list(data.get("notes", []))
    stm._files = list(data.get("files", []))
    stm._summary = str(data.get("summary", ""))
    return stm
