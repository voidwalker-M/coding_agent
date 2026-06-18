"""
agent/event_log.py

Append-only JSONL event log.
A complete record of the entire agent run, supporting:
- Real-time writes (each event is flushed to disk immediately)
- Deterministic replay (reconstruct the full event sequence)
- Per-task isolation (one independent file per run)
- Human-readable format (JSONL; can be inspected with cat / tail -f)

Design principles:
- Append-only; records are never modified after being written
- Flushed to disk after every write so recent events survive crashes
- Filenames include a timestamp so multiple runs never overwrite each other
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from agent.task import Event, EventType, Task, Action, Observation


# ---------------------------------------------------------------------------
# EventLog
# ---------------------------------------------------------------------------

class EventLog:
    """
    Append-only event log in JSONL format.

    Usage:
        log = EventLog.create(task, log_dir="./logs")
        log.log_task_start(task)
        log.log_action(step=1, action=action)
        log.log_observation(step=1, observation=obs)
        log.close()

    File path format:
        {log_dir}/{task_id}_{timestamp}.jsonl
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = open(path, "a", encoding="utf-8")  # append mode

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, task: Task, log_dir: str = "./logs") -> "EventLog":
        """
        Create an EventLog for a new run.
        The directory is created automatically if it does not exist.
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{task.task_id}_{timestamp}.jsonl"
        return cls(log_path / filename)

    @classmethod
    def open_existing(cls, path: str | Path) -> "EventLog":
        """Open an existing EventLog file for appending (e.g. resume from checkpoint)."""
        return cls(Path(path))

    # ------------------------------------------------------------------
    # Write methods (one semantic method per EventType)
    # ------------------------------------------------------------------

    def log_task_start(self, task: Task) -> None:
        """Task started."""
        self._append(Event(
            event_type=EventType.TASK_START,
            task_id=task.task_id,
            payload={"task": task.to_dict()},
        ))

    def log_action(self, step: int, action: Action, raw_content: str = "") -> None:
        """Each decision step of the agent. raw_content is the full raw text from the model."""
        self._append(Event(
            event_type=EventType.ACTION,
            task_id=self._current_task_id,
            payload={
                "step":        step,
                "action":      action.to_dict(),
                "raw_content": raw_content,  # raw model output including the full reasoning chain
            },
        ))

    def log_observation(self, step: int, observation: Observation) -> None:
        """Tool execution result."""
        self._append(Event(
            event_type=EventType.OBSERVATION,
            task_id=self._current_task_id,
            payload={
                "step":        step,
                "observation": observation.to_dict(),
            },
        ))

    def log_reflection(self, step: int, reason: str, prompt: str) -> None:
        """
        Record when a Reflection is triggered.
        reason: trigger cause ("test_failed" / "no_edit_n_steps")
        prompt: the reflection prompt injected into the LLM
        """
        self._append(Event(
            event_type=EventType.REFLECTION,
            task_id=self._current_task_id,
            payload={
                "step":   step,
                "reason": reason,
                "prompt": prompt,
            },
        ))

    def log_task_complete(self, steps: int, summary: str) -> None:
        """Task completed successfully."""
        self._append(Event(
            event_type=EventType.TASK_COMPLETE,
            task_id=self._current_task_id,
            payload={
                "steps":   steps,
                "summary": summary,
            },
        ))

    def log_task_failed(self, steps: int, reason: str) -> None:
        """Task failed or was circuit-broken."""
        self._append(Event(
            event_type=EventType.TASK_FAILED,
            task_id=self._current_task_id,
            payload={
                "steps":  steps,
                "reason": reason,
            },
        ))

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def replay(self) -> list[Event]:
        """
        Read all events from the beginning and reconstruct the full event sequence.
        Used for debugging and checkpoint-resume analysis. Callable after file is closed.
        """
        if not self._file.closed:
            self._file.flush()
        events: list[Event] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                events.append(Event(
                    event_id=raw["event_id"],
                    event_type=EventType(raw["event_type"]),
                    task_id=raw["task_id"],
                    timestamp=raw["timestamp"],
                    payload=raw["payload"],
                ))
        return events

    def iter_events(self) -> Iterator[Event]:
        """Lazily iterate over all events; suitable for large files. Callable after file is closed."""
        if not self._file.closed:
            self._file.flush()
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                yield Event(
                    event_id=raw["event_id"],
                    event_type=EventType(raw["event_type"]),
                    task_id=raw["task_id"],
                    timestamp=raw["timestamp"],
                    payload=raw["payload"],
                )

    def get_actions(self) -> list[Action]:
        """
        Extract all Actions from the event log, used for loop detection.
        (Circuit-breaker fires when consecutive identical actions are detected.)
        """
        from agent.task import ActionType, ToolCall

        actions: list[Action] = []
        for event in self.iter_events():
            if event.event_type != EventType.ACTION:
                continue
            raw_action = event.payload["action"]
            raw_tc = raw_action.get("tool_call")
            tool_call = None
            if raw_tc:
                tool_call = ToolCall(
                    name=raw_tc["name"],
                    params=raw_tc["params"],
                )
            actions.append(Action(
                action_type=ActionType(raw_action["action_type"]),
                thought=raw_action["thought"],
                tool_call=tool_call,
                message=raw_action.get("message"),
            ))
        return actions

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def _current_task_id(self) -> str:
        """Extract the task_id (8-char prefix) from the filename."""
        return self._path.stem.split("_")[0]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _append(self, event: Event) -> None:
        """
        Write one event.
        Flushed to disk immediately after each write to prevent data loss on crash.
        """
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Explicitly close the file. Normally called when Agent.run() finishes."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"EventLog(path={self._path})"


# ---------------------------------------------------------------------------
# Helper: generate a summary from a completed log
# ---------------------------------------------------------------------------

def summarize_run(log: EventLog) -> dict:
    """
    Read a complete run's event log and return summary statistics.
    Used in analysis scripts; not part of the main agent flow.
    """
    events = log.replay()

    stats = {
        "total_events":    len(events),
        "actions":         0,
        "reflections":     0,
        "tool_calls":      {},   # tool_name -> count
        "observations_ok": 0,
        "observations_err": 0,
        "final_status":    None,
    }

    for event in events:
        if event.event_type == EventType.ACTION:
            stats["actions"] += 1
            tc = event.payload["action"].get("tool_call")
            if tc:
                name = tc["name"]
                stats["tool_calls"][name] = stats["tool_calls"].get(name, 0) + 1

        elif event.event_type == EventType.OBSERVATION:
            obs = event.payload["observation"]
            if obs["status"] == "success":
                stats["observations_ok"] += 1
            else:
                stats["observations_err"] += 1

        elif event.event_type == EventType.REFLECTION:
            stats["reflections"] += 1

        elif event.event_type in (EventType.TASK_COMPLETE, EventType.TASK_FAILED):
            stats["final_status"] = event.event_type.value

    return stats
