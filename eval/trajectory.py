"""
eval/trajectory.py

Process-level metrics from an EventLog, used to compare engines/policies
beyond pass@k.

The coding-eval harness already grades *outcome* (did tests pass?). These
metrics grade *how* the agent got there:

- tool_errors / tool_error_rate — failed tool calls
- duplicate_actions — consecutive identical tool calls (wasted loop)
- reflections — policy nudges (test-failed / no-edit / noop-finish)
- time_to_first_edit — seconds of logged tool time until the first write
"""

from __future__ import annotations

from agent.event_log import EventLog
from agent.task import EventType
from context.workspace_sync import EDIT_TOOLS


def trajectory_metrics(log: EventLog) -> dict:
    """Compute process metrics from a completed (or in-progress) event log."""
    events = log.replay()

    tool_ok = tool_err = 0
    reflections = 0
    duplicate_actions = 0
    time_to_first_edit: float | None = None
    tool_elapsed_until_edit = 0.0
    saw_edit = False
    last_call: tuple[str, str] | None = None  # (name, params-repr)

    for event in events:
        if event.event_type == EventType.ACTION:
            tc = event.payload.get("action", {}).get("tool_call")
            if tc:
                key = (tc.get("name") or "", repr(tc.get("params") or {}))
                if last_call is not None and key == last_call:
                    duplicate_actions += 1
                last_call = key
            else:
                last_call = None

        elif event.event_type == EventType.OBSERVATION:
            obs = event.payload.get("observation") or {}
            if obs.get("status") == "success":
                tool_ok += 1
            else:
                tool_err += 1
            dur = event.payload.get("duration_ms")
            secs = (dur / 1000.0) if dur is not None else 0.0
            if not saw_edit:
                tool_elapsed_until_edit += secs
                if obs.get("tool_name") in EDIT_TOOLS:
                    saw_edit = True
                    time_to_first_edit = round(tool_elapsed_until_edit, 4)

        elif event.event_type == EventType.REFLECTION:
            reflections += 1

    total_obs = tool_ok + tool_err
    return {
        "tool_ok": tool_ok,
        "tool_errors": tool_err,
        "tool_error_rate": round(tool_err / total_obs, 4) if total_obs else 0.0,
        "duplicate_actions": duplicate_actions,
        "reflections": reflections,
        "time_to_first_edit": time_to_first_edit,
        "edited": saw_edit,
    }


def merge_trajectory(acc: dict | None, piece: dict) -> dict:
    """Sum/min-merge trajectory dicts across pass@k attempts."""
    if acc is None:
        return dict(piece)
    out = dict(acc)
    for key in ("tool_ok", "tool_errors", "duplicate_actions", "reflections"):
        out[key] = acc.get(key, 0) + piece.get(key, 0)
    total = out["tool_ok"] + out["tool_errors"]
    out["tool_error_rate"] = round(out["tool_errors"] / total, 4) if total else 0.0
    out["edited"] = bool(acc.get("edited")) or bool(piece.get("edited"))
    times = [
        t for t in (acc.get("time_to_first_edit"), piece.get("time_to_first_edit"))
        if t is not None
    ]
    out["time_to_first_edit"] = round(min(times), 4) if times else None
    return out
