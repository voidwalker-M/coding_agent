"""
agent/task.py

The data foundation of the entire project. Defines all core concepts used
during an agent run: Task / ToolCall / Action / Observation / Event / RunResult

Design principles:
- All types use dataclass for type safety and IDE friendliness
- Every class can be serialized to JSON via asdict(), for use by EventLog
- Enum types are str enums so they serialize to human-readable strings
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Event types written to the event log. str enum serializes directly to a string."""
    TASK_START      = "task_start"
    ACTION          = "action"
    OBSERVATION     = "observation"
    REFLECTION      = "reflection"
    TASK_COMPLETE   = "task_complete"
    TASK_FAILED     = "task_failed"


class ActionType(str, Enum):
    """Action types the agent can produce."""
    TOOL_CALL   = "tool_call"    # call a tool
    REFLECTION  = "reflection"   # trigger self-reflection
    FINISH      = "finish"       # declare the task complete
    GIVE_UP     = "give_up"      # give up voluntarily when beyond capability


class ObservationStatus(str, Enum):
    """Status of a tool execution result."""
    SUCCESS = "success"
    ERROR   = "error"
    TIMEOUT = "timeout"


class RunStatus(str, Enum):
    """Final status of an entire agent run."""
    SUCCESS     = "success"
    FAILED      = "failed"
    MAX_STEPS   = "max_steps"    # hit the step limit
    GAVE_UP     = "gave_up"      # agent voluntarily gave up


# ---------------------------------------------------------------------------
# Task — input
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    Input descriptor for one agent run.
    Constructed by the entry layer (CLI / GitHub Issue) and passed to Agent.run().
    """
    # Required
    description: str            # task description in natural language
    repo_path: str              # local path to the target repository

    # Optional
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    issue_url: str | None = None        # GitHub issue URL (filled in automatic-fix mode)
    test_cmd: str | None = None         # command to run tests, e.g. "pytest tests/"
    max_steps: int = 40                 # maximum loop steps; circuit-breaker when exceeded
    budget_tokens: int = 80_000         # token budget for the entire run

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return f"Task(id={self.task_id!r}, desc={self.description[:60]!r})"


# ---------------------------------------------------------------------------
# ToolCall — payload of an Action
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    Concrete parameters when the agent decides to call a tool.
    Nested inside Action; not written to EventLog separately.
    """
    name: str                   # tool name, e.g. "shell", "file_read"
    params: dict[str, Any]      # tool parameters; each Tool defines its own schema

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Action — the agent's decision output
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    The decision produced by the LLM each step.
    Parsed and returned by LLMBackend; Agent Core executes it.
    """
    action_type: ActionType
    thought: str                            # LLM's reasoning chain; must be preserved for debugging
    tool_call: ToolCall | None = None       # non-null when action_type == TOOL_CALL
    message: str | None = None             # explanation when action_type == FINISH / GIVE_UP

    def to_dict(self) -> dict[str, Any]:
        d = {
            "action_type": self.action_type.value,
            "thought": self.thought,
            "message": self.message,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
        }
        return d

    def is_terminal(self) -> bool:
        """Whether this action is terminal (no further tool execution needed)."""
        return self.action_type in (ActionType.FINISH, ActionType.GIVE_UP)

    def __repr__(self) -> str:
        if self.tool_call:
            return f"Action({self.action_type.value}, tool={self.tool_call.name})"
        return f"Action({self.action_type.value})"


# ---------------------------------------------------------------------------
# Observation — tool execution result
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Result returned to the agent after a tool executes.
    The output field is injected into the LLM context for the next step.
    """
    status: ObservationStatus
    output: str                         # tool output, already truncated to a safe length
    tool_name: str                      # which tool produced this result
    tokens_used: int = 0                # estimated tokens consumed by this observation
    error: str | None = None            # error message when status == ERROR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_success(self) -> bool:
        return self.status == ObservationStatus.SUCCESS

    def __repr__(self) -> str:
        return (
            f"Observation(tool={self.tool_name}, "
            f"status={self.status.value}, "
            f"len={len(self.output)})"
        )


# ---------------------------------------------------------------------------
# Event — unified unit written to EventLog
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    A single record in the EventLog.
    All information is encapsulated here and written append-only to a JSONL file.

    payload content depends on event_type:
    - TASK_START:    {"task": Task.to_dict()}
    - ACTION:        {"step": int, "action": Action.to_dict()}
    - OBSERVATION:   {"step": int, "observation": Observation.to_dict()}
    - REFLECTION:    {"step": int, "reason": str, "prompt": str}
    - TASK_COMPLETE: {"steps": int, "summary": str}
    - TASK_FAILED:   {"steps": int, "reason": str}
    """
    event_type: EventType
    task_id: str
    payload: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":   self.event_id,
            "event_type": self.event_type.value,
            "task_id":    self.task_id,
            "timestamp":  self.timestamp,
            "payload":    self.payload,
        }


# ---------------------------------------------------------------------------
# RunResult — the final result of a complete run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Return value of Agent.run().
    Contains the final status, patch content (if any), and statistics.
    """
    task_id: str
    status: RunStatus
    summary: str                        # human-readable result summary
    steps_taken: int
    total_tokens: int = 0
    patch: str | None = None            # changes in git-diff format
    error: str | None = None            # reason when status == FAILED

    def is_success(self) -> bool:
        return self.status == RunStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"RunResult(status={self.status.value}, "
            f"steps={self.steps_taken}, "
            f"tokens={self.total_tokens})"
        )
