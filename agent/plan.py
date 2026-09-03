"""
agent/plan.py

Structured task plan the agent maintains while it works.

Planning used to live only in the system-prompt "Workflow" paragraph and in the
multi-agent planner's free-text finish message. A first-class Plan object lets
the model create, update, and complete numbered steps via the `plan` tool, and
lets the ReAct loop inject remaining work into the next prompt.

This is the coding-agent analogue of a todo list: the model is still autonomous
(it decides *how* to execute), but progress is visible and remaining work is
hard to forget after a long tool trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


_ACTIVE = (StepStatus.PENDING, StepStatus.IN_PROGRESS)


@dataclass
class PlanStep:
    id: int
    title: str
    status: StepStatus = StepStatus.PENDING
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "detail": self.detail,
        }

    def marker(self) -> str:
        return {
            StepStatus.PENDING: "[ ]",
            StepStatus.IN_PROGRESS: "[>]",
            StepStatus.DONE: "[x]",
            StepStatus.SKIPPED: "[-]",
        }[self.status]


@dataclass
class Plan:
    """Mutable numbered checklist shared with PlanTool and the system prompt."""

    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    _next_id: int = 1

    def remaining(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status in _ACTIVE]

    def is_empty(self) -> bool:
        return not self.steps

    def render(self) -> str:
        if not self.steps:
            return "(no plan yet — use the plan tool to create numbered steps)"
        lines = []
        if self.goal:
            lines.append(f"Goal: {self.goal}")
        for step in self.steps:
            extra = f" — {step.detail}" if step.detail else ""
            lines.append(f"{step.marker()} {step.id}. {step.title}{extra}")
        left = len(self.remaining())
        lines.append(f"Remaining: {left}/{len(self.steps)}")
        return "\n".join(lines)

    def render_for_prompt(self) -> str:
        """Empty string when unused so existing prompts stay unchanged."""
        if self.is_empty():
            return ""
        return (
            "## Current plan\n"
            f"{self.render()}\n"
            "Keep this plan current with the plan tool. Prefer completing the "
            "next pending step over open-ended exploration.\n"
        )

    def replace(self, titles: Iterable[str], goal: str = "") -> None:
        cleaned = [t.strip() for t in titles if str(t).strip()]
        if not cleaned:
            raise ValueError("plan must contain at least one non-empty step")
        if goal.strip():
            self.goal = goal.strip()
        self.steps = [
            PlanStep(id=i, title=title)
            for i, title in enumerate(cleaned, start=1)
        ]
        self._next_id = len(self.steps) + 1

    def add(self, title: str, detail: str = "") -> PlanStep:
        title = title.strip()
        if not title:
            raise ValueError("step title is required")
        step = PlanStep(id=self._next_id, title=title, detail=detail.strip())
        self._next_id += 1
        self.steps.append(step)
        return step

    def get(self, step_id: int) -> PlanStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def set_status(self, step_id: int, status: StepStatus, detail: str = "") -> PlanStep:
        step = self.get(step_id)
        if step is None:
            raise KeyError(f"no plan step with id={step_id}")
        step.status = status
        if detail.strip():
            step.detail = detail.strip()
        return step
