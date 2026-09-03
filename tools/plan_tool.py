"""
tools/plan_tool.py

Agent-facing checklist for structured task planning.

    plan create  — replace the plan with a numbered list of steps
    plan add     — append a step
    plan start   — mark a step in progress
    plan complete / skip — close a step
    plan list    — dump the current plan

The Plan object is shared with Agent so remaining steps are injected into the
system prompt on the next turn.
"""

from __future__ import annotations

from typing import Any

from agent.plan import Plan, StepStatus
from tools.base import BaseTool, ToolResult


class PlanTool(BaseTool):
    """Create and update the in-run task plan."""

    def __init__(self, plan: Plan | None = None) -> None:
        self.plan = plan if plan is not None else Plan()

    @property
    def name(self) -> str:
        return "plan"

    @property
    def description(self) -> str:
        return (
            "Maintain a numbered task plan. Use `create` after exploring so later "
            "steps stay visible; `complete` a step when the corresponding edit or "
            "verification is done. Prefer this over keeping the plan only in thought."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "add", "start", "complete", "skip", "list"],
                    "description": "create=replace plan, add=append a step, "
                                   "start/complete/skip=update one step, list=show plan.",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step titles (required for create).",
                },
                "title": {
                    "type": "string",
                    "description": "Step title (required for add).",
                },
                "step_id": {
                    "type": "integer",
                    "description": "Step id (required for start/complete/skip).",
                },
                "goal": {
                    "type": "string",
                    "description": "Optional one-line goal stored with the plan.",
                },
                "detail": {
                    "type": "string",
                    "description": "Optional note attached to the updated step.",
                },
            },
            "required": ["action"],
        }

    def execute(self, params: dict[str, Any]) -> ToolResult:
        action = (params.get("action") or "").strip().lower()
        try:
            if action == "create":
                steps = params.get("steps") or []
                if not isinstance(steps, list) or not steps:
                    return ToolResult(
                        success=False, output="",
                        error="`steps` (non-empty list of titles) is required for create",
                    )
                self.plan.replace(steps, goal=params.get("goal") or "")
            elif action == "add":
                self.plan.add(params.get("title") or "", detail=params.get("detail") or "")
            elif action == "start":
                self.plan.set_status(
                    int(params["step_id"]), StepStatus.IN_PROGRESS,
                    detail=params.get("detail") or "",
                )
            elif action == "complete":
                self.plan.set_status(
                    int(params["step_id"]), StepStatus.DONE,
                    detail=params.get("detail") or "",
                )
            elif action == "skip":
                self.plan.set_status(
                    int(params["step_id"]), StepStatus.SKIPPED,
                    detail=params.get("detail") or "",
                )
            elif action == "list":
                pass
            else:
                return ToolResult(
                    success=False, output="",
                    error=f"unknown action {action!r}; use create|add|start|complete|skip|list",
                )
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(success=True, output=self.plan.render())
