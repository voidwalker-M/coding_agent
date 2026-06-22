"""
agent/orchestrator.py

Multi-agent orchestration (#3): a planner → coder → reviewer pipeline.

## Why multi-agent?

A single agent carries one ever-growing context and one generic prompt. Splitting
the work into specialized roles buys four things:

1. **Context isolation** — each role gets a small, focused context (the planner
   never sees tool spew; the reviewer sees only the diff + tests). This *reduces*
   tokens per call and keeps each model on-task — directly complementing the
   token-efficiency work (#4).
2. **Specialization** — a reviewer prompted to *find problems* catches bugs the
   coder, biased toward "I'm done", misses. (OpenHands raised SWE-bench results
   exactly this way, with a separate critic model.)
3. **Least privilege** — the planner and reviewer get a READ-ONLY tool set; only
   the coder can edit or run shell. A planning step can't accidentally mutate the repo.
4. **Verification** — an independent review gate is more trustworthy than the
   coder's self-reported FINISH.

## Pipeline

    Planner (read-only)  →  produces a short plan
        ↓
    Coder (full tools)   →  implements the plan         ⟲ loop while REVISE
        ↓
    Reviewer (read-only + tests) → APPROVE / REVISE + feedback

Each role is a fresh `Agent` run with its own restricted registry and its own
EventLog (per-role auditability). Roles are coordinated through the task text
(plan and review feedback are threaded forward), giving real context isolation.

This is built entirely on the existing Agent/ToolRegistry — no core changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import RunResult, RunStatus, Task
from llm.base import LLMBackend
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Read-only tools: safe for planning and review (no edits, no shell).
READ_ONLY_TOOLS = (
    "file_read", "file_view", "find_files", "find_symbol",
    "search_text", "git_status", "git_diff",
)
# Reviewer may additionally run the test suite.
REVIEWER_EXTRA_TOOLS = ("pytest",)

_APPROVE_TOKEN = "APPROVE"
_REVISE_TOKEN = "REVISE"


@dataclass
class RoleResult:
    role: str
    status: str
    summary: str
    steps: int
    tokens: int


@dataclass
class OrchestratorResult:
    status: RunStatus
    plan: str
    approved: bool
    iterations: int
    summary: str
    total_steps: int = 0
    total_tokens: int = 0
    roles: list[RoleResult] = field(default_factory=list)
    patch: str | None = None

    def is_success(self) -> bool:
        return self.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# Role task prompts
# ---------------------------------------------------------------------------

def _planner_task(description: str) -> str:
    return (
        "You are the PLANNER. Do NOT edit any files — you only have read-only tools.\n"
        "Explore the repository as needed and produce a SHORT, concrete plan for this task:\n\n"
        f"{description}\n\n"
        "Then call finish with the plan as a numbered list of the specific edits to make "
        "(files + what changes). Keep it under ~10 lines."
    )


def _coder_task(description: str, plan: str, feedback: str) -> str:
    fb = f"\n\n## Reviewer feedback to address\n{feedback}\n" if feedback else ""
    return (
        "You are the CODER. Implement the task below by editing files and running tests.\n\n"
        f"## Task\n{description}\n\n"
        f"## Plan\n{plan}\n{fb}\n"
        "Make the minimal changes, run the tests, then call finish with a summary of what you changed."
    )


def _reviewer_task(description: str, plan: str) -> str:
    return (
        "You are the REVIEWER. You have read-only tools plus the ability to run tests.\n"
        "Do NOT edit files. Review whether the coder correctly completed this task:\n\n"
        f"## Task\n{description}\n\n## Plan that was followed\n{plan}\n\n"
        "Inspect the diff (git_diff), read changed files, and run the tests (pytest).\n"
        f"End your finish message with the single token {_APPROVE_TOKEN} if the change is "
        f"correct and tests pass, or {_REVISE_TOKEN} followed by a short list of required "
        "fixes if not."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Coordinates planner/coder/reviewer agents over a single task."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
        max_iterations: int = 2,
        on_role_start=None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._cfg = config or AgentConfig()
        self._max_iterations = max(1, max_iterations)
        self._on_role_start = on_role_start   # optional callback(role: str) for UIs

        self._readonly = registry.subset(READ_ONLY_TOOLS)
        self._reviewer_reg = registry.subset(READ_ONLY_TOOLS + REVIEWER_EXTRA_TOOLS)

    def _run_role(self, role: str, description: str, registry: ToolRegistry,
                  repo_path: str, log_dir: str) -> tuple[RoleResult, RunResult]:
        if self._on_role_start:
            self._on_role_start(role)
        logger.info("orchestrator: running role=%s", role)
        agent = Agent(self._backend, registry, self._cfg)
        task = Task(description=description, repo_path=repo_path,
                    max_steps=self._cfg.max_steps)
        with EventLog.create(task, log_dir=log_dir) as log:
            result = agent.run(task, log)
        role_result = RoleResult(
            role=role,
            status=result.status.value,
            summary=result.summary or "",
            steps=result.steps_taken,
            tokens=result.total_tokens,
        )
        return role_result, result

    def run(self, task: Task, log_dir: str = "./logs") -> OrchestratorResult:
        roles: list[RoleResult] = []
        total_steps = total_tokens = 0

        # 1. Plan (read-only).
        plan_role, _ = self._run_role(
            "planner", _planner_task(task.description), self._readonly,
            task.repo_path, log_dir)
        roles.append(plan_role)
        total_steps += plan_role.steps
        total_tokens += plan_role.tokens
        plan = plan_role.summary or "(planner produced no plan; proceed directly)"

        # 2/3. Code + review loop.
        approved = False
        feedback = ""
        final_code: RunResult | None = None
        iterations = 0
        for i in range(self._max_iterations):
            iterations = i + 1
            code_role, code_result = self._run_role(
                "coder", _coder_task(task.description, plan, feedback),
                self._registry, task.repo_path, log_dir)
            roles.append(code_role)
            total_steps += code_role.steps
            total_tokens += code_role.tokens
            final_code = code_result

            review_role, _ = self._run_role(
                "reviewer", _reviewer_task(task.description, plan),
                self._reviewer_reg, task.repo_path, log_dir)
            roles.append(review_role)
            total_steps += review_role.steps
            total_tokens += review_role.tokens

            verdict = review_role.summary.upper()
            if _APPROVE_TOKEN in verdict and _REVISE_TOKEN not in verdict:
                approved = True
                break
            feedback = review_role.summary

        status = RunStatus.SUCCESS if approved else RunStatus.GAVE_UP
        summary = (
            f"Multi-agent run: {'APPROVED' if approved else 'NOT approved'} after "
            f"{iterations} iteration(s).\nPlan:\n{plan}"
        )
        return OrchestratorResult(
            status=status,
            plan=plan,
            approved=approved,
            iterations=iterations,
            summary=summary,
            total_steps=total_steps,
            total_tokens=total_tokens,
            roles=roles,
            patch=(final_code.patch if final_code else None),
        )
