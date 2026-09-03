"""
agent/orchestrator.py

Multi-agent orchestration: configurable topologies over specialized roles.

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

## Topologies

    pipeline     Planner (read-only) → Coder ⟲ Reviewer     (default)
    pair         Coder ⟲ Reviewer  (skip planner)
    debate       two Planners propose; Coder implements both-informed plan ⟲ Reviewer
    autonomous   single full-tool Agent (structured plan lives on the plan tool)

Roles share a Scratchpad so later stages see prior summaries without merging
full EventLogs. Each role is still a fresh `Agent` run with its own EventLog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from pathlib import Path

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import RunResult, RunStatus, Task
from llm.base import LLMBackend
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

TOPOLOGIES = ("pipeline", "pair", "debate", "autonomous")

# Read-only tools: safe for planning and review (no edits, no shell).
# `plan` mutates an in-memory checklist, not the repo, so planners may use it.
READ_ONLY_TOOLS = (
    "file_read", "file_view", "find_files", "find_symbol",
    "search_text", "git_status", "git_diff", "plan",
    "skill", "web_fetch",
)
# Reviewer may additionally run the test suite. The SWE registry names the
# pytest wrapper `test`; unit tests sometimes register it as `pytest`.
REVIEWER_EXTRA_TOOLS = ("test", "pytest")

_APPROVE_TOKEN = "APPROVE"
_REVISE_TOKEN = "REVISE"


@dataclass
class Scratchpad:
    """Shared notes passed forward between roles (not a full conversation)."""

    entries: list[tuple[str, str]] = field(default_factory=list)

    def write(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.entries.append((role, text))

    def render(self, limit: int = 1200) -> str:
        if not self.entries:
            return ""
        chunks = ["## Shared scratchpad (prior roles)"]
        budget = limit
        for role, text in self.entries:
            snippet = text if len(text) <= budget else text[: max(0, budget - 3)] + "..."
            chunks.append(f"### {role}\n{snippet}")
            budget -= len(snippet)
            if budget <= 0:
                break
        return "\n".join(chunks)


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
    topology: str = "pipeline"

    def is_success(self) -> bool:
        return self.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# Role task prompts
# ---------------------------------------------------------------------------

def _with_scratchpad(body: str, scratch: Scratchpad) -> str:
    pad = scratch.render()
    return f"{body}\n\n{pad}" if pad else body


def _planner_task(description: str, variant: str = "") -> str:
    alt = (
        " Propose an *alternative* plan that a second planner would not have written — "
        "different files, a smaller change, or a different root cause."
        if variant == "alt" else ""
    )
    return (
        "You are the PLANNER. Do NOT edit any files — you only have read-only tools.\n"
        "Explore the repository as needed and produce a SHORT, concrete plan for this task:\n\n"
        f"{description}\n\n"
        f"{alt}"
        "Then call finish with the plan as a numbered list of the specific edits to make "
        "(files + what changes). Keep it under ~10 lines. Optionally use the plan tool "
        "to record the same steps."
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


def _autonomous_task(description: str) -> str:
    return (
        "You are an autonomous coding agent. Explore, plan (use the plan tool), "
        "edit, verify with tests, then finish.\n\n"
        f"## Task\n{description}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Coordinates specialized agents over a single task."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
        max_iterations: int = 2,
        on_role_start=None,
        topology: str = "pipeline",
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._cfg = config or AgentConfig()
        self._max_iterations = max(1, max_iterations)
        self._on_role_start = on_role_start   # optional callback(role: str) for UIs
        topo = (topology or "pipeline").lower().strip()
        if topo not in TOPOLOGIES:
            raise ValueError(f"unknown topology {topology!r}; expected one of {TOPOLOGIES}")
        self.topology = topo
        self.scratchpad = Scratchpad()

        self._readonly = registry.subset(READ_ONLY_TOOLS)
        self._reviewer_reg = registry.subset(READ_ONLY_TOOLS + REVIEWER_EXTRA_TOOLS)

    def _config_for_role(self, role: str) -> AgentConfig:
        """Planner/reviewer must be able to finish without editing the tree."""
        if role in ("planner", "planner_a", "planner_b"):
            return replace(
                self._cfg,
                require_edit_before_finish=False,
                require_test_before_finish=False,
            )
        if role == "reviewer":
            return replace(
                self._cfg,
                require_edit_before_finish=False,
                require_test_before_finish=True,
            )
        return self._cfg

    def _run_role(self, role: str, description: str, registry: ToolRegistry,
                  repo_path: str, log_dir: str) -> tuple[RoleResult, RunResult]:
        if self._on_role_start:
            self._on_role_start(role)
        logger.info("orchestrator: running role=%s topology=%s", role, self.topology)
        agent = Agent(self._backend, registry, self._config_for_role(role))
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
        self.scratchpad.write(role, role_result.summary)
        return role_result, result

    def _code_review_loop(
        self,
        task: Task,
        plan: str,
        log_dir: str,
        roles: list[RoleResult],
    ) -> tuple[bool, int, RunResult | None, list[RoleResult], int, int]:
        approved = False
        feedback = ""
        final_code: RunResult | None = None
        iterations = 0
        total_steps = total_tokens = 0
        for i in range(self._max_iterations):
            iterations = i + 1
            code_role, code_result = self._run_role(
                "coder",
                _with_scratchpad(_coder_task(task.description, plan, feedback), self.scratchpad),
                self._registry, task.repo_path, log_dir)
            roles.append(code_role)
            total_steps += code_role.steps
            total_tokens += code_role.tokens
            final_code = code_result

            review_role, _ = self._run_role(
                "reviewer",
                _with_scratchpad(_reviewer_task(task.description, plan), self.scratchpad),
                self._reviewer_reg, task.repo_path, log_dir)
            roles.append(review_role)
            total_steps += review_role.steps
            total_tokens += review_role.tokens

            verdict = review_role.summary.upper()
            if _APPROVE_TOKEN in verdict and _REVISE_TOKEN not in verdict:
                approved = True
                break
            feedback = review_role.summary
        return approved, iterations, final_code, roles, total_steps, total_tokens

    def run(self, task: Task, log_dir: str = "./logs") -> OrchestratorResult:
        if self.topology == "autonomous":
            return self._run_autonomous(task, log_dir)
        if self.topology == "debate":
            return self._run_debate(task, log_dir)
        if self.topology == "pair":
            return self._run_pair(task, log_dir)
        return self._run_pipeline(task, log_dir)

    def _finish(
        self,
        *,
        plan: str,
        approved: bool,
        iterations: int,
        roles: list[RoleResult],
        total_steps: int,
        total_tokens: int,
        final_code: RunResult | None,
    ) -> OrchestratorResult:
        status = RunStatus.SUCCESS if approved else RunStatus.GAVE_UP
        summary = (
            f"Multi-agent ({self.topology}): {'APPROVED' if approved else 'NOT approved'} after "
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
            topology=self.topology,
        )

    def _run_pipeline(self, task: Task, log_dir: str) -> OrchestratorResult:
        roles: list[RoleResult] = []
        plan_role, _ = self._run_role(
            "planner",
            _with_scratchpad(_planner_task(task.description), self.scratchpad),
            self._readonly, task.repo_path, log_dir)
        roles.append(plan_role)
        plan = plan_role.summary or "(planner produced no plan; proceed directly)"
        approved, iterations, final_code, roles, loop_steps, loop_tokens = (
            self._code_review_loop(task, plan, log_dir, roles)
        )
        return self._finish(
            plan=plan, approved=approved, iterations=iterations, roles=roles,
            total_steps=plan_role.steps + loop_steps,
            total_tokens=plan_role.tokens + loop_tokens,
            final_code=final_code,
        )

    def _run_pair(self, task: Task, log_dir: str) -> OrchestratorResult:
        plan = "(no planner; coder owns the approach)"
        approved, iterations, final_code, roles, steps, tokens = (
            self._code_review_loop(task, plan, log_dir, [])
        )
        return self._finish(
            plan=plan, approved=approved, iterations=iterations, roles=roles,
            total_steps=steps, total_tokens=tokens, final_code=final_code,
        )

    def _run_debate(self, task: Task, log_dir: str) -> OrchestratorResult:
        roles: list[RoleResult] = []
        a_role, _ = self._run_role(
            "planner_a",
            _with_scratchpad(_planner_task(task.description), self.scratchpad),
            self._readonly, task.repo_path, log_dir)
        roles.append(a_role)
        b_role, _ = self._run_role(
            "planner_b",
            _with_scratchpad(_planner_task(task.description, variant="alt"), self.scratchpad),
            self._readonly, task.repo_path, log_dir)
        roles.append(b_role)
        plan = (
            f"Plan A:\n{a_role.summary or '(empty)'}\n\n"
            f"Plan B:\n{b_role.summary or '(empty)'}\n\n"
            "Implement the stronger plan; say which one you chose and why."
        )
        approved, iterations, final_code, roles, loop_steps, loop_tokens = (
            self._code_review_loop(task, plan, log_dir, roles)
        )
        return self._finish(
            plan=plan, approved=approved, iterations=iterations, roles=roles,
            total_steps=a_role.steps + b_role.steps + loop_steps,
            total_tokens=a_role.tokens + b_role.tokens + loop_tokens,
            final_code=final_code,
        )

    def _run_autonomous(self, task: Task, log_dir: str) -> OrchestratorResult:
        role, result = self._run_role(
            "autonomous",
            _autonomous_task(task.description),
            self._registry, task.repo_path, log_dir)
        approved = result.is_success()
        return self._finish(
            plan=role.summary or "(autonomous run)",
            approved=approved,
            iterations=1,
            roles=[role],
            total_steps=role.steps,
            total_tokens=role.tokens,
            final_code=result,
        )


class OrchestratorAgent:
    """Duck-types Agent.run(task, log) so EvalHarness can drive a multi-agent topology."""

    def __init__(self, orch: Orchestrator) -> None:
        self._orch = orch

    def run(self, task: Task, log: EventLog) -> RunResult:
        result = self._orch.run(task, log_dir=str(Path(log.path).parent))
        return RunResult(
            task_id=task.task_id,
            status=result.status,
            summary=result.summary,
            steps_taken=result.total_steps,
            total_tokens=result.total_tokens,
            patch=result.patch,
        )
