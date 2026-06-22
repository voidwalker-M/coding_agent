"""
eval/harness.py

Evaluation harness: runs the agent on a set of verifiable tasks, aggregates
metrics, and generates a report.

Per-task workflow:
1. Set up initial files in an isolated temporary repo (setup_files / setup_dir)
2. git init (so git diff and git tools are available)
3. Construct a fresh Agent using agent_factory, then run(task)
4. Use an independent verifier to determine objective success (not the agent's self-reported FINISH)
5. Record passed / agent_status / steps / tokens / time

Design principles:
- agent_factory(spec) -> Agent: injected by the caller; easy to swap real backend vs MockBackend
- Verifiers are independent of the agent's tool layer: success is an objective re-run result
- Each task uses its own isolated temporary directory; report can be saved as JSON for cross-run comparison
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from agent.event_log import EventLog
from agent.task import RunResult, Task
from eval.verifiers import Verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Complete definition of one evaluation task."""
    id: str
    description: str                                   # task prompt given to the agent
    verify: Verifier                                   # independent grader
    setup_files: dict[str, str] = field(default_factory=dict)  # relpath -> content
    setup_dir: str | None = None                       # alternatively: copy an existing fixture directory
    max_steps: int = 20


@dataclass
class EvalResult:
    """Evaluation result for a single task.

    With pass@k (attempts > 1): `passed` is pass@k (any attempt passed),
    `pass_at_1` is whether the first attempt passed, `num_passed` counts passing
    attempts, and steps/tokens/elapsed/cost are summed across all attempts (the
    total compute spent to evaluate this task). For attempts == 1 these collapse
    to the original single-run semantics.
    """
    task_id: str
    passed: bool                # verifier judgment (objective ground truth); pass@k when attempts>1
    agent_status: str           # agent's self-reported final status
    steps: int
    tokens: int
    elapsed: float
    detail: str                 # verifier explanation
    error: str | None = None
    pass_at_1: bool = False     # whether the FIRST attempt passed
    attempts: int = 1
    num_passed: int = 0         # how many of `attempts` passed
    cost_usd: float = 0.0       # estimated $ cost (blended; 0 if tokens unknown)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "pass_at_1": self.pass_at_1,
            "attempts": self.attempts,
            "num_passed": self.num_passed,
            "agent_status": self.agent_status,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed": round(self.elapsed, 2),
            "detail": self.detail,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregated report for the entire evaluation suite."""
    results: list[EvalResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def success_rate(self) -> float:
        """pass@k rate (equals pass@1 when attempts == 1)."""
        return self.passed / self.total if self.total else 0.0

    @property
    def pass_at_1_count(self) -> int:
        return sum(1 for r in self.results if r.pass_at_1)

    @property
    def pass_at_1_rate(self) -> float:
        return self.pass_at_1_count / self.total if self.total else 0.0

    @property
    def multi_attempt(self) -> bool:
        return any(r.attempts > 1 for r in self.results)

    @property
    def avg_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.total if self.total else 0.0

    @property
    def avg_tokens(self) -> float:
        return sum(r.tokens for r in self.results) / self.total if self.total else 0.0

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.total if self.total else 0.0

    @property
    def total_time(self) -> float:
        return sum(r.elapsed for r in self.results)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "success_rate": round(self.success_rate, 4),
                "pass_at_1": self.pass_at_1_count,
                "pass_at_1_rate": round(self.pass_at_1_rate, 4),
                "avg_steps": round(self.avg_steps, 2),
                "avg_tokens": round(self.avg_tokens, 1),
                "total_cost_usd": round(self.total_cost, 4),
                "avg_cost_usd": round(self.avg_cost, 4),
                "total_time": round(self.total_time, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def format_table(self) -> str:
        """Render a human-readable results table."""
        rows = [
            f"{'TASK':<24} {'RESULT':<8} {'AGENT':<10} {'STEPS':>5} {'TOKENS':>8} {'COST$':>7} {'TIME':>7}",
            "-" * 80,
        ]
        for r in self.results:
            if r.attempts > 1:
                verdict = f"{r.num_passed}/{r.attempts}"
            else:
                verdict = "PASS" if r.passed else "FAIL"
            rows.append(
                f"{r.task_id[:24]:<24} {verdict:<8} {r.agent_status[:10]:<10} "
                f"{r.steps:>5} {r.tokens:>8} {r.cost_usd:>7.3f} {r.elapsed:>6.1f}s"
            )
        rows.append("-" * 80)
        rows.append(
            f"Success rate: {self.passed}/{self.total} = {self.success_rate:.0%}   "
            f"avg_steps={self.avg_steps:.1f}  avg_tokens={self.avg_tokens:.0f}  "
            f"total_cost=${self.total_cost:.3f}  total_time={self.total_time:.1f}s"
        )
        if self.multi_attempt:
            rows.append(
                f"pass@1: {self.pass_at_1_count}/{self.total} = {self.pass_at_1_rate:.0%}   "
                f"pass@k: {self.passed}/{self.total} = {self.success_rate:.0%}"
            )
        return "\n".join(rows)


# Factory signature: given a TaskSpec + the pre-populated repo path, return an agent
# (Agent or LangGraphAgent) that can be called as run(task, log).
# repo_path is passed to allow building a RAG retriever per task if needed.
AgentFactory = Callable[[TaskSpec, str], object]


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------

class EvalHarness:
    """
    Evaluation executor.

    Usage:
        harness = EvalHarness(agent_factory=make_agent, results_dir="./eval_runs")
        report = harness.run_suite(default_suite())
        print(report.format_table())
        report.save_json("report.json")
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        results_dir: str | Path = "./eval_runs",
        keep_workdirs: bool = False,
        on_result: Callable[[EvalResult], None] | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Args:
            agent_factory: factory that takes a TaskSpec and returns a fresh agent
            results_dir:   root directory for temporary repos and event logs
            keep_workdirs: when True, retain each task's temporary repo (useful for debugging)
            on_result:     callback invoked after each task completes (for real-time progress printing)
            model_name:    model name used for blended $ cost estimation (optional)
        """
        self._factory = agent_factory
        self._results_dir = Path(results_dir).resolve()
        self._keep = keep_workdirs
        self._on_result = on_result
        self._model_name = model_name

    def run_suite(self, specs: Sequence[TaskSpec], attempts: int = 1) -> EvalReport:
        results = [self.run_task(spec, attempts=attempts) for spec in specs]
        return EvalReport(results=results)

    def run_task(self, spec: TaskSpec, attempts: int = 1) -> EvalResult:
        """
        Run one task `attempts` times (pass@k). Each attempt gets a fresh repo so
        attempts are independent. steps/tokens/elapsed/cost are summed across
        attempts; pass@1 is the first attempt, pass@k is any attempt.
        """
        from eval.pricing import estimate_cost

        attempts = max(1, attempts)
        run_root = self._results_dir / spec.id
        repo_path = run_root / "repo"
        log_dir = run_root / "logs"

        passes: list[bool] = []
        tot_steps = tot_tokens = 0
        tot_elapsed = 0.0
        # Representative metadata: the first PASSING attempt if any, else the first attempt.
        rep_status = "crashed"
        rep_detail = ""
        rep_error: str | None = None
        rep_locked = False   # True once we've recorded a passing attempt's metadata

        for attempt_i in range(attempts):
            self._prepare_repo(spec, repo_path)
            agent = self._factory(spec, str(repo_path))
            task = Task(description=spec.description, repo_path=str(repo_path),
                        max_steps=spec.max_steps)

            t0 = time.time()
            run_error: str | None = None
            run_result: RunResult | None = None
            # CWD is the task repo so file_write / pytest resolve relative paths correctly
            prev_cwd = os.getcwd()
            try:
                os.chdir(repo_path)
                with EventLog.create(task, log_dir=str(log_dir)) as log:
                    run_result = agent.run(task, log)
            except Exception as exc:
                run_error = f"agent crashed: {exc}"
                logger.exception("Agent crashed on task %s", spec.id)
            finally:
                os.chdir(prev_cwd)
            tot_elapsed += time.time() - t0

            # Independent verification (runs even if the agent crashed)
            try:
                a_passed, a_detail = spec.verify(str(repo_path))
            except Exception as exc:
                a_passed, a_detail = False, f"verifier error: {exc}"

            passes.append(a_passed)
            tot_steps += (run_result.steps_taken if run_result else 0)
            tot_tokens += (run_result.total_tokens if run_result else 0)
            a_status = run_result.status.value if run_result else "crashed"
            a_error = run_error or (run_result.error if run_result else None)

            # Representative metadata: take attempt 0 first, then upgrade to the
            # first passing attempt (and lock so later attempts don't overwrite).
            if (attempt_i == 0) or (a_passed and not rep_locked):
                rep_status, rep_detail, rep_error = a_status, a_detail, a_error
                if a_passed:
                    rep_locked = True

        result = EvalResult(
            task_id=spec.id,
            passed=any(passes),
            agent_status=rep_status,
            steps=tot_steps,
            tokens=tot_tokens,
            elapsed=tot_elapsed,
            detail=rep_detail,
            error=rep_error,
            pass_at_1=passes[0],
            attempts=attempts,
            num_passed=sum(passes),
            cost_usd=estimate_cost(self._model_name, tot_tokens),
        )

        if not self._keep:
            shutil.rmtree(repo_path, ignore_errors=True)
        if self._on_result:
            self._on_result(result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_repo(self, spec: TaskSpec, repo_path: Path) -> None:
        """Set up the initial task files and run git init."""
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        repo_path.mkdir(parents=True, exist_ok=True)

        if spec.setup_dir:
            src = Path(spec.setup_dir)
            if src.is_dir():
                shutil.copytree(src, repo_path, dirs_exist_ok=True)

        for rel, content in spec.setup_files.items():
            dest = repo_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        self._git_init(repo_path)

    @staticmethod
    def _git_init(repo_path: Path) -> None:
        """git init + initial commit so git diff and git tools are available. Failures are silent."""
        try:
            env = {"GIT_TERMINAL_PROMPT": "0"}
            for args in (
                ["git", "init", "-q"],
                ["git", "config", "user.email", "eval@forge.agent"],
                ["git", "config", "user.name", "Forge Eval"],
                ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "eval baseline"],
            ):
                subprocess.run(args, cwd=repo_path, capture_output=True,
                               timeout=20, env={**__import__("os").environ, **env})
        except Exception:
            pass
