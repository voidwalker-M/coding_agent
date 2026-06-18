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
    """Evaluation result for a single task."""
    task_id: str
    passed: bool                # verifier judgment (objective ground truth)
    agent_status: str           # agent's self-reported final status
    steps: int
    tokens: int
    elapsed: float
    detail: str                 # verifier explanation
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "agent_status": self.agent_status,
            "steps": self.steps,
            "tokens": self.tokens,
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
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.total if self.total else 0.0

    @property
    def avg_tokens(self) -> float:
        return sum(r.tokens for r in self.results) / self.total if self.total else 0.0

    @property
    def total_time(self) -> float:
        return sum(r.elapsed for r in self.results)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "success_rate": round(self.success_rate, 4),
                "avg_steps": round(self.avg_steps, 2),
                "avg_tokens": round(self.avg_tokens, 1),
                "total_time": round(self.total_time, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def format_table(self) -> str:
        """Render a human-readable results table."""
        rows = [
            f"{'TASK':<24} {'RESULT':<8} {'AGENT':<10} {'STEPS':>5} {'TOKENS':>8} {'TIME':>7}",
            "-" * 66,
        ]
        for r in self.results:
            verdict = "PASS" if r.passed else "FAIL"
            rows.append(
                f"{r.task_id[:24]:<24} {verdict:<8} {r.agent_status[:10]:<10} "
                f"{r.steps:>5} {r.tokens:>8} {r.elapsed:>6.1f}s"
            )
        rows.append("-" * 66)
        rows.append(
            f"Success rate: {self.passed}/{self.total} = {self.success_rate:.0%}   "
            f"avg_steps={self.avg_steps:.1f}  avg_tokens={self.avg_tokens:.0f}  "
            f"total_time={self.total_time:.1f}s"
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
    ) -> None:
        """
        Args:
            agent_factory: factory that takes a TaskSpec and returns a fresh agent
            results_dir:   root directory for temporary repos and event logs
            keep_workdirs: when True, retain each task's temporary repo (useful for debugging)
            on_result:     callback invoked after each task completes (for real-time progress printing)
        """
        self._factory = agent_factory
        self._results_dir = Path(results_dir).resolve()
        self._keep = keep_workdirs
        self._on_result = on_result

    def run_suite(self, specs: Sequence[TaskSpec]) -> EvalReport:
        results = [self.run_task(spec) for spec in specs]
        return EvalReport(results=results)

    def run_task(self, spec: TaskSpec) -> EvalResult:
        run_root = self._results_dir / spec.id
        repo_path = run_root / "repo"
        log_dir = run_root / "logs"
        self._prepare_repo(spec, repo_path)

        agent = self._factory(spec, str(repo_path))
        task = Task(description=spec.description, repo_path=str(repo_path),
                    max_steps=spec.max_steps)

        t0 = time.time()
        run_error: str | None = None
        run_result: RunResult | None = None
        # CWD is set to the task repo so that file_write / pytest resolve relative paths correctly
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
        elapsed = time.time() - t0

        # Independent verification (runs even if the agent crashed)
        try:
            passed, detail = spec.verify(str(repo_path))
        except Exception as exc:
            passed, detail = False, f"verifier error: {exc}"

        result = EvalResult(
            task_id=spec.id,
            passed=passed,
            agent_status=(run_result.status.value if run_result else "crashed"),
            steps=(run_result.steps_taken if run_result else 0),
            tokens=(run_result.total_tokens if run_result else 0),
            elapsed=elapsed,
            detail=detail,
            error=run_error or (run_result.error if run_result else None),
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
