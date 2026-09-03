"""
agent/workflow.py

Thin business closed loop: ticket → agent run → verify → persist a lesson.

GitHub Issue → PR (`entry/github_issue.py`) is one instance of this loop.
This module is the reusable core so local tickets, eval tasks, and the GitHub
path share the same "did the change actually work, and what should we remember?"
step instead of stopping at the model's self-reported FINISH.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from agent.event_log import EventLog
from agent.task import RunResult, Task


@dataclass
class Ticket:
    """Work item that kicks off a closed-loop run."""

    id: str
    title: str
    body: str = ""
    source: str = "local"  # local | github | eval

    def as_task_description(self) -> str:
        body = self.body.strip()
        if body:
            return f"{self.title}\n\n{body}"
        return self.title


@dataclass
class ClosedLoopResult:
    ticket: Ticket
    run: RunResult
    verified: bool
    verify_detail: str
    memory_id: str | None = None

    def is_success(self) -> bool:
        return self.run.is_success() and self.verified


class ClosedLoop:
    """
    Run an agent on a ticket, optionally re-run a verifier, and write a
    long-term memory on success so the next ticket can reuse the lesson.
    """

    def __init__(
        self,
        agent: Any,
        *,
        memory: Any = None,
        verify_cmd: str | None = None,
        remember_on_success: bool = True,
        verify_timeout: int = 60,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._verify_cmd = verify_cmd
        self._remember_on_success = remember_on_success
        self._verify_timeout = verify_timeout

    def execute(self, ticket: Ticket, task: Task, log: EventLog) -> ClosedLoopResult:
        run = self._agent.run(task, log)
        verified, detail = self._verify(task.repo_path)
        # If no verifier was configured, the agent's own status is the gate.
        if self._verify_cmd is None:
            verified = run.is_success()
            detail = "no independent verifier; used agent status"

        memory_id = None
        if (
            self._remember_on_success
            and self._memory is not None
            and run.is_success()
            and verified
        ):
            memory_id = self._remember(ticket, run, detail)
        return ClosedLoopResult(
            ticket=ticket,
            run=run,
            verified=verified,
            verify_detail=detail,
            memory_id=memory_id,
        )

    def _verify(self, repo_path: str) -> tuple[bool, str]:
        if not self._verify_cmd:
            return True, "skipped"
        try:
            proc = subprocess.run(
                self._verify_cmd,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self._verify_timeout,
            )
        except Exception as exc:
            return False, f"verifier error: {exc}"
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if len(output) > 2000:
            output = output[-2000:]
        ok = proc.returncode == 0
        return ok, output or ("PASS" if ok else f"exit {proc.returncode}")

    def _remember(self, ticket: Ticket, run: RunResult, detail: str) -> str | None:
        text = (
            f"Closed-loop ticket {ticket.id} ({ticket.source}): {ticket.title}. "
            f"Outcome: {run.status.value}. Summary: {run.summary or '(none)'}"
        )
        try:
            rec = self._memory.remember(
                text,
                kind="feedback",
                description=f"ticket {ticket.id} resolved",
                tags=("closed-loop", ticket.source),
                source=ticket.source,
                outcome="success",
                importance=0.6,
                scope="global",
                visibility="public",
            )
            return getattr(rec, "id", None) or getattr(rec, "name", None) or "ok"
        except Exception:
            return None
