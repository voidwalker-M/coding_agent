"""
tests/test_timing.py

Timing instrumentation: per-step durations in the event log, run totals on
RunResult, and aggregation through summarize_run and the eval harness.

Durations are asserted as ">= 0" or "roughly >= a known sleep" rather than exact
values — wall-clock timing is inherently noisy.
"""

import time

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog, summarize_run
from agent.task import Action, ActionType, EventType, Task, ToolCall
from eval.harness import EvalReport, EvalResult
from llm.base import LLMBackend, LLMResponse, MockBackend
from tools.base import BaseTool, NoopTool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SlowTool(BaseTool):
    """Tool that sleeps a known duration so tool_time is measurably non-zero."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay

    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "Sleeps."

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, params: dict) -> ToolResult:
        time.sleep(self._delay)
        return ToolResult(success=True, output="slept")


class SlowBackend(LLMBackend):
    """Backend that sleeps before returning each scripted action."""

    def __init__(self, script: list[Action], delay: float = 0.05) -> None:
        self._inner = MockBackend(script)
        self._delay = delay

    @property
    def model_name(self) -> str:
        return "slow-mock"

    def complete(self, messages, tools) -> LLMResponse:
        time.sleep(self._delay)
        return self._inner.complete(messages, tools)


def _run(backend, registry, tmp_path, max_steps: int = 5):
    task = Task(description="t", repo_path=str(tmp_path), max_steps=max_steps)
    agent = Agent(backend, registry, AgentConfig(max_steps=max_steps))
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)
        events = log.replay()
        stats = summarize_run(log)
    return result, events, stats


# ---------------------------------------------------------------------------
# EventLog duration payloads
# ---------------------------------------------------------------------------

def test_action_and_observation_events_carry_durations(tmp_path):
    script = [
        Action(ActionType.TOOL_CALL, "use it", ToolCall("noop", {})),
        Action(ActionType.FINISH, "done", message="Done"),
    ]
    registry = ToolRegistry().register(NoopTool())
    _, events, _ = _run(MockBackend(script), registry, tmp_path)

    actions = [e for e in events if e.event_type == EventType.ACTION]
    observations = [e for e in events if e.event_type == EventType.OBSERVATION]

    assert actions and all(e.payload["duration_ms"] >= 0 for e in actions)
    assert observations and all(e.payload["duration_ms"] >= 0 for e in observations)


def test_log_action_duration_defaults_to_none(tmp_path):
    """Callers that don't measure still work — the key is present but null."""
    task = Task(description="t", repo_path=str(tmp_path))
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        log.log_action(step=1, action=Action(ActionType.FINISH, "x", message="m"))
        events = log.replay()

    assert events[0].payload["duration_ms"] is None


def test_replay_tolerates_logs_without_duration(tmp_path):
    """Event logs written before instrumentation must still parse."""
    legacy = tmp_path / "old_20240101_000000.jsonl"
    legacy.write_text(
        '{"event_id":"e1","event_type":"observation","task_id":"old",'
        '"timestamp":"2024-01-01T00:00:00+00:00",'
        '"payload":{"step":1,"observation":{"status":"success","output":"o",'
        '"tool_name":"noop","tokens_used":0,"error":null}}}\n',
        encoding="utf-8",
    )
    with EventLog.open_existing(legacy) as log:
        stats = summarize_run(log)

    assert stats["observations_ok"] == 1
    assert stats["tool_time_total"] == 0.0
    assert stats["tool_time_by_name"] == {}


# ---------------------------------------------------------------------------
# RunResult totals
# ---------------------------------------------------------------------------

def test_run_result_reports_llm_and_tool_time(tmp_path):
    script = [
        Action(ActionType.TOOL_CALL, "use it", ToolCall("slow", {})),
        Action(ActionType.FINISH, "done", message="Done"),
    ]
    registry = ToolRegistry().register(SlowTool(delay=0.05))
    result, _, _ = _run(SlowBackend(script, delay=0.05), registry, tmp_path)

    assert result.llm_time >= 0.09    # two LLM calls at ~50ms each
    assert result.tool_time >= 0.04   # one tool call at ~50ms


def test_timing_recorded_when_run_hits_max_steps(tmp_path):
    """Totals must survive the non-FINISH exit paths too."""
    script = [Action(ActionType.TOOL_CALL, "loop", ToolCall("noop", {"i": i}))
              for i in range(3)]
    registry = ToolRegistry().register(NoopTool())
    result, _, _ = _run(MockBackend(script), registry, tmp_path, max_steps=3)

    assert result.status.value == "max_steps"
    assert result.llm_time > 0


def test_run_result_serializes_timing():
    from agent.task import RunResult, RunStatus

    d = RunResult(task_id="t", status=RunStatus.SUCCESS, summary="s",
                  steps_taken=1, llm_time=1.5, tool_time=0.5).to_dict()

    assert d["llm_time"] == 1.5
    assert d["tool_time"] == 0.5


# ---------------------------------------------------------------------------
# summarize_run aggregation
# ---------------------------------------------------------------------------

def test_summarize_run_breaks_time_down_by_tool(tmp_path):
    script = [
        Action(ActionType.TOOL_CALL, "slow", ToolCall("slow", {})),
        Action(ActionType.TOOL_CALL, "fast", ToolCall("noop", {})),
        Action(ActionType.FINISH, "done", message="Done"),
    ]
    registry = ToolRegistry().register(SlowTool(delay=0.05)).register(NoopTool())
    _, _, stats = _run(MockBackend(script), registry, tmp_path)

    assert set(stats["tool_time_by_name"]) == {"slow", "noop"}
    assert stats["tool_time_by_name"]["slow"] > stats["tool_time_by_name"]["noop"]
    assert stats["tool_time_total"] > 0
    assert stats["llm_time_total"] >= 0


# ---------------------------------------------------------------------------
# Eval harness plumbing
# ---------------------------------------------------------------------------

def test_eval_result_timing_defaults_keep_positional_construction_valid():
    r = EvalResult("t1", True, "success", 3, 500, 1.2, "ok")
    assert r.llm_time == 0.0 and r.tool_time == 0.0


def test_eval_report_aggregates_and_serializes_timing():
    results = [
        EvalResult("a", True, "success", 1, 10, 2.0, "", llm_time=1.0, tool_time=0.5),
        EvalResult("b", False, "gave_up", 1, 10, 3.0, "", llm_time=2.0, tool_time=0.25),
    ]
    report = EvalReport(results=results)

    assert report.total_llm_time == 3.0
    assert report.total_tool_time == 0.75

    summary = report.to_dict()["summary"]
    assert summary["total_llm_time"] == 3.0
    assert summary["total_tool_time"] == 0.75
    assert report.to_dict()["results"][0]["llm_time"] == 1.0
    assert "llm=3.0s" in report.format_table()
