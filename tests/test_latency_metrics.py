"""Tests for TTFT / E2E latency metrics in EventLog."""

from __future__ import annotations

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog, summarize_run
from agent.task import Action, ActionType, Task
from llm.base import MockBackend
from tools.base import NoopTool, ToolRegistry


def _streaming_backend(script):
    backend = MockBackend(script)

    def fake_stream(messages, tools, on_text=None, on_thought=None):
        if on_thought:
            on_thought("think")
        if on_text:
            on_text("answer")
        return backend.complete(messages, tools)

    backend.stream = fake_stream  # type: ignore[method-assign]
    return backend


def test_ttft_logged_in_event_log(tmp_path):
    script = [Action(ActionType.FINISH, "done", message="ok")]
    backend = _streaming_backend(script)
    registry = ToolRegistry().register(NoopTool("noop"))
    agent = Agent(
        backend, registry,
        AgentConfig(stream=True, max_steps=5, thought_callback=lambda _: None),
    )
    task = Task(description="latency test", repo_path=str(tmp_path))
    log = EventLog.create(task, log_dir=str(tmp_path))
    agent.run(task, log)
    log.close()

    stats = summarize_run(log)
    assert stats["actions"] == 1
    assert stats.get("avg_ttft_ms") is not None
    assert stats["avg_ttft_ms"] >= 0
    events = log.replay()
    action = next(e for e in events if e.event_type.value == "action")
    assert "ttft_ms" in action.payload
    assert "e2e_ms" in action.payload
