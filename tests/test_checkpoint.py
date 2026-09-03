"""Tests for checkpoint save/load and agent resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    RunCheckpoint,
    short_term_from_dict,
    short_term_to_dict,
)
from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from context.memory import ShortTermMemory
from llm.base import MockBackend
from tools.base import NoopTool, ToolRegistry


def _registry():
    return ToolRegistry().register(NoopTool("noop"))


def test_checkpoint_roundtrip(tmp_path):
    cp = RunCheckpoint.from_task(
        Task(description="fix bug", repo_path="/repo"),
        step=3,
        total_tokens=100,
        llm_time_total=1.5,
        tool_time_total=0.2,
        steps_without_edit=1,
        noop_finish_rejections=0,
        history=[{"role": "user", "content": "go"}],
        short_term={"notes": ["n1"], "files": ["a.py"], "summary": "s"},
        log_path="/logs/x.jsonl",
    )
    path = save_checkpoint(cp, tmp_path)
    loaded = load_checkpoint(path)
    assert loaded.step == 3
    assert loaded.total_tokens == 100
    assert loaded.history[0]["content"] == "go"


def test_short_term_serialization():
    stm = ShortTermMemory()
    stm.add_note("remember this")
    stm.note_file("foo.py")
    stm.append_query("what broke")
    data = short_term_to_dict(stm)
    restored = short_term_from_dict(data)
    assert "remember this" in restored._notes
    assert restored._files == ["foo.py"]
    assert restored.queries[0].user_text == "what broke"


def test_agent_resume_continues_from_checkpoint(tmp_path):
    script = [
        Action(ActionType.TOOL_CALL, "t1", ToolCall("noop", {})),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    backend = MockBackend(script)
    agent = Agent(backend, _registry(), AgentConfig(max_steps=10, stream=False))
    task = Task(description="resume me", repo_path=str(tmp_path))
    log1 = EventLog.create(task, log_dir=str(tmp_path / "logs"))
    ckpt_dir = tmp_path / "ckpt"

    agent.run(task, log1, checkpoint_dir=str(ckpt_dir))
    log1.close()

    ckpt_path = tmp_path / "ckpt" / f"{task.task_id}.checkpoint.json"
    assert ckpt_path.exists()
    cp = load_checkpoint(ckpt_path)
    assert cp.step >= 1

    backend.reset()
    log2 = EventLog.open_existing(log1.path)
    result = agent.run(task, log2, checkpoint_dir=str(ckpt_dir), resume_from=str(ckpt_path))
    log2.close()
    assert result.is_success()
    assert result.resumed_from_step == cp.step
