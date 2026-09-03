"""Keep repo-map / RAG / symbols aligned with disk after file edits."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from context.memory import ShortTermMemory
from context.workspace_sync import (
    ANY_PATH,
    EDIT_TOOLS,
    note_for_prompt,
    path_from_edit,
    record_edit,
)
from llm.base import MockBackend
from tools.base import FailingTool, NoopTool, ToolRegistry
from tools.file_tool import FileWriteTool


class DummyRetriever:
    def __init__(self) -> None:
        self.builds = 0
        self.chunk_count = 0

    def build(self) -> None:
        self.builds += 1
        self.chunk_count = 1

    def retrieve(self, query: str, k: int = 5) -> str:
        return f"rag-{self.builds}:{query[:20]}"


def test_path_from_edit_and_record_edit():
    assert path_from_edit("shell", {"cmd": "ls"}) is None
    assert path_from_edit("file_write", {"path": "a.py"}) == "a.py"
    assert path_from_edit("file_edit", {"file": "b.py"}) == "b.py"
    assert path_from_edit("undo", {}) == ANY_PATH
    assert "undo" in EDIT_TOOLS

    dirty: set[str] = set()
    record_edit(dirty, "file_write", {"path": "a.py"}, succeeded=True)
    record_edit(dirty, "file_write", {"path": "b.py"}, succeeded=False)
    record_edit(dirty, "shell", {"cmd": "ls"}, succeeded=True)
    record_edit(dirty, "undo", {}, succeeded=True)
    assert dirty == {"a.py", ANY_PATH}

    note = note_for_prompt({"src/app.py"})
    assert "src/app.py" in note
    assert "stale" in note.lower()


def test_repo_map_rebuilds_after_successful_write(tmp_path):
    (tmp_path / "mod.py").write_text("def foo():\n    return 1\n")
    task = Task(
        task_id="ws-write",
        description="edit then finish",
        repo_path=str(tmp_path),
        max_steps=5,
    )
    registry = ToolRegistry().register(FileWriteTool()).register(NoopTool("shell"))
    script = [
        Action(ActionType.TOOL_CALL, "write", ToolCall("file_write", {
            "path": str(tmp_path / "mod.py"),
            "content": "def foo():\n    return 2\n",
        })),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    builds = 0
    original = __import__("context.repo_map", fromlist=["RepoMap"]).RepoMap.build

    def counting_build(self, budget=8000, query=None):
        nonlocal builds
        builds += 1
        return original(self, budget, query)

    with patch("context.repo_map.RepoMap.build", counting_build):
        agent = Agent(MockBackend(script), registry)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            agent.run(task, log)

    assert builds == 2


def test_failed_write_does_not_rebuild_repo_map(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    task = Task(
        task_id="ws-fail",
        description="failed write",
        repo_path=str(tmp_path),
        max_steps=5,
    )
    registry = ToolRegistry().register(FailingTool("file_write")).register(NoopTool("shell"))
    script = [
        Action(ActionType.TOOL_CALL, "write", ToolCall("file_write", {
            "path": str(tmp_path / "mod.py"),
            "content": "nope",
        })),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    builds = 0
    original = __import__("context.repo_map", fromlist=["RepoMap"]).RepoMap.build

    def counting_build(self, budget=8000, query=None):
        nonlocal builds
        builds += 1
        return original(self, budget, query)

    with patch("context.repo_map.RepoMap.build", counting_build):
        agent = Agent(MockBackend(script), registry)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            agent.run(task, log)

    assert builds == 1


def test_rag_rebuilds_after_write(tmp_path):
    (tmp_path / "mod.py").write_text("def foo(): pass\n")
    retriever = DummyRetriever()
    stm = ShortTermMemory()
    task = Task(
        task_id="ws-rag",
        description="edit with rag",
        repo_path=str(tmp_path),
        max_steps=5,
    )
    registry = ToolRegistry().register(FileWriteTool())
    script = [
        Action(ActionType.TOOL_CALL, "write", ToolCall("file_write", {
            "path": str(tmp_path / "mod.py"),
            "content": "def foo():\n    return 1\n",
        })),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    agent = Agent(
        MockBackend(script),
        registry,
        AgentConfig(retriever=retriever, short_term_memory=stm),
    )
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)

    assert retriever.builds == 2
    assert any("mod.py" in n for n in stm._notes)


def test_prompt_tells_model_to_reread_after_edits():
    from agent.prompt import build_system_prompt

    prompt = build_system_prompt(".", [])
    assert "already read" in prompt
    assert "file_write" in prompt
    assert "stale" in prompt.lower()


def test_langgraph_rebuilds_repo_map_after_write(tmp_path):
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")

    from agent.langgraph_loop import LangGraphAgent
    from tools.file_tool import FileWriteTool

    (tmp_path / "mod.py").write_text("def foo():\n    return 1\n")
    task = Task(
        task_id="ws-lg",
        description="edit then finish",
        repo_path=str(tmp_path),
        max_steps=5,
    )
    registry = ToolRegistry().register(FileWriteTool()).register(NoopTool("shell"))
    script = [
        Action(ActionType.TOOL_CALL, "write", ToolCall("file_write", {
            "path": str(tmp_path / "mod.py"),
            "content": "def foo():\n    return 2\n",
        })),
        Action(ActionType.FINISH, "done", message="ok"),
    ]
    builds = 0
    original = __import__("context.repo_map", fromlist=["RepoMap"]).RepoMap.build

    def counting_build(self, budget=8000, query=None):
        nonlocal builds
        builds += 1
        return original(self, budget, query)

    with patch("context.repo_map.RepoMap.build", counting_build):
        agent = LangGraphAgent(MockBackend(script), registry)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            agent.run(task, log)

    assert builds == 2
