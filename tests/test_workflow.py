"""Closed-loop: ticket → run → verify → optional memory."""

import os

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from agent.workflow import ClosedLoop, Ticket
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.file_tool import FileWriteTool


def test_closed_loop_verifies_with_command(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.py").write_text("print('ok')\n")
    script = [
        Action(ActionType.TOOL_CALL, "write",
               tool_call=ToolCall("file_write", {
                   "path": "marker.txt", "content": "done\n",
               })),
        Action(ActionType.FINISH, "done", message="wrote marker"),
    ]
    registry = ToolRegistry().register(FileWriteTool())
    agent = Agent(MockBackend(script), registry, AgentConfig(max_steps=5))
    task = Task(description="write marker", repo_path=str(repo), max_steps=5)
    ticket = Ticket(id="t1", title="write marker", source="eval")
    loop = ClosedLoop(agent, verify_cmd="test -f marker.txt", remember_on_success=False)

    prev = os.getcwd()
    try:
        os.chdir(repo)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            outcome = loop.execute(ticket, task, log)
    finally:
        os.chdir(prev)

    assert outcome.run.is_success()
    assert outcome.verified is True
    assert outcome.is_success() is True


def test_closed_loop_fails_when_verifier_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script = [Action(ActionType.FINISH, "done", message="claimed success")]
    agent = Agent(MockBackend(script), ToolRegistry(), AgentConfig(max_steps=3))
    task = Task(description="x", repo_path=str(repo), max_steps=3)
    loop = ClosedLoop(agent, verify_cmd="false", remember_on_success=False)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        outcome = loop.execute(Ticket(id="t2", title="x"), task, log)
    assert outcome.run.is_success()
    assert outcome.verified is False
    assert outcome.is_success() is False
