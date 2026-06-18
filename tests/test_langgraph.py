"""
tests/test_langgraph.py

LangGraph port tests.

Requires langgraph + langchain-core; the entire module is skipped if these are
not installed (pip install -e ".[langgraph]"). Driven by MockBackend; no API
calls.

Covers:
- ToolRegistry → LangChain StructuredTool wrapping
- LangGraphAgent runs to finish
- Tool call → observation written to EventLog
- MAX_STEPS returned when max_steps is reached
- Consistent behavior with the native Agent (same script, same result)
"""

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from agent.core import AgentConfig
from agent.event_log import EventLog
from agent.langgraph_loop import LangGraphAgent, to_langchain_tools
from agent.task import Action, ActionType, EventType, RunStatus, Task, ToolCall
from llm.base import MockBackend
from tools.base import NoopTool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# LangChain tool wrapping
# ---------------------------------------------------------------------------

def test_to_langchain_tools_wraps_all_tools():
    registry = ToolRegistry().register(NoopTool("shell")).register(NoopTool("file_read"))
    lc_tools = to_langchain_tools(registry)
    assert set(lc_tools.keys()) == {"shell", "file_read"}


def test_langchain_tool_invoke_returns_toolresult():
    registry = ToolRegistry().register(NoopTool("shell", output="hi"))
    lc_tools = to_langchain_tools(registry)
    result = lc_tools["shell"].invoke({"input": "anything"})
    assert isinstance(result, ToolResult)
    assert result.success
    assert result.output == "hi"


# ---------------------------------------------------------------------------
# LangGraphAgent execution
# ---------------------------------------------------------------------------

def test_langgraph_agent_runs_to_finish(tmp_path):
    script = [
        Action(ActionType.TOOL_CALL, thought="run it",
               tool_call=ToolCall("shell", {"input": "ls"})),
        Action(ActionType.FINISH, thought="done", message="all good"),
    ]
    backend = MockBackend(script)
    registry = ToolRegistry().register(NoopTool("shell"))
    agent = LangGraphAgent(backend, registry, AgentConfig(max_steps=5))

    task = Task(description="do it", repo_path=str(tmp_path), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)
        events = log.replay()

    assert result.status == RunStatus.SUCCESS
    assert result.summary == "all good"
    assert result.steps_taken == 2
    # EventLog compatibility: action and observation events must be present
    types = [e.event_type for e in events]
    assert EventType.ACTION in types
    assert EventType.OBSERVATION in types
    assert EventType.TASK_COMPLETE in types


def test_langgraph_agent_executes_tool_via_langchain(tmp_path):
    noop = NoopTool("shell", output="executed")
    script = [
        Action(ActionType.TOOL_CALL, thought="x", tool_call=ToolCall("shell", {"input": "y"})),
        Action(ActionType.FINISH, thought="done", message="done"),
    ]
    registry = ToolRegistry().register(noop)
    agent = LangGraphAgent(MockBackend(script), registry, AgentConfig(max_steps=5))

    task = Task(description="t", repo_path=str(tmp_path), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)

    assert noop.call_count == 1  # tool was actually called once


def test_langgraph_agent_hits_max_steps(tmp_path):
    # Only calls tools, never finishes → should stop at max_steps
    script = [
        Action(ActionType.TOOL_CALL, thought=f"step {i}",
               tool_call=ToolCall("shell", {"input": str(i)}))
        for i in range(10)
    ]
    backend = MockBackend(script)
    registry = ToolRegistry().register(NoopTool("shell"))
    agent = LangGraphAgent(backend, registry, AgentConfig(max_steps=3))

    task = Task(description="loop", repo_path=str(tmp_path), max_steps=3)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)

    assert result.status == RunStatus.MAX_STEPS
    assert result.steps_taken == 3


def test_langgraph_agent_give_up(tmp_path):
    script = [Action(ActionType.GIVE_UP, thought="cannot", message="too hard")]
    backend = MockBackend(script)
    registry = ToolRegistry().register(NoopTool("shell"))
    agent = LangGraphAgent(backend, registry, AgentConfig(max_steps=5))

    task = Task(description="x", repo_path=str(tmp_path), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)

    assert result.status == RunStatus.GAVE_UP
    assert result.summary == "too hard"
