"""
tests/test_orchestrator.py

Tests for the multi-agent orchestrator (#3): planner → coder → reviewer.
Driven by a single MockBackend whose script is consumed across the three roles
in order (roles run sequentially), so no API is used.
"""

import os

import pytest

from agent.core import AgentConfig
from agent.orchestrator import Orchestrator, OrchestratorAgent, READ_ONLY_TOOLS, REVIEWER_EXTRA_TOOLS
from agent.task import Action, ActionType, RunStatus, Task, ToolCall
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.file_tool import FileReadTool, FileWriteTool
from tools.test_tool import PytestTool


def _full_registry():
    return (ToolRegistry()
            .register(FileReadTool())
            .register(FileWriteTool())
            .register(PytestTool()))


def test_subset_restricts_tools():
    reg = _full_registry()
    ro = reg.subset(READ_ONLY_TOOLS)
    assert "file_read" in ro.tool_names
    assert "file_write" not in ro.tool_names      # editing tool excluded
    assert "pytest" not in ro.tool_names
    assert "test" not in ro.tool_names
    reviewer = reg.subset(READ_ONLY_TOOLS + REVIEWER_EXTRA_TOOLS)
    assert "test" in reviewer.tool_names
    assert "file_write" not in reviewer.tool_names


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    return repo


def test_orchestrator_approves_when_reviewer_says_approve(tmp_path):
    repo = _make_repo(tmp_path)
    fix = "def add(a, b):\n    return a + b\n"
    script = [
        # Planner (read-only) → finishes with a plan
        Action(ActionType.FINISH, "plan", message="1. Edit calc.py: return a + b"),
        # Coder → writes the fix, finishes
        Action(ActionType.TOOL_CALL, "fix",
               tool_call=ToolCall("file_write", {"path": "calc.py", "content": fix})),
        Action(ActionType.FINISH, "done", message="Changed add() to use +"),
        # Reviewer → approves
        Action(ActionType.FINISH, "looks good", message="Diff is correct, tests pass. APPROVE"),
    ]
    orch = Orchestrator(MockBackend(script), _full_registry(),
                        AgentConfig(max_steps=5), max_iterations=2)

    prev = os.getcwd()
    try:
        os.chdir(repo)
        result = orch.run(Task(description="fix add()", repo_path=str(repo), max_steps=5),
                          log_dir=str(tmp_path / "logs"))
    finally:
        os.chdir(prev)

    assert result.approved is True
    assert result.status == RunStatus.SUCCESS
    assert result.iterations == 1
    assert [r.role for r in result.roles] == ["planner", "coder", "reviewer"]
    assert "Edit calc.py" in result.plan


def test_orchestrator_loops_then_gives_up_when_never_approved(tmp_path):
    repo = _make_repo(tmp_path)
    # Reviewer always says REVISE → orchestrator loops to max_iterations then gives up.
    script = [
        Action(ActionType.FINISH, "plan", message="1. fix it"),               # planner
        Action(ActionType.FINISH, "attempt 1", message="tried"),              # coder it1
        Action(ActionType.FINISH, "review 1", message="not done. REVISE: fix X"),  # reviewer it1
        Action(ActionType.FINISH, "attempt 2", message="tried again"),        # coder it2
        Action(ActionType.FINISH, "review 2", message="still wrong. REVISE"),  # reviewer it2
    ]
    orch = Orchestrator(MockBackend(script), _full_registry(),
                        AgentConfig(max_steps=5), max_iterations=2)

    prev = os.getcwd()
    try:
        os.chdir(repo)
        result = orch.run(Task(description="fix add()", repo_path=str(repo), max_steps=5),
                          log_dir=str(tmp_path / "logs"))
    finally:
        os.chdir(prev)

    assert result.approved is False
    assert result.status == RunStatus.GAVE_UP
    assert result.iterations == 2
    # planner + 2 * (coder + reviewer) = 5 role runs
    assert len(result.roles) == 5


def test_on_role_start_callback_fires(tmp_path):
    repo = _make_repo(tmp_path)
    seen = []
    script = [
        Action(ActionType.FINISH, "plan", message="plan"),
        Action(ActionType.FINISH, "code", message="done"),
        Action(ActionType.FINISH, "review", message="APPROVE"),
    ]
    orch = Orchestrator(MockBackend(script), _full_registry(),
                        AgentConfig(max_steps=3), max_iterations=1,
                        on_role_start=seen.append)
    prev = os.getcwd()
    try:
        os.chdir(repo)
        orch.run(Task(description="x", repo_path=str(repo), max_steps=3),
                 log_dir=str(tmp_path / "logs"))
    finally:
        os.chdir(prev)
    assert seen == ["planner", "coder", "reviewer"]


def test_pair_topology_skips_planner(tmp_path):
    repo = _make_repo(tmp_path)
    script = [
        Action(ActionType.FINISH, "code", message="done"),
        Action(ActionType.FINISH, "review", message="APPROVE"),
    ]
    orch = Orchestrator(MockBackend(script), _full_registry(),
                        AgentConfig(max_steps=3), max_iterations=1,
                        topology="pair")
    prev = os.getcwd()
    try:
        os.chdir(repo)
        result = orch.run(Task(description="x", repo_path=str(repo), max_steps=3),
                          log_dir=str(tmp_path / "logs"))
    finally:
        os.chdir(prev)
    assert result.topology == "pair"
    assert result.approved is True
    assert [r.role for r in result.roles] == ["coder", "reviewer"]


def test_autonomous_topology_single_role(tmp_path):
    repo = _make_repo(tmp_path)
    script = [Action(ActionType.FINISH, "solo", message="finished")]
    orch = Orchestrator(MockBackend(script), _full_registry(),
                        AgentConfig(max_steps=3), topology="autonomous")
    prev = os.getcwd()
    try:
        os.chdir(repo)
        result = orch.run(Task(description="x", repo_path=str(repo), max_steps=3),
                          log_dir=str(tmp_path / "logs"))
    finally:
        os.chdir(prev)
    assert result.approved is True
    assert [r.role for r in result.roles] == ["autonomous"]
    assert orch.scratchpad.entries


def test_orchestrator_agent_matches_agent_run_signature(tmp_path):
    from agent.event_log import EventLog
    from agent.task import RunResult

    repo = _make_repo(tmp_path)
    script = [
        Action(ActionType.FINISH, "code", message="done"),
        Action(ActionType.FINISH, "review", message="APPROVE"),
    ]
    wrapper = OrchestratorAgent(
        Orchestrator(MockBackend(script), _full_registry(),
                     AgentConfig(max_steps=3), max_iterations=1, topology="pair"),
    )
    task = Task(description="x", repo_path=str(repo), max_steps=3)
    prev = os.getcwd()
    try:
        os.chdir(repo)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = wrapper.run(task, log)
    finally:
        os.chdir(prev)
    assert isinstance(result, RunResult)
    assert result.status == RunStatus.SUCCESS
