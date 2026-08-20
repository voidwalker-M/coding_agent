"""
tests/test_memory_integration.py

End-to-end wiring of long-term memory through a real Agent run: the agent makes an
actual file edit (so the no-op-finish guard passes and git diff is non-empty), then
finishes; core.py should capture an episodic memory, and a later run should recall
it. Uses MockBackend + a real FileWriteTool in a temporary git repo (no API).
"""

import subprocess

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from context.memory import LongTermMemory
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.file_tool import FileWriteTool


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def _init_repo(tmp_path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b\n")   # buggy
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path


def _registry():
    return ToolRegistry().register(FileWriteTool())


def test_run_captures_episode_and_later_recalls(tmp_path):
    repo = _init_repo(tmp_path)
    mem_dir = tmp_path / "mem"
    ltm = LongTermMemory(str(mem_dir)).load()

    # Script: fix the bug with a real write, then finish.
    script = [
        Action(ActionType.TOOL_CALL, "fix the bug",
               tool_call=ToolCall("file_write", {"path": str(repo / "app.py"),
                                                 "content": "def add(a, b):\n    return a + b\n"})),
        Action(ActionType.FINISH, "", message="Fixed the add() operator bug in app.py"),
    ]
    backend = MockBackend(script)
    cfg = AgentConfig(max_steps=5, long_term_memory=ltm, capture_episodes=True)
    agent = Agent(backend, _registry(), cfg)

    task = Task(description="Fix the broken add() function in app.py", repo_path=str(repo), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)

    assert result.status.value == "success"
    # An episodic memory was captured.
    assert ltm.count == 1
    rec = ltm._records[0]
    assert rec.kind == "episodic"
    assert rec.outcome == "success"
    assert "app.py" in rec.text                       # the changed file was recorded

    # A fresh store reloaded from disk recalls it.
    reloaded = LongTermMemory(str(mem_dir)).load()
    block = reloaded.recall("how did we fix the add function", k=3)
    assert "app.py" in block


def test_give_up_with_memory_off_does_no_git_diff(tmp_path):
    """Regression: the give_up path used to compute `git diff` as an argument to
    _capture_episode, so it ran even with memory disabled. Memory off must be inert."""
    repo = _init_repo(tmp_path)
    agent = Agent(MockBackend([Action(ActionType.GIVE_UP, "nope", message="cannot")]),
                  _registry(), AgentConfig(max_steps=3))

    calls = []
    original = agent._get_git_diff
    agent._get_git_diff = lambda p: (calls.append(p), original(p))[1]

    task = Task(description="impossible", repo_path=str(repo), max_steps=3)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)
    assert result.status.value == "gave_up"
    assert calls == []                      # no git subprocess when memory is off


def test_give_up_with_memory_on_captures_failure_episode(tmp_path):
    repo = _init_repo(tmp_path)
    ltm = LongTermMemory(str(tmp_path / "mem")).load()
    agent = Agent(MockBackend([Action(ActionType.GIVE_UP, "nope", message="could not fix it")]),
                  _registry(), AgentConfig(max_steps=3, long_term_memory=ltm))
    task = Task(description="impossible task", repo_path=str(repo), max_steps=3)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)
    assert ltm.count == 1
    assert ltm._records[0].outcome == "failure"
    assert ltm._records[0].importance == pytest.approx(0.4)   # failures rank below wins


def test_caller_owned_short_term_is_not_wiped(tmp_path):
    """A chat session owns its ShortTermMemory across rounds — the agent must not
    clear it at the start of each run."""
    from context.memory import ShortTermMemory
    repo = _init_repo(tmp_path)
    stm = ShortTermMemory()
    stm.add_note("carried over from a previous round")

    agent = Agent(MockBackend([Action(ActionType.GIVE_UP, "x", message="stop")]),
                  _registry(), AgentConfig(max_steps=2, short_term_memory=stm))
    task = Task(description="anything", repo_path=str(repo), max_steps=2)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)
    assert "carried over from a previous round" in stm.render()


def test_injected_history_gets_evict_hook(tmp_path):
    """Chat mode builds the history itself; the agent must still attach the fold
    hook so trimmed turns reach working memory."""
    from context.history import ConversationHistory
    from context.memory import ShortTermMemory
    from llm.base import LLMMessage

    repo = _init_repo(tmp_path)
    stm = ShortTermMemory()
    hist = ConversationHistory(max_messages=3)
    hist.add(LLMMessage(role="user", content="the original task"))
    assert not hist.has_evict_callback()

    agent = Agent(MockBackend([Action(ActionType.GIVE_UP, "x", message="stop")]),
                  _registry(), AgentConfig(max_steps=2, short_term_memory=stm))
    agent._pending_history = hist
    task = Task(description="anything", repo_path=str(repo), max_steps=2)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        agent.run(task, log)

    assert hist.has_evict_callback()
    # Overflow the window; the evicted turn should land in working memory.
    for i in range(4):
        hist.add(LLMMessage(role="user", content=f"MARKER-{i} content"))
    assert "MARKER-0" in stm.render()


def test_memory_disabled_is_inert(tmp_path):
    """With memory off (default), no memory dir is created and behavior is unchanged."""
    repo = _init_repo(tmp_path)
    script = [
        Action(ActionType.TOOL_CALL, "fix",
               tool_call=ToolCall("file_write", {"path": str(repo / "app.py"),
                                                 "content": "def add(a, b):\n    return a + b\n"})),
        Action(ActionType.FINISH, "", message="fixed"),
    ]
    agent = Agent(MockBackend(script), _registry(), AgentConfig(max_steps=5))
    task = Task(description="fix add", repo_path=str(repo), max_steps=5)
    with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
        result = agent.run(task, log)
    assert result.status.value == "success"
    assert not (tmp_path / ".agent_memory").exists()
