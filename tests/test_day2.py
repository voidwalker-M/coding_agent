"""
tests/test_day2.py

Covers all key paths in agent/core.py, driven entirely by MockBackend without
consuming a real API.

Test scenarios:
- Normal completion (FINISH)
- Agent gives up voluntarily (GIVE_UP)
- Step limit reached (MAX_STEPS)
- Infinite-loop detection (LOOP_DETECTED)
- Reflection triggered: test failure
- Reflection triggered: N consecutive steps without an edit
- LLM call exception
- Unknown tool
- ToolRegistry basic functionality
"""

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, RunStatus, Task, ToolCall
from llm.base import MockBackend
from tools.base import FailingTool, NoopTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task(tmp_path) -> Task:
    return Task(
        task_id="test0001",
        description="Fix the failing test",
        repo_path=str(tmp_path),
        max_steps=10,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(NoopTool("shell", output="all tests passed"))
    r.register(NoopTool("file_read", output="def foo(): pass"))
    r.register(NoopTool("file_write", output="file written"))
    return r


@pytest.fixture
def log(tmp_path, task) -> EventLog:
    return EventLog.create(task, log_dir=str(tmp_path / "logs"))


def make_agent(backend, registry=None, config=None) -> Agent:
    if registry is None:
        registry = ToolRegistry()
        registry.register(NoopTool())
    return Agent(backend, registry, config)


def make_tool_call_action(tool="shell", params=None, thought="Let me run the tests.") -> Action:
    return Action(
        action_type=ActionType.TOOL_CALL,
        thought=thought,
        tool_call=ToolCall(name=tool, params=params or {}),
    )


def make_finish_action(message="All done.") -> Action:
    return Action(
        action_type=ActionType.FINISH,
        thought="Task is complete.",
        message=message,
    )


def make_give_up_action(message="Cannot solve.") -> Action:
    return Action(
        action_type=ActionType.GIVE_UP,
        thought="I'm stuck.",
        message=message,
    )


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_execute(self):
        tool = NoopTool("mytool", output="hello")
        registry = ToolRegistry()
        registry.register(tool)

        result = registry.execute_tool("mytool", {})
        assert result.success
        assert result.output == "hello"

    def test_duplicate_register_raises(self):
        registry = ToolRegistry()
        registry.register(NoopTool("mytool"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(NoopTool("mytool"))

    def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        result = registry.execute_tool("nonexistent", {})
        assert not result.success
        assert "nonexistent" in result.error

    def test_get_schemas_returns_all(self):
        registry = ToolRegistry()
        registry.register(NoopTool("a"))
        registry.register(NoopTool("b"))
        schemas = registry.get_schemas()
        names = [s.name for s in schemas]
        assert "a" in names
        assert "b" in names

    def test_chain_register(self):
        registry = (
            ToolRegistry()
            .register(NoopTool("x"))
            .register(NoopTool("y"))
        )
        assert len(registry) == 2

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(NoopTool("z"))
        assert "z" in registry
        assert "w" not in registry

    def test_tool_exception_returns_error(self):
        """When a tool raises internally, registry wraps it as an error result without propagating."""
        class BrokenTool(NoopTool):
            def execute(self, params):
                raise RuntimeError("disk full")

        registry = ToolRegistry()
        registry.register(BrokenTool("broken"))
        result = registry.execute_tool("broken", {})
        assert not result.success
        assert "disk full" in result.error


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------

class TestMockBackend:
    def test_returns_scripted_actions_in_order(self):
        actions = [make_tool_call_action(), make_finish_action()]
        backend = MockBackend(actions)

        r1 = backend.complete([], [])
        r2 = backend.complete([], [])
        assert r1.action.action_type == ActionType.TOOL_CALL
        assert r2.action.action_type == ActionType.FINISH

    def test_script_exhausted_returns_give_up(self):
        backend = MockBackend([make_finish_action()])
        backend.complete([], [])   # script exhausted
        r = backend.complete([], [])
        assert r.action.action_type == ActionType.GIVE_UP

    def test_tracks_call_count(self):
        backend = MockBackend([make_finish_action(), make_finish_action()])
        backend.complete([], [])
        backend.complete([], [])
        assert backend.call_count == 2

    def test_reset(self):
        backend = MockBackend([make_finish_action()])
        backend.complete([], [])
        backend.reset()
        r = backend.complete([], [])
        assert r.action.action_type == ActionType.FINISH


# ---------------------------------------------------------------------------
# Agent.run — normal completion
# ---------------------------------------------------------------------------

class TestAgentFinish:
    def test_finish_on_first_action(self, task, log, registry):
        backend = MockBackend([make_finish_action("Fixed it!")])
        agent = Agent(backend, registry)

        result = agent.run(task, log)

        assert result.status == RunStatus.SUCCESS
        assert result.summary == "Fixed it!"
        assert result.steps_taken == 1
        assert result.total_tokens > 0

    def test_finish_after_tool_calls(self, task, log, registry):
        script = [
            make_tool_call_action("shell"),
            make_tool_call_action("file_read"),
            make_finish_action("Done after exploration."),
        ]
        backend = MockBackend(script)
        agent = Agent(backend, registry)

        result = agent.run(task, log)

        assert result.status == RunStatus.SUCCESS
        assert result.steps_taken == 3

    def test_event_log_records_correct_sequence(self, task, log, registry):
        from agent.task import EventType

        script = [make_tool_call_action("shell"), make_finish_action()]
        backend = MockBackend(script)
        agent = Agent(backend, registry)
        agent.run(task, log)

        events = log.replay()
        types = [e.event_type for e in events]

        assert types[0] == EventType.TASK_START
        assert EventType.ACTION in types
        assert EventType.OBSERVATION in types
        assert types[-1] == EventType.TASK_COMPLETE


# ---------------------------------------------------------------------------
# Agent.run — voluntary give-up
# ---------------------------------------------------------------------------

class TestAgentGiveUp:
    def test_give_up_returns_gave_up_status(self, task, log, registry):
        backend = MockBackend([make_give_up_action("Too complex.")])
        agent = Agent(backend, registry)

        result = agent.run(task, log)

        assert result.status == RunStatus.GAVE_UP
        assert result.summary == "Too complex."

    def test_give_up_event_logged(self, task, log, registry):
        from agent.task import EventType

        backend = MockBackend([make_give_up_action()])
        agent = Agent(backend, registry)
        agent.run(task, log)

        events = log.replay()
        assert events[-1].event_type == EventType.TASK_FAILED


# ---------------------------------------------------------------------------
# Agent.run — step limit exceeded
# ---------------------------------------------------------------------------

class TestAgentMaxSteps:
    def test_max_steps_returns_correct_status(self, tmp_path):
        task = Task(
            task_id="maxtest",
            description="run forever",
            repo_path=str(tmp_path),
            max_steps=3,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
        # Different cmd each step to avoid triggering loop detection; max_steps=3 hits first
        script = [
            make_tool_call_action("shell", {"cmd": f"echo {i}"})
            for i in range(10)
        ]
        backend = MockBackend(script)
        registry = ToolRegistry()
        registry.register(NoopTool("shell"))
        config = AgentConfig(loop_detection_window=10)
        agent = Agent(backend, registry, config)

        result = agent.run(task, log)

        assert result.status == RunStatus.MAX_STEPS
        assert result.steps_taken == 3
        log.close()

    def test_max_steps_event_logged(self, tmp_path):
        from agent.task import EventType

        task = Task(
            task_id="maxtest2",
            description="run forever",
            repo_path=str(tmp_path),
            max_steps=2,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
        backend = MockBackend([make_tool_call_action()] * 10)
        registry = ToolRegistry()
        registry.register(NoopTool())
        agent = Agent(backend, registry)
        agent.run(task, log)

        events = log.replay()
        assert events[-1].event_type == EventType.TASK_FAILED
        assert "max_steps" in events[-1].payload["reason"]
        log.close()


# ---------------------------------------------------------------------------
# Agent.run — infinite loop detection
# ---------------------------------------------------------------------------

class TestLoopDetection:
    def test_loop_detected_gives_up(self, tmp_path):
        task = Task(
            task_id="looptest",
            description="infinite loop",
            repo_path=str(tmp_path),
            max_steps=20,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))
        # Same action repeated N times
        repeated = make_tool_call_action("shell", {"cmd": "echo hi"})
        backend = MockBackend([repeated] * 20)
        registry = ToolRegistry()
        registry.register(NoopTool("shell"))
        config = AgentConfig(loop_detection_window=3)
        agent = Agent(backend, registry, config)

        result = agent.run(task, log)

        assert result.status == RunStatus.GAVE_UP
        assert "Loop detected" in result.summary
        log.close()

    def test_different_actions_not_detected_as_loop(self, task, log, registry):
        """Alternating different actions should not trigger loop detection."""
        script = [
            make_tool_call_action("shell", {"cmd": "pytest"}),
            make_tool_call_action("file_read", {"path": "foo.py"}),
            make_tool_call_action("shell", {"cmd": "pytest"}),
            make_finish_action(),
        ]
        backend = MockBackend(script)
        config = AgentConfig(loop_detection_window=3)
        agent = Agent(backend, registry, config)

        result = agent.run(task, log)
        assert result.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# Agent.run — Reflection: test failure
# ---------------------------------------------------------------------------

class TestReflectionTestFailed:
    def test_reflection_triggered_on_test_failure(self, tmp_path):
        from agent.task import EventType

        task = Task(
            task_id="refltest",
            description="fix tests",
            repo_path=str(tmp_path),
            max_steps=10,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))

        # FailingTool simulates a pytest failure
        registry = ToolRegistry()
        registry.register(FailingTool("test", "AssertionError: 1 != 2"))
        registry.register(NoopTool("file_write"))

        script = [
            make_tool_call_action("test"),    # triggers reflection
            make_tool_call_action("file_write"),
            make_finish_action(),
        ]
        backend = MockBackend(script)
        config = AgentConfig(test_tool_names=("test",))
        agent = Agent(backend, registry, config)

        agent.run(task, log)

        events = log.replay()
        reflection_events = [e for e in events if e.event_type == EventType.REFLECTION]
        assert len(reflection_events) >= 1
        assert reflection_events[0].payload["reason"] == "test_failed"
        log.close()

    def test_reflection_injected_into_history(self, tmp_path):
        """After a reflection, the next LLM call's messages should contain the reflection prompt."""
        task = Task(
            task_id="reflhist",
            description="fix tests",
            repo_path=str(tmp_path),
            max_steps=10,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))

        registry = ToolRegistry()
        registry.register(FailingTool("test"))
        registry.register(NoopTool("file_write"))

        script = [
            make_tool_call_action("test"),
            make_finish_action(),
        ]
        backend = MockBackend(script)
        config = AgentConfig(test_tool_names=("test",))
        agent = Agent(backend, registry, config)
        agent.run(task, log)

        # The second LLM call's messages should contain the word REFLECTION
        assert backend.call_count >= 2
        second_call_messages = backend.received_messages[1]
        contents = " ".join(m.content for m in second_call_messages)
        assert "REFLECTION" in contents
        log.close()


# ---------------------------------------------------------------------------
# Agent.run — Reflection: N consecutive steps without an edit
# ---------------------------------------------------------------------------

class TestReflectionNoEdit:
    def test_reflection_triggered_after_no_edit_steps(self, tmp_path):
        from agent.task import EventType

        task = Task(
            task_id="noedit",
            description="explore forever",
            repo_path=str(tmp_path),
            max_steps=20,
        )
        log = EventLog.create(task, log_dir=str(tmp_path / "logs"))

        registry = ToolRegistry()
        registry.register(NoopTool("shell"))
        registry.register(NoopTool("file_read"))

        # Different cmd each step to avoid loop detection; 6 consecutive steps without file_write triggers no_edit
        script = [
            make_tool_call_action("shell", {"cmd": f"echo {i}"})
            for i in range(6)
        ] + [make_finish_action()]
        backend = MockBackend(script)
        config = AgentConfig(reflection_no_edit_steps=6, loop_detection_window=10)
        agent = Agent(backend, registry, config)

        agent.run(task, log)

        events = log.replay()
        reflection_events = [e for e in events if e.event_type == EventType.REFLECTION]
        no_edit_reflections = [e for e in reflection_events if e.payload["reason"] == "no_edit"]
        assert len(no_edit_reflections) >= 1
        log.close()


# ---------------------------------------------------------------------------
# Agent.run — LLM call exception
# ---------------------------------------------------------------------------

class TestLLMError:
    def test_llm_exception_returns_failed(self, task, log, registry):
        class CrashingBackend(MockBackend):
            def complete(self, messages, tools):
                raise ConnectionError("API unreachable")

        agent = Agent(CrashingBackend([]), registry)
        result = agent.run(task, log)

        assert result.status == RunStatus.FAILED
        assert "API unreachable" in result.error

    def test_llm_error_logged(self, task, log, registry):
        from agent.task import EventType

        class CrashingBackend(MockBackend):
            def complete(self, messages, tools):
                raise TimeoutError("timeout")

        agent = Agent(CrashingBackend([]), registry)
        agent.run(task, log)

        events = log.replay()
        assert events[-1].event_type == EventType.TASK_FAILED


# ---------------------------------------------------------------------------
# Agent.run — unknown tool
# ---------------------------------------------------------------------------

class TestUnknownTool:
    def test_unknown_tool_does_not_crash_agent(self, task, log):
        """When LLM calls a nonexistent tool, agent records the error in observation and continues."""
        script = [
            make_tool_call_action("nonexistent_tool"),
            make_finish_action(),
        ]
        backend = MockBackend(script)
        registry = ToolRegistry()   # empty registry, no tools registered
        agent = Agent(backend, registry)

        result = agent.run(task, log)
        # agent should not crash; it should eventually complete
        assert result.status == RunStatus.SUCCESS

    def test_unknown_tool_error_in_observation(self, task, log):
        from agent.task import EventType

        script = [
            make_tool_call_action("ghost"),
            make_finish_action(),
        ]
        backend = MockBackend(script)
        registry = ToolRegistry()
        agent = Agent(backend, registry)
        agent.run(task, log)

        events = log.replay()
        obs_events = [e for e in events if e.event_type == EventType.OBSERVATION]
        assert len(obs_events) >= 1
        obs = obs_events[0].payload["observation"]
        assert obs["status"] == "error"
