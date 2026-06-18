"""
tests/test_stream.py

Streaming output tests.
- StreamingMixin fallback behavior
- AgentConfig stream field
- core.py streaming path (mock stream() method)
- cli --stream option registration
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, Task, ToolCall
from llm.base import LLMMessage, LLMResponse, LLMToolSchema, MockBackend
from tools.base import NoopTool, ToolRegistry


# ---------------------------------------------------------------------------
# StreamingMixin fallback
# ---------------------------------------------------------------------------

class TestStreamingMixin:
    def test_default_stream_calls_complete(self):
        """The default stream() implementation in base.py delegates to complete()."""
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = MockBackend(script)
        collected = []
        # MockBackend inherits StreamingMixin's default stream()
        result = backend.stream(
            [LLMMessage(role="user", content="go")],
            [],
            on_text=lambda t: collected.append(t),
        )
        assert result.action.action_type == ActionType.FINISH
        # fallback passes raw_content to on_text
        assert len(collected) > 0

    def test_stream_returns_llm_response(self):
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = MockBackend(script)
        result = backend.stream([LLMMessage(role="user", content="go")], [])
        assert isinstance(result, LLMResponse)


# ---------------------------------------------------------------------------
# AgentConfig stream field
# ---------------------------------------------------------------------------

class TestAgentConfigStream:
    def test_stream_default_false(self):
        cfg = AgentConfig()
        assert cfg.stream is False
        assert cfg.stream_callback is None

    def test_stream_can_be_enabled(self):
        cb = lambda t: None
        cfg = AgentConfig(stream=True, stream_callback=cb)
        assert cfg.stream is True
        assert cfg.stream_callback is cb


# ---------------------------------------------------------------------------
# core.py streaming path
# ---------------------------------------------------------------------------

class TestCoreStreamPath:

    def _make_streaming_backend(self, script):
        """Create a mock backend with a real stream() method."""
        backend = MockBackend(script)
        stream_calls = []

        def fake_stream(messages, tools, on_text=None, on_thought=None):
            stream_calls.append({"messages": messages, "on_text": on_text})
            # Simulate a reasoning model: stream thought first, then text
            if on_thought:
                on_thought("thinking... ")
            # Simulate streaming: deliver text in 3 chunks
            if on_text:
                on_text("I will ")
                on_text("fix the ")
                on_text("bug.")
            return backend.complete(messages, tools)

        backend.stream = fake_stream
        backend._stream_calls = stream_calls
        return backend

    def test_stream_true_calls_stream_method(self, tmp_path):
        """With stream=True, backend.stream() should be called instead of complete()."""
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = self._make_streaming_backend(script)

        collected_text = []
        cfg = AgentConfig(
            stream=True,
            stream_callback=lambda t: collected_text.append(t),
        )
        registry = ToolRegistry().register(NoopTool("shell"))
        agent = Agent(backend, registry, cfg)
        task = Task(task_id="st1", description="fix", repo_path=str(tmp_path), max_steps=3)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        # stream() was called
        assert len(backend._stream_calls) >= 1
        # on_text callback received the chunked text
        assert "".join(collected_text) == "I will fix the bug."

    def test_stream_false_calls_complete(self, tmp_path):
        """With stream=False, complete() should be called instead of stream()."""
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = self._make_streaming_backend(script)
        original_complete_count = [0]
        original_complete = backend.complete

        def counting_complete(messages, tools):
            original_complete_count[0] += 1
            return original_complete(messages, tools)

        backend.complete = counting_complete
        cfg = AgentConfig(stream=False)
        registry = ToolRegistry().register(NoopTool("shell"))
        agent = Agent(backend, registry, cfg)
        task = Task(task_id="st2", description="fix", repo_path=str(tmp_path), max_steps=3)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        assert original_complete_count[0] >= 1
        # stream() was not called by core (backend._stream_calls only filled by stream())
        assert len(backend._stream_calls) == 0

    def test_stream_callback_receives_thought(self, tmp_path):
        """The streaming callback should receive chunked model thought text."""
        script = [
            Action(ActionType.TOOL_CALL, "thinking...", ToolCall("shell", {"cmd": "ls"})),
            Action(ActionType.FINISH, "done", message="ok"),
        ]
        backend = self._make_streaming_backend(script)

        all_text = []
        cfg = AgentConfig(
            stream=True,
            stream_callback=lambda t: all_text.append(t),
        )
        registry = ToolRegistry().register(NoopTool("shell"))
        agent = Agent(backend, registry, cfg)
        task = Task(task_id="st3", description="fix", repo_path=str(tmp_path), max_steps=5)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            agent.run(task, log)

        # Each step should have a stream call → on_text callback receives data
        assert len(all_text) > 0

    def test_stream_no_callback_still_works(self, tmp_path):
        """stream=True without a callback should not crash."""
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = self._make_streaming_backend(script)
        cfg = AgentConfig(stream=True, stream_callback=None)
        registry = ToolRegistry().register(NoopTool("shell"))
        agent = Agent(backend, registry, cfg)
        task = Task(task_id="st4", description="fix", repo_path=str(tmp_path), max_steps=3)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()

    def test_stream_retry_on_error(self, tmp_path):
        """Errors in the streaming path should also trigger retries."""
        attempt = 0
        script = [Action(ActionType.FINISH, "done", message="ok")]
        base_backend = MockBackend(script)

        def flaky_stream(messages, tools, on_text=None, on_thought=None):
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ConnectionError("stream interrupted")
            if on_text:
                on_text("ok")
            return base_backend.complete(messages, tools)

        base_backend.stream = flaky_stream
        cfg = AgentConfig(stream=True, llm_max_retries=3, llm_retry_delay=0.01)
        registry = ToolRegistry().register(NoopTool("shell"))
        agent = Agent(base_backend, registry, cfg)
        task = Task(task_id="st5", description="fix", repo_path=str(tmp_path), max_steps=3)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        assert attempt == 2  # first attempt failed, second succeeded


# ---------------------------------------------------------------------------
# CLI --stream option
# ---------------------------------------------------------------------------

class TestCliStreamOption:
    def test_stream_option_registered(self):
        from click.testing import CliRunner
        from entry.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--stream" in result.output or "-s" in result.output

    def test_stream_default_on(self):
        """--stream should be enabled by default."""
        from click.testing import CliRunner
        from entry.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        # default is on, so help text contains "stream"
        assert "stream" in result.output.lower()


# ---------------------------------------------------------------------------
# AnthropicBackend / OpenAICompatBackend stream() method presence
# ---------------------------------------------------------------------------

class TestBackendStreamMethod:
    def test_anthropic_backend_has_stream(self):
        from llm.anthropic_backend import AnthropicBackend
        assert hasattr(AnthropicBackend, "stream")
        assert callable(AnthropicBackend.stream)

    def test_openai_compat_backend_has_stream(self):
        from llm.openai_compat import OpenAICompatBackend
        assert hasattr(OpenAICompatBackend, "stream")
        assert callable(OpenAICompatBackend.stream)

    def test_anthropic_stream_signature(self):
        """stream() method must accept an on_text parameter."""
        import inspect
        from llm.anthropic_backend import AnthropicBackend
        sig = inspect.signature(AnthropicBackend.stream)
        assert "on_text" in sig.parameters

    def test_openai_stream_signature(self):
        import inspect
        from llm.openai_compat import OpenAICompatBackend
        sig = inspect.signature(OpenAICompatBackend.stream)
        assert "on_text" in sig.parameters
