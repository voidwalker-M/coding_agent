"""
tests/test_day7.py

Day 7 tests: retry logic, git diff patch, repo_map cache isolation, and
end-to-end integration.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Action, ActionType, RunStatus, Task, ToolCall
from llm.base import LLMMessage, LLMResponse, LLMToolSchema, MockBackend
from tools.base import NoopTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task(tmp_path) -> Task:
    return Task(
        task_id="day7test",
        description="Fix the bug",
        repo_path=str(tmp_path),
        max_steps=10,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry().register(NoopTool("shell"))


@pytest.fixture
def git_repo(tmp_path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    f = tmp_path / "main.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


# ===========================================================================
# _call_with_retry
# ===========================================================================

class TestCallWithRetry:

    def _make_agent(self, backend, config=None) -> Agent:
        registry = ToolRegistry().register(NoopTool("shell"))
        return Agent(backend, registry, config)

    def test_success_on_first_attempt(self, tmp_path):
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = MockBackend(script)
        agent = self._make_agent(backend)

        msgs = [LLMMessage(role="user", content="go")]
        result = agent._call_with_retry(msgs, [])
        assert result.action.action_type == ActionType.FINISH

    def test_retries_on_network_error(self, tmp_path):
        """Network errors should trigger retries and ultimately succeed."""
        call_count = 0

        class FlakyBackend(MockBackend):
            def complete(self, messages, tools):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("network timeout")
                return super().complete(messages, tools)

        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = FlakyBackend(script)
        config = AgentConfig(llm_max_retries=3, llm_retry_delay=0.01)
        agent = self._make_agent(backend, config)

        msgs = [LLMMessage(role="user", content="go")]
        result = agent._call_with_retry(msgs, [])
        assert result.action.action_type == ActionType.FINISH
        assert call_count == 3

    def test_raises_after_max_retries(self, tmp_path):
        """An exception should be raised after exceeding the maximum retry count."""
        class AlwaysFailBackend(MockBackend):
            def complete(self, messages, tools):
                raise ConnectionError("always fails")

        backend = AlwaysFailBackend([])
        config = AgentConfig(llm_max_retries=2, llm_retry_delay=0.01)
        agent = self._make_agent(backend, config)

        with pytest.raises(ConnectionError):
            agent._call_with_retry([LLMMessage(role="user", content="go")], [])

    def test_no_retry_on_auth_error(self, tmp_path):
        """Authentication errors should not be retried; raise immediately."""
        call_count = 0

        class AuthFailBackend(MockBackend):
            def complete(self, messages, tools):
                nonlocal call_count
                call_count += 1
                raise PermissionError("401 invalid api key")

        backend = AuthFailBackend([])
        config = AgentConfig(llm_max_retries=3, llm_retry_delay=0.01)
        agent = self._make_agent(backend, config)

        with pytest.raises(PermissionError):
            agent._call_with_retry([LLMMessage(role="user", content="go")], [])

        assert call_count == 1   # called only once, no retry

    def test_no_retry_on_400(self, tmp_path):
        """400 Bad Request should not be retried."""
        call_count = 0

        class BadRequestBackend(MockBackend):
            def complete(self, messages, tools):
                nonlocal call_count
                call_count += 1
                raise ValueError("400 bad request: invalid parameters")

        backend = BadRequestBackend([])
        config = AgentConfig(llm_max_retries=3, llm_retry_delay=0.01)
        agent = self._make_agent(backend, config)

        with pytest.raises(ValueError):
            agent._call_with_retry([LLMMessage(role="user", content="go")], [])

        assert call_count == 1

    def test_exponential_backoff(self, tmp_path):
        """Verify that retry delays grow exponentially."""
        sleep_calls = []
        call_count = 0

        class FlakyBackend(MockBackend):
            def complete(self, messages, tools):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("timeout")
                return super().complete(messages, tools)

        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = FlakyBackend(script)
        config = AgentConfig(llm_max_retries=3, llm_retry_delay=1.0)
        agent = self._make_agent(backend, config)

        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            agent._call_with_retry([LLMMessage(role="user", content="go")], [])

        assert len(sleep_calls) == 2
        assert sleep_calls[1] == sleep_calls[0] * 2   # exponential backoff


# ===========================================================================
# _get_git_diff
# ===========================================================================

class TestGetGitDiff:

    def _make_agent(self) -> Agent:
        registry = ToolRegistry().register(NoopTool("shell"))
        return Agent(MockBackend([]), registry)

    def test_returns_diff_after_modification(self, git_repo):
        agent = self._make_agent()
        # Modify file without committing
        (git_repo / "main.py").write_text("x = 2\n")
        diff = agent._get_git_diff(str(git_repo))
        assert diff is not None
        assert "-x = 1" in diff or "+x = 2" in diff

    def test_returns_none_when_no_changes(self, git_repo):
        agent = self._make_agent()
        diff = agent._get_git_diff(str(git_repo))
        assert diff is None

    def test_returns_none_on_non_git_dir(self, tmp_path):
        agent = self._make_agent()
        diff = agent._get_git_diff(str(tmp_path))
        assert diff is None   # not a git repo, should not crash

    def test_patch_included_in_run_result(self, git_repo):
        """RunResult.patch should contain the diff after agent FINISH."""
        # Modify the file (simulating an agent file_write operation)
        (git_repo / "main.py").write_text("x = 99\n")

        task = Task(
            task_id="patchtest",
            description="fix",
            repo_path=str(git_repo),
            max_steps=5,
        )
        registry = ToolRegistry().register(NoopTool("shell"))
        script = [Action(ActionType.FINISH, "done", message="Fixed it")]
        backend = MockBackend(script)
        agent = Agent(backend, registry)

        with EventLog.create(task, log_dir=str(git_repo / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        assert result.patch is not None
        assert "main.py" in result.patch


# ===========================================================================
# repo_map cache isolation
# ===========================================================================

class TestRepoMapCache:

    def test_cache_reused_within_same_run(self, tmp_path):
        """Within a single run, repo_map should be built only once."""
        (tmp_path / "mod.py").write_text("def foo(): pass\n")

        task = Task(
            task_id="cache1",
            description="fix",
            repo_path=str(tmp_path),
            max_steps=5,
        )
        registry = ToolRegistry().register(NoopTool("shell"))
        script = [
            Action(ActionType.TOOL_CALL, "step1", ToolCall("shell", {"cmd": "ls"})),
            Action(ActionType.TOOL_CALL, "step2", ToolCall("shell", {"cmd": "pwd"})),
            Action(ActionType.FINISH, "done", message="ok"),
        ]
        backend = MockBackend(script)

        build_call_count = 0
        original_build = __import__("context.repo_map", fromlist=["RepoMap"]).RepoMap.build

        def counting_build(self, budget=8000, query=None):
            nonlocal build_call_count
            build_call_count += 1
            return original_build(self, budget, query)

        with patch("context.repo_map.RepoMap.build", counting_build):
            agent = Agent(backend, registry)
            with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
                agent.run(task, log)

        assert build_call_count == 1   # built only once

    def test_cache_reset_on_new_repo(self, tmp_path):
        """The cache should reset automatically when repo_path changes."""
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()

        registry = ToolRegistry().register(NoopTool("shell"))
        backend = MockBackend([
            Action(ActionType.FINISH, "done_a", message="ok_a"),
            Action(ActionType.FINISH, "done_b", message="ok_b"),
        ])
        agent = Agent(backend, registry)

        task_a = Task(task_id="ca", description="a", repo_path=str(repo_a), max_steps=5)
        task_b = Task(task_id="cb", description="b", repo_path=str(repo_b), max_steps=5)

        with EventLog.create(task_a, log_dir=str(tmp_path / "logs")) as log_a:
            agent.run(task_a, log_a)
        cache_after_a = agent._repo_map_cache

        with EventLog.create(task_b, log_dir=str(tmp_path / "logs")) as log_b:
            agent.run(task_b, log_b)
        cache_after_b = agent._repo_map_cache

        # The two runs have different cache keys (confirming the key switched)
        assert agent._repo_map_cache_key == str(repo_b)


# ===========================================================================
# Integration: full run + statistics
# ===========================================================================

class TestIntegration:

    def test_full_run_with_all_features(self, tmp_path):
        """End-to-end: repo_map + token_budget + reflection + retry all active."""
        (tmp_path / "main.py").write_text("def broken(): pass\n")
        (tmp_path / "utils.py").write_text("def helper(): return 1\n")

        task = Task(
            task_id="integration1",
            description="Fix the broken function",
            repo_path=str(tmp_path),
            max_steps=8,
            budget_tokens=40_000,
        )

        from tools.file_tool import FileReadTool, FileWriteTool
        registry = (
            ToolRegistry()
            .register(NoopTool("shell"))
            .register(FileReadTool())
            .register(FileWriteTool())
        )

        script = [
            Action(ActionType.TOOL_CALL, "read", ToolCall("file_read", {"path": str(tmp_path / "main.py")})),
            Action(ActionType.TOOL_CALL, "write", ToolCall("file_write", {
                "path": str(tmp_path / "main.py"),
                "content": "def broken():\n    return 42\n",
            })),
            Action(ActionType.FINISH, "Fixed broken()", message="Added return value"),
        ]
        backend = MockBackend(script)
        config = AgentConfig(
            max_steps=8,
            budget_tokens=40_000,
            llm_max_retries=1,
            llm_retry_delay=0.01,
        )
        agent = Agent(backend, registry, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        assert result.steps_taken == 3
        assert "return value" in result.summary.lower() or "Fixed" in result.summary

        # Verify the file was actually written
        content = (tmp_path / "main.py").read_text()
        assert "return 42" in content

    def test_retry_transparent_to_caller(self, tmp_path):
        """Retries are completely transparent to run() callers; the final result matches a no-retry run."""
        attempt = 0

        class OnceFailBackend(MockBackend):
            def complete(self, messages, tools):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise ConnectionError("first attempt fails")
                return super().complete(messages, tools)

        task = Task(
            task_id="retry_transparent",
            description="fix",
            repo_path=str(tmp_path),
            max_steps=5,
        )
        registry = ToolRegistry().register(NoopTool("shell"))
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = OnceFailBackend(script)
        config = AgentConfig(llm_max_retries=2, llm_retry_delay=0.01)
        agent = Agent(backend, registry, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
        assert attempt == 2   # first attempt failed, second succeeded

    def test_event_log_stats_accurate(self, tmp_path):
        """Event log statistics should be accurate."""
        from agent.event_log import summarize_run
        from tools.base import FailingTool

        task = Task(
            task_id="statstest",
            description="fix",
            repo_path=str(tmp_path),
            max_steps=10,
        )
        registry = (
            ToolRegistry()
            .register(FailingTool("test"))
            .register(NoopTool("shell"))
        )
        script = [
            Action(ActionType.TOOL_CALL, "run tests", ToolCall("test", {})),
            Action(ActionType.TOOL_CALL, "explore", ToolCall("shell", {"cmd": "ls"})),
            Action(ActionType.FINISH, "done", message="ok"),
        ]
        backend = MockBackend(script)
        config = AgentConfig(test_tool_names=("test",))
        agent = Agent(backend, registry, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)
            stats = summarize_run(log)

        assert result.is_success()
        assert stats["actions"] == 3
        assert stats["reflections"] == 1      # test failure triggers one reflection
        assert stats["tool_calls"]["test"] == 1
        assert stats["tool_calls"]["shell"] == 1
        assert stats["final_status"] == "task_complete"


# ===========================================================================
# pyproject.toml integrity check
# ===========================================================================

class TestPyprojectToml:
    def test_pyproject_has_entry_point(self):
        content = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert "agent = " in content
        assert "entry.cli:main" in content

    def test_pyproject_has_dev_extras(self):
        content = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert "[project.optional-dependencies]" in content
        assert "pytest" in content

    def test_pyproject_has_full_extras(self):
        content = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert "tiktoken" in content
        assert "tree-sitter-javascript" in content
