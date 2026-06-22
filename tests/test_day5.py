"""
tests/test_day5.py

Day 5 tests: RepoMap, TokenBudget, ConversationHistory, and core.py integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from context.history import ConversationHistory
from context.repo_map import RepoMap, _extract_python_symbols, _extract_symbols_regex
from context.token_budget import TokenBudget, estimate_tokens
from llm.base import LLMMessage, MockBackend
from agent.task import Action, ActionType, Task, ToolCall
from tools.base import NoopTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def py_repo(tmp_path) -> Path:
    """Sample repo containing several Python files."""
    (tmp_path / "main.py").write_text(
        "def run():\n    pass\n\nclass App:\n    def start(self):\n        pass\n"
    )
    (tmp_path / "utils.py").write_text(
        "def helper():\n    return 1\n\ndef another():\n    pass\n"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "module.py").write_text("class SubModule:\n    pass\n")
    # should be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-312.pyc").write_bytes(b"\x00" * 10)
    return tmp_path


# ===========================================================================
# estimate_tokens
# ===========================================================================

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") >= 1

    def test_short_string(self):
        assert estimate_tokens("hello") >= 1

    def test_longer_string(self):
        text = "a" * 400
        # Don't assert exact value, just that it's greater than a short string
        short = estimate_tokens("a" * 10)
        assert estimate_tokens(text) > short

    def test_proportional(self):
        # tiktoken compresses repeated characters, so growth is not strictly linear;
        # just assert that a longer text has a higher token count
        t1 = estimate_tokens("hello world " * 10)
        t2 = estimate_tokens("hello world " * 20)
        assert t2 > t1


# ===========================================================================
# RepoMap
# ===========================================================================

class TestRepoMap:
    def test_build_returns_string(self, py_repo):
        rm = RepoMap(py_repo)
        result = rm.build(budget=10_000)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_file_names(self, py_repo):
        rm = RepoMap(py_repo)
        result = rm.build(budget=10_000)
        assert "main.py" in result
        assert "utils.py" in result

    def test_contains_symbol_names(self, py_repo):
        rm = RepoMap(py_repo)
        result = rm.build(budget=10_000)
        assert "run" in result
        assert "App" in result
        assert "helper" in result

    def test_skips_pycache(self, py_repo):
        rm = RepoMap(py_repo)
        result = rm.build(budget=10_000)
        assert "__pycache__" not in result

    def test_budget_limits_output(self, py_repo):
        rm = RepoMap(py_repo)
        # Very small budget; output should be truncated
        result = rm.build(budget=10)
        assert len(result) <= 10 * 4 + 100  # leave some room for a truncation message

    def test_empty_repo(self, tmp_path):
        rm = RepoMap(tmp_path)
        result = rm.build()
        assert "empty" in result.lower()

    def test_nonexistent_path_graceful(self, tmp_path):
        # A nonexistent path should not crash
        rm = RepoMap(tmp_path / "no_such_dir")
        result = rm.build()
        assert isinstance(result, str)

    def test_large_file_skipped(self, tmp_path):
        # Files exceeding 500 KB should be skipped
        big = tmp_path / "big.py"
        big.write_bytes(b"x = 1\n" * 100_000)   # ~600 KB
        rm = RepoMap(tmp_path)
        result = rm.build()
        assert "big.py" not in result

    def test_query_aware_ranking_promotes_relevant_file(self, tmp_path):
        # A big, structurally "important" but task-irrelevant file
        (tmp_path / "payments.py").write_text(
            "def charge():\n    pass\n"
            + "\n".join(f"def f{i}():\n    pass" for i in range(20))
        )
        # A small file directly relevant to a 'token budget' task
        (tmp_path / "token_budget.py").write_text(
            "def trim_history():\n    pass\n\ndef estimate_tokens():\n    pass\n"
        )
        rm = RepoMap(tmp_path)

        # Without a query, the structurally important file ranks first.
        no_query = rm.build(budget=10_000)
        assert no_query.index("payments.py") < no_query.index("token_budget.py")

        # With a relevant query, the relevant file is promoted and marked.
        with_query = rm.build(budget=10_000, query="fix token budget trimming")
        assert with_query.index("token_budget.py") < with_query.index("payments.py")
        assert "★ token_budget.py" in with_query

    def test_query_aware_falls_back_when_no_match(self, tmp_path):
        # An empty/irrelevant query must not change behavior or add markers.
        (tmp_path / "a.py").write_text("def alpha():\n    pass\n")
        rm = RepoMap(tmp_path)
        assert rm.build(budget=10_000, query=None) == rm.build(budget=10_000)
        assert "★" not in rm.build(budget=10_000, query="zzzznomatch")


class TestExtractPythonSymbols:
    def test_extracts_function(self, tmp_path):
        code = "def foo():\n    pass\n"
        syms = _extract_python_symbols(code, Path("test.py"))
        names = [s.name for s in syms]
        assert "foo" in names

    def test_extracts_class(self, tmp_path):
        code = "class MyClass:\n    pass\n"
        syms = _extract_python_symbols(code, Path("test.py"))
        names = [s.name for s in syms]
        assert "MyClass" in names

    def test_method_has_indent(self, tmp_path):
        code = "class Foo:\n    def bar(self):\n        pass\n"
        syms = _extract_python_symbols(code, Path("test.py"))
        bar = next((s for s in syms if s.name == "bar"), None)
        assert bar is not None
        assert bar.indent > 0
        assert not bar.is_toplevel

    def test_toplevel_function_no_indent(self, tmp_path):
        code = "def baz():\n    pass\n"
        syms = _extract_python_symbols(code, Path("test.py"))
        baz = next((s for s in syms if s.name == "baz"), None)
        assert baz is not None
        assert baz.is_toplevel

    def test_line_numbers_correct(self, tmp_path):
        code = "# comment\ndef foo():\n    pass\n"
        syms = _extract_python_symbols(code, Path("test.py"))
        foo = next((s for s in syms if s.name == "foo"), None)
        assert foo is not None
        assert foo.line == 2

    def test_syntax_error_falls_back_to_regex(self, tmp_path):
        # Syntactically broken code should fall back to regex, not crash
        code = "def foo(\n    # broken syntax"
        syms = _extract_python_symbols(code, Path("test.py"))
        assert isinstance(syms, list)


class TestExtractSymbolsRegex:
    def test_extracts_def(self):
        code = "def my_func():\n    pass\n"
        syms = _extract_symbols_regex(code, Path("test.py"))
        assert any(s.name == "my_func" for s in syms)

    def test_extracts_class(self):
        code = "class MyClass:\n    pass\n"
        syms = _extract_symbols_regex(code, Path("test.py"))
        assert any(s.name == "MyClass" for s in syms)

    def test_javascript_function(self):
        code = "function myFunc() {\n    return 1;\n}\n"
        syms = _extract_symbols_regex(code, Path("test.js"))
        assert any(s.name == "myFunc" for s in syms)


# ===========================================================================
# TokenBudget
# ===========================================================================

class TestTokenBudget:
    def test_default_plan_sums_to_budget(self):
        budget = TokenBudget(total=80_000)
        plan = budget.default_plan()
        assert plan.total == 80_000
        assert plan.reserve > 0
        assert plan.system_core + plan.repo_map + plan.history + plan.observation <= plan.available

    def test_trim_to_short_text_unchanged(self):
        budget = TokenBudget()
        text = "hello world"
        assert budget.trim_to(text, token_limit=1000) == text

    def test_trim_to_long_text_truncated(self):
        budget = TokenBudget()
        text = "x" * 10_000
        result = budget.trim_to(text, token_limit=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_trim_history_short_unchanged(self):
        budget = TokenBudget()
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
        ]
        result = budget.trim_history(msgs, token_limit=10_000)
        assert len(result) == 2

    def test_trim_history_preserves_first_message(self):
        budget = TokenBudget(total=1000)
        # First message is short; many long messages follow
        msgs = [{"role": "user", "content": "task"}]
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 200})
        result = budget.trim_history(msgs, token_limit=50)
        assert result[0]["content"] == "task"

    def test_trim_history_keeps_recent_messages(self):
        budget = TokenBudget()
        msgs = [{"role": "user", "content": "task"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"message {i}"})
        # Budget is only enough for 3 messages
        result = budget.trim_history(msgs, token_limit=15)
        contents = [m["content"] for m in result]
        # The most recent message (message 9) should be present
        assert any("message 9" in c for c in contents)

    def test_trim_history_adds_truncation_notice(self):
        budget = TokenBudget()
        msgs = [{"role": "user", "content": "task"}]
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 100})
        result = budget.trim_history(msgs, token_limit=50)
        # There should be a truncation notice message
        contents = " ".join(m["content"] for m in result)
        assert "truncated" in contents.lower()

    def test_trim_history_empty(self):
        budget = TokenBudget()
        assert budget.trim_history([], token_limit=1000) == []

    def test_usage_report(self):
        budget = TokenBudget(total=10_000)
        report = budget.usage_report("system", "repo", [], "obs")
        assert "system" in report
        assert "total" in report
        assert report["budget"] == 10_000


# ===========================================================================
# ConversationHistory
# ===========================================================================

class TestConversationHistory:
    def test_add_and_retrieve(self):
        h = ConversationHistory(max_messages=10)
        h.add(LLMMessage(role="user", content="hello"))
        assert h.message_count == 1
        assert h.to_list()[0].content == "hello"

    def test_sliding_window_drops_oldest(self):
        h = ConversationHistory(max_messages=3)
        h.add(LLMMessage(role="user", content="first"))
        h.add(LLMMessage(role="user", content="second"))
        h.add(LLMMessage(role="user", content="third"))
        h.add(LLMMessage(role="user", content="fourth"))
        # Max 3 messages; "first" should be dropped (except index 0)
        assert h.message_count == 3
        contents = [m.content for m in h.to_list()]
        assert "first" in contents      # index 0 is never dropped
        assert "second" not in contents  # dropped
        assert "fourth" in contents

    def test_first_message_never_dropped(self):
        h = ConversationHistory(max_messages=2)
        h.add(LLMMessage(role="user", content="task_description"))
        for i in range(10):
            h.add(LLMMessage(role="user", content=f"msg_{i}"))
        assert h.to_list()[0].content == "task_description"

    def test_to_dicts(self):
        h = ConversationHistory()
        h.add(LLMMessage(role="user", content="hello"))
        dicts = h.to_dicts()
        assert dicts == [{"role": "user", "content": "hello"}]

    def test_from_dicts(self):
        dicts = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "ok"}]
        h = ConversationHistory.from_dicts(dicts)
        assert h.message_count == 2
        assert h.to_list()[0].content == "task"

    def test_add_many(self):
        h = ConversationHistory(max_messages=5)
        msgs = [LLMMessage(role="user", content=f"m{i}") for i in range(3)]
        h.add_many(msgs)
        assert h.message_count == 3

    def test_clear_except_first(self):
        h = ConversationHistory()
        h.add(LLMMessage(role="user", content="task"))
        h.add(LLMMessage(role="assistant", content="ok"))
        h.add(LLMMessage(role="user", content="more"))
        h.clear_except_first()
        assert h.message_count == 1
        assert h.to_list()[0].content == "task"

    def test_last_message(self):
        h = ConversationHistory()
        h.add(LLMMessage(role="user", content="first"))
        h.add(LLMMessage(role="assistant", content="last"))
        assert h.last_message.content == "last"

    def test_empty_history_last_message_is_none(self):
        h = ConversationHistory()
        assert h.last_message is None

    def test_len(self):
        h = ConversationHistory()
        h.add(LLMMessage(role="user", content="x"))
        assert len(h) == 1


# ===========================================================================
# core.py integration: context modules work correctly after wiring in
# ===========================================================================

class TestCoreWithContext:
    def _make_task(self, tmp_path) -> Task:
        return Task(
            task_id="ctx001",
            description="Fix the bug",
            repo_path=str(tmp_path),
            max_steps=5,
            budget_tokens=80_000,
        )

    def test_run_with_context_succeeds(self, tmp_path):
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog

        task = self._make_task(tmp_path)
        registry = ToolRegistry().register(NoopTool("shell"))
        script = [
            Action(ActionType.TOOL_CALL, "explore", ToolCall("shell", {"cmd": "ls"})),
            Action(ActionType.FINISH, "done", message="Task complete"),
        ]
        backend = MockBackend(script)
        config = AgentConfig(budget_tokens=80_000, history_max_messages=20)
        agent = Agent(backend, registry, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()

    def test_repo_map_injected_in_system_prompt(self, tmp_path):
        """Content generated by repo_map should appear in the system prompt sent to the LLM."""
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog

        # Put a Python file in the repo so repo_map has content
        (tmp_path / "mymodule.py").write_text("def my_function():\n    pass\n")

        task = self._make_task(tmp_path)
        registry = ToolRegistry().register(NoopTool("shell"))
        script = [Action(ActionType.FINISH, "done", message="ok")]
        backend = MockBackend(script)
        config = AgentConfig(budget_tokens=80_000)
        agent = Agent(backend, registry, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            agent.run(task, log)

        # Check that the first system message received by the LLM contains repo content
        assert backend.call_count >= 1
        first_messages = backend.received_messages[0]
        system_content = next(
            (m.content for m in first_messages if m.role == "system"), ""
        )
        assert "mymodule.py" in system_content or "my_function" in system_content

    def test_large_history_trimmed(self, tmp_path):
        """When history is very long, TokenBudget should trim it rather than crash."""
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog

        task = self._make_task(tmp_path)
        registry = ToolRegistry().register(NoopTool("shell"))

        # Create 40 tool_call steps each with a very long observation
        class BigOutputTool(NoopTool):
            def execute(self, params):
                from tools.base import ToolResult
                return ToolResult(success=True, output="x" * 2000)

        registry2 = ToolRegistry().register(BigOutputTool("shell"))
        script = [
            Action(ActionType.TOOL_CALL, f"step {i}", ToolCall("shell", {"cmd": f"echo {i}"}))
            for i in range(4)
        ] + [Action(ActionType.FINISH, "done", message="ok")]

        backend = MockBackend(script)
        config = AgentConfig(budget_tokens=10_000, history_max_messages=10)
        agent = Agent(backend, registry2, config)

        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()
