"""
tests/test_eval.py

Evaluation harness tests. MockBackend scripts drive real file/test tools
without consuming any API quota.

Key verifications:
- Verifier grading correctness
- Harness end-to-end: agent solves task → verifier returns PASS
- **Verifier independence**: agent self-reports FINISH without really solving → still FAIL
- Report aggregation (success_rate / avg_steps / JSON)
"""

import json

import pytest

from agent.core import Agent, AgentConfig
from agent.task import Action, ActionType, ToolCall
from eval.harness import EvalHarness, EvalReport, EvalResult, TaskSpec
from eval.verifiers import (
    AllOfVerifier,
    CommandVerifier,
    FileContainsVerifier,
    FileExistsVerifier,
    PytestVerifier,
)
from llm.base import MockBackend
from tools.base import ToolRegistry
from tools.file_tool import FileWriteTool
from tools.test_tool import PytestTool


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

def test_file_exists_verifier(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    assert FileExistsVerifier("a.txt")(str(tmp_path))[0] is True
    assert FileExistsVerifier("missing.txt")(str(tmp_path))[0] is False


def test_file_contains_verifier(tmp_path):
    (tmp_path / "h.py").write_text("print('hello world')")
    assert FileContainsVerifier("h.py", "hello world")(str(tmp_path))[0] is True
    assert FileContainsVerifier("h.py", "goodbye")(str(tmp_path))[0] is False


def test_command_verifier(tmp_path):
    (tmp_path / "hello.py").write_text("print('hello world')")
    ok, _ = CommandVerifier("python hello.py", expect_substring="hello world")(str(tmp_path))
    assert ok is True
    bad, _ = CommandVerifier("python hello.py", expect_substring="nope")(str(tmp_path))
    assert bad is False


def test_pytest_verifier_pass_and_fail(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1 == 1\n")
    assert PytestVerifier()(str(tmp_path))[0] is True
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n")
    assert PytestVerifier()(str(tmp_path))[0] is False


def test_all_of_verifier(tmp_path):
    (tmp_path / "f.py").write_text("def foo(): return 1")
    v = AllOfVerifier(FileExistsVerifier("f.py"), FileContainsVerifier("f.py", "def foo"))
    assert v(str(tmp_path))[0] is True
    v2 = AllOfVerifier(FileExistsVerifier("f.py"), FileContainsVerifier("f.py", "def bar"))
    assert v2(str(tmp_path))[0] is False


# ---------------------------------------------------------------------------
# Harness end-to-end
# ---------------------------------------------------------------------------

def _bugfix_spec():
    return TaskSpec(
        id="bugfix_add",
        description="Fix add() in calc.py so tests pass.",
        setup_files={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        verify=PytestVerifier(),
        max_steps=5,
    )


def _solving_factory():
    """Returns an agent that correctly fixes calc.py then calls FINISH."""
    correct = "def add(a, b):\n    return a + b\n"
    script = [
        Action(ActionType.TOOL_CALL, "fix the bug",
               tool_call=ToolCall("file_write", {"path": "calc.py", "content": correct})),
        Action(ActionType.FINISH, "fixed", message="done"),
    ]

    def factory(spec, repo_path):
        registry = ToolRegistry().register(FileWriteTool()).register(PytestTool())
        return Agent(MockBackend(script), registry, AgentConfig(max_steps=spec.max_steps))

    return factory


def test_harness_passes_when_agent_solves(tmp_path):
    harness = EvalHarness(_solving_factory(), results_dir=str(tmp_path / "runs"))
    result = harness.run_task(_bugfix_spec())
    assert result.passed is True
    assert result.agent_status == "success"
    assert result.steps == 2
    assert result.error is None


def test_verifier_independent_of_agent_self_report(tmp_path):
    """
    Core principle: agent self-reports FINISH without actually fixing the bug.
    The harness must use the verifier as ground truth → passed=False, even
    though agent_status=success.
    """
    def lazy_factory(spec, repo_path):
        script = [Action(ActionType.FINISH, "I think it's fine", message="done")]
        registry = ToolRegistry().register(FileWriteTool()).register(PytestTool())
        return Agent(MockBackend(script), registry, AgentConfig(max_steps=spec.max_steps))

    harness = EvalHarness(lazy_factory, results_dir=str(tmp_path / "runs"))
    result = harness.run_task(_bugfix_spec())
    assert result.agent_status == "success"   # agent believes it succeeded
    assert result.passed is False             # but objective verification fails


def test_harness_handles_agent_crash(tmp_path):
    class CrashBackend(MockBackend):
        def complete(self, messages, tools):
            raise RuntimeError("boom")

    def crash_factory(spec, repo_path):
        registry = ToolRegistry().register(FileWriteTool())
        return Agent(CrashBackend([]), registry, AgentConfig(max_steps=3, llm_max_retries=1))

    harness = EvalHarness(crash_factory, results_dir=str(tmp_path / "runs"))
    result = harness.run_task(_bugfix_spec())
    assert result.passed is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------

def test_run_suite_and_report(tmp_path):
    solved = _bugfix_spec()
    unsolved = TaskSpec(
        id="unsolved",
        description="create missing.py",
        setup_files={},
        verify=FileExistsVerifier("missing.py"),  # agent won't create it
        max_steps=3,
    )

    correct = "def add(a, b):\n    return a + b\n"
    scripts = {
        "bugfix_add": [
            Action(ActionType.TOOL_CALL, "fix",
                   tool_call=ToolCall("file_write", {"path": "calc.py", "content": correct})),
            Action(ActionType.FINISH, "done", message="done"),
        ],
        "unsolved": [Action(ActionType.FINISH, "nothing to do", message="done")],
    }

    def factory(spec, repo_path):
        registry = ToolRegistry().register(FileWriteTool()).register(PytestTool())
        return Agent(MockBackend(scripts[spec.id]), registry, AgentConfig(max_steps=spec.max_steps))

    harness = EvalHarness(factory, results_dir=str(tmp_path / "runs"))
    report = harness.run_suite([solved, unsolved])

    assert report.total == 2
    assert report.passed == 1
    assert report.success_rate == 0.5
    assert report.avg_steps > 0
    assert "Success rate: 1/2" in report.format_table()


def test_report_json_roundtrip(tmp_path):
    report = EvalReport(results=[
        EvalResult("t1", True, "success", 3, 500, 1.2, "ok"),
        EvalResult("t2", False, "max_steps", 5, 900, 2.0, "tests failed"),
    ])
    out = tmp_path / "report.json"
    report.save_json(out)
    data = json.loads(out.read_text())
    assert data["summary"]["total"] == 2
    assert data["summary"]["passed"] == 1
    assert data["summary"]["success_rate"] == 0.5
    assert len(data["results"]) == 2
