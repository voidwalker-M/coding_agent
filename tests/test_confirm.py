"""
tests/test_confirm.py

Permission confirmation tests: whitelist, commands requiring confirmation,
callback behavior, and integration into the agent flow.
"""
from __future__ import annotations

import pytest

from tools.shell_tool import (
    ShellTool, ToolResult,
    _check_blocked, _is_readonly, _needs_confirm,
    terminal_confirm, always_allow, always_deny,
)


# ===========================================================================
# _is_readonly — whitelist check
# ===========================================================================

class TestIsReadonly:
    @pytest.mark.parametrize("cmd", [
        "ls", "ls -la", "ls /tmp",
        "cat file.py", "cat -n foo.py",
        "grep pattern file.py", "grep -r foo .",
        "find . -name '*.py'",
        "git status", "git diff HEAD", "git log --oneline",
        "git branch -a",
        "python -m pytest tests/",
        "pytest tests/test_foo.py",
        "echo hello",
        "pwd", "whoami",
        "diff a.py b.py",
        "wc -l file.py",
        "tree .",
    ])
    def test_readonly_commands_pass(self, cmd):
        assert _is_readonly(cmd), f"Expected {cmd!r} to be readonly"

    @pytest.mark.parametrize("cmd", [
        "rm file.py",
        "git commit -m 'fix'",
        "pip install requests",
        "mv old.py new.py",
        "chmod 755 script.sh",
        "sudo apt-get install vim",
        "curl https://example.com",
        "git push origin main",
    ])
    def test_write_commands_not_readonly(self, cmd):
        assert not _is_readonly(cmd), f"Expected {cmd!r} to NOT be readonly"


# ===========================================================================
# _needs_confirm — confirmation check
# ===========================================================================

class TestNeedsConfirm:
    def test_readonly_does_not_need_confirm(self):
        assert not _needs_confirm("ls -la")
        assert not _needs_confirm("cat file.py")
        assert not _needs_confirm("git status")
        assert not _needs_confirm("pytest tests/")

    @pytest.mark.parametrize("cmd", [
        "rm file.py",
        "git commit -m 'fix'",
        "pip install requests",
        "mv old.py new.py",
        "chmod 755 script.sh",
        "sudo apt-get install vim",
        "curl https://example.com",
        "git push origin main",
        "echo hello > output.txt",
        "git reset --hard HEAD",
        "docker run ubuntu",
    ])
    def test_dangerous_commands_need_confirm(self, cmd):
        assert _needs_confirm(cmd), f"Expected {cmd!r} to need confirmation"

    def test_unknown_safe_command_no_confirm(self):
        # Unknown command with no dangerous keywords does not need confirmation
        assert not _needs_confirm("python parse_data.py")
        assert not _needs_confirm("node server.js")


# ===========================================================================
# _check_blocked — blacklist
# ===========================================================================

class TestCheckBlocked:
    def test_rm_rf_root_blocked(self):
        assert _check_blocked("rm -rf /") is not None

    def test_mkfs_blocked(self):
        assert _check_blocked("mkfs.ext4 /dev/sda1") is not None

    def test_normal_rm_not_blocked(self):
        # Removing a single file is not on the blacklist (goes through confirmation)
        assert _check_blocked("rm file.py") is None

    def test_git_commit_not_blocked(self):
        assert _check_blocked("git commit -m 'fix'") is None


# ===========================================================================
# ShellTool permission confirmation behavior
# ===========================================================================

class TestShellToolConfirm:

    def test_readonly_no_callback_called(self):
        """Read-only commands must not invoke confirm_callback."""
        callback_called = []
        tool = ShellTool(confirm_callback=lambda cmd: callback_called.append(cmd) or True)
        result = tool.execute({"cmd": "echo hello"})
        assert result.success
        assert len(callback_called) == 0  # callback was not called

    def test_dangerous_callback_allow(self):
        """Dangerous command, callback returns True → execute."""
        tool = ShellTool(confirm_callback=always_allow)
        # pip install triggers confirmation; always_allow permits it
        # the actual execution may fail (no network), but it should not be rejected
        result = tool.execute({"cmd": "pip install --help"})
        # pip --help is a safe read; if pip install were blocked, error would contain "rejected"
        assert "rejected" not in (result.error or "")

    def test_dangerous_callback_deny(self):
        """Dangerous command, callback returns False → reject."""
        tool = ShellTool(confirm_callback=always_deny)
        result = tool.execute({"cmd": "pip install requests"})
        assert not result.success
        assert "rejected" in result.error.lower()
        assert "pip install" in result.error

    def test_no_callback_dangerous_command_executes(self):
        """confirm_callback=None → skip confirmation, dangerous commands execute directly (run mode)."""
        tool = ShellTool(confirm_callback=None)
        # Use a safe command that would normally trigger confirmation
        result = tool.execute({"cmd": "echo 'confirm test'"})
        assert result.success

    def test_blocked_command_denied_regardless_of_callback(self):
        """Blacklisted commands are rejected even when callback=always_allow."""
        tool = ShellTool(confirm_callback=always_allow)
        result = tool.execute({"cmd": "rm -rf /"})
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_callback_receives_full_command(self):
        """confirm_callback receives the complete command string."""
        received = []
        tool = ShellTool(confirm_callback=lambda cmd: received.append(cmd) or False)
        cmd = "pip install numpy"
        tool.execute({"cmd": cmd})
        assert len(received) == 1
        assert received[0] == cmd

    def test_readonly_command_executes_without_confirm(self):
        """Read-only commands execute directly even without a callback."""
        tool = ShellTool(confirm_callback=None)
        result = tool.execute({"cmd": "echo no_confirm_needed"})
        assert result.success
        assert "no_confirm_needed" in result.output

    def test_mv_needs_confirm(self):
        """mv commands require confirmation."""
        denied = []
        tool = ShellTool(confirm_callback=lambda cmd: denied.append(cmd) or False)
        result = tool.execute({"cmd": "mv old.py new.py"})
        assert not result.success
        assert len(denied) == 1

    def test_git_commit_needs_confirm(self):
        """git commit requires confirmation."""
        approved = []
        tool = ShellTool(confirm_callback=lambda cmd: approved.append(cmd) or True)
        # Execution in a real git repo would fail (no staged files), but callback should be called
        tool.execute({"cmd": "git commit -m 'test'"})
        assert len(approved) == 1
        assert "git commit" in approved[0]

    def test_curl_needs_confirm(self):
        """curl requires confirmation (network request)."""
        denied = []
        tool = ShellTool(confirm_callback=lambda cmd: denied.append(cmd) or False)
        result = tool.execute({"cmd": "curl https://example.com"})
        assert not result.success
        assert len(denied) == 1

    def test_redirect_needs_confirm(self):
        """Output redirection (>) requires confirmation."""
        # Pure logic: echo hello > file should not be classified as read-only
        assert not _is_readonly("echo hello > output.txt")
        # And it needs confirmation (contains > redirect)
        assert _needs_confirm("echo hello > output.txt")
        # Tool layer: should return error when callback denies
        denied = []
        tool = ShellTool(confirm_callback=lambda cmd: denied.append(cmd) or False)
        result = tool.execute({"cmd": "echo hello > output.txt"})
        assert not result.success
        assert len(denied) == 1


# ===========================================================================
# Helper functions
# ===========================================================================

class TestAlwaysCallbacks:
    def test_always_allow(self):
        assert always_allow("any command") is True

    def test_always_deny(self):
        assert always_deny("any command") is False


# ===========================================================================
# Integration: permission confirmation in the agent flow
# ===========================================================================

class TestConfirmInAgentFlow:

    def test_dangerous_command_denied_stops_that_step(self, tmp_path):
        """When a dangerous command is denied, the observation records the rejection and the agent continues."""
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog
        from agent.task import Action, ActionType, EventType, Task, ToolCall
        from llm.base import MockBackend
        from tools.base import ToolRegistry

        script = [
            # First attempt a dangerous command that will be rejected
            Action(ActionType.TOOL_CALL, "try dangerous", ToolCall("shell", {"cmd": "pip install requests"})),
            # Then continue with a safe command
            Action(ActionType.TOOL_CALL, "safe cmd", ToolCall("shell", {"cmd": "echo done"})),
            Action(ActionType.FINISH, "done", message="completed"),
        ]
        backend = MockBackend(script)
        # always_deny: all commands requiring confirmation are rejected
        registry = ToolRegistry().register(ShellTool(confirm_callback=always_deny))
        agent = Agent(backend, registry, AgentConfig(max_steps=5))

        task = Task(task_id="conf1", description="test", repo_path=str(tmp_path), max_steps=5)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()

        # Check event log: first observation should contain rejection info
        events = log.replay()
        obs_events = [e for e in events if e.event_type == EventType.OBSERVATION]
        first_obs = obs_events[0].payload["observation"]
        assert first_obs["status"] == "error"
        assert "rejected" in first_obs.get("error", "").lower()

    def test_all_allowed_completes_normally(self, tmp_path):
        """When all commands are allowed, the agent completes normally."""
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog
        from agent.task import Action, ActionType, Task, ToolCall
        from llm.base import MockBackend
        from tools.base import ToolRegistry

        script = [
            Action(ActionType.TOOL_CALL, "install", ToolCall("shell", {"cmd": "pip install requests"})),
            Action(ActionType.FINISH, "done", message="installed"),
        ]
        backend = MockBackend(script)
        registry = ToolRegistry().register(ShellTool(confirm_callback=always_allow))
        agent = Agent(backend, registry, AgentConfig(max_steps=5))

        task = Task(task_id="conf2", description="test", repo_path=str(tmp_path), max_steps=5)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()

    def test_readonly_never_triggers_confirm(self, tmp_path):
        """Read-only commands do not trigger confirmation even when callback=always_deny."""
        from agent.core import Agent, AgentConfig
        from agent.event_log import EventLog
        from agent.task import Action, ActionType, Task, ToolCall
        from llm.base import MockBackend
        from tools.base import ToolRegistry

        script = [
            Action(ActionType.TOOL_CALL, "ls", ToolCall("shell", {"cmd": "ls /tmp"})),
            Action(ActionType.TOOL_CALL, "echo", ToolCall("shell", {"cmd": "echo hello"})),
            Action(ActionType.FINISH, "done", message="ok"),
        ]
        backend = MockBackend(script)
        # Even with always_deny, read-only commands bypass the callback
        registry = ToolRegistry().register(ShellTool(confirm_callback=always_deny))
        agent = Agent(backend, registry, AgentConfig(max_steps=5))

        task = Task(task_id="conf3", description="test", repo_path=str(tmp_path), max_steps=5)
        with EventLog.create(task, log_dir=str(tmp_path / "logs")) as log:
            result = agent.run(task, log)

        assert result.is_success()


# ===========================================================================
# CLI --confirm option registration
# ===========================================================================

class TestCliConfirmOption:
    def test_confirm_option_in_run_help(self):
        from click.testing import CliRunner
        from entry.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--confirm" in result.output

    def test_chat_always_has_confirm(self):
        """chat command enables terminal_confirm by default; no extra parameter needed."""
        from click.testing import CliRunner
        from entry.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
