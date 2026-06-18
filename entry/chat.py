"""
entry/chat.py

Interactive conversation mode. Persistent session: the agent continues working
after each user input; history is retained across rounds, enabling continuous
dialogue similar to Claude Code.

Architecture:
- ChatSession holds backend / registry / history, reused across rounds
- Each round creates a new Task, but history is continued via agent._inject_history()
- EventLog is per-round (for per-round auditability); statistics are shown cumulatively
- Real-time printing: each event is echoed immediately after being written to the log

Usage:
    agent chat --repo /path/to/repo
    agent chat --repo . --model deepseek-chat
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Callable

import click

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Colored output (same style as cli.py)
# ---------------------------------------------------------------------------

def _c(t: str, code: str) -> str:
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t

def green(t: str) -> str:  return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str) -> str:    return _c(t, "31")
def cyan(t: str) -> str:   return _c(t, "36")
def bold(t: str) -> str:   return _c(t, "1")
def dim(t: str) -> str:    return _c(t, "2")
def magenta(t: str) -> str: return _c(t, "35")


# ---------------------------------------------------------------------------
# Real-time event printing (more compact than cli.py, suited for continuous conversation)
# ---------------------------------------------------------------------------

def _print_event_live(event) -> None:
    """Called immediately after each event is written to the log for real-time display."""
    from agent.task import EventType
    etype = event.event_type
    p = event.payload

    if etype == EventType.ACTION:
        step = p["step"]
        action = p["action"]
        thought = (action.get("thought") or "").strip()
        atype = action.get("action_type", "")
        tc = action.get("tool_call")

        # In streaming mode, thought has already been printed by stream_callback in real time.
        # In non-streaming mode or when thought is empty, print it here.
        # finish/give_up thought is the answer content; already shown in TASK_COMPLETE — don't repeat.
        if thought and thought != "(no thought)" and atype not in ("finish", "give_up"):
            import sys
            sys.stdout.write("\n")   # ensure tool call starts on a new line
            sys.stdout.flush()

        if tc:
            _print_event_live._last_tool_name = tc['name']  # stored for the observation handler
            click.echo(cyan(f"  [{step}] {tc['name']}"), nl=False)
            # Print the most informative parameter
            params = tc.get("params", {})
            key_param = (
                params.get("cmd")
                or params.get("path")
                or params.get("pattern")
                or params.get("symbol")
                or params.get("message")
                or ""
            )
            if key_param:
                short_param = str(key_param)[:60]
                suffix = "..." if len(str(key_param)) > 60 else ""
                click.echo(cyan(f"  {short_param}{suffix}"))
            else:
                click.echo()
        elif atype == "finish":
            click.echo(green(f"\n  [{step}] ✓ finish"))
            # Store the message globally for TASK_COMPLETE event to print
            _finish_message = action.get("message", "") or ""
            _print_event_live._pending_message = _finish_message
        elif atype == "give_up":
            click.echo(red(f"\n  [{step}] ✗ give_up"))

    elif etype == EventType.OBSERVATION:
        obs = p["observation"]
        status = obs.get("status", "")
        output = (obs.get("output") or "").strip()
        error = obs.get("error")

        # Get tool name from the previous action event (_last_tool_name set in ACTION branch)
        tool_name = getattr(_print_event_live, "_last_tool_name", "")

        # Read-only tools: only show ✓ or ✗, no content (model already read it; user doesn't need to see it)
        SILENT_TOOLS = {"file_read", "file_view", "file_write", "find_files", "find_symbol"}
        silent = tool_name in SILENT_TOOLS

        if status == "success":
            if silent:
                click.echo(green("  ✓"))
            else:
                lines = output.splitlines()
                MAX_PREVIEW = 20
                preview = "\n".join(f"    {l}" for l in lines[:MAX_PREVIEW])
                if lines:
                    click.echo(green("  ✓") + dim(f"\n{preview}"))
                    if len(lines) > MAX_PREVIEW:
                        click.echo(dim(f"    ... ({len(lines)-MAX_PREVIEW} more lines)"))
                else:
                    click.echo(green("  ✓"))
        else:
            click.echo(red(f"  ✗ {error or output[:120]}"))

    elif etype == EventType.REFLECTION:
        reason = p.get("reason", "")
        click.echo(yellow(f"\n  ⟳ Reflection ({reason}) — reconsidering approach...\n"))

    elif etype == EventType.TASK_COMPLETE:
        # Retrieve the message stored by the finish action
        message = getattr(_print_event_live, "_pending_message", "")
        _print_event_live._pending_message = ""

        if message:
            # Get the thought content that was streamed (stored in stream_callback)
            streamed = getattr(_print_event_live, "_streamed_thought", "").strip()
            msg_stripped = message.strip()

            if msg_stripped and msg_stripped != streamed:
                # thought and message differ (e.g. Claude) → print the final answer separately
                import sys
                sys.stdout.write("\n")
                sys.stdout.flush()
                click.echo(msg_stripped)
            # thought == message (e.g. DeepSeek flash) → already streamed, don't repeat

    elif etype == EventType.TASK_FAILED:
        reason = p.get("reason", "")
        click.echo(red(bold(f"\n  ❌ Failed: {reason}")))


# ---------------------------------------------------------------------------
# ChatSession — persistent session state across rounds
# ---------------------------------------------------------------------------

class ChatSession:
    """
    Persistent session. Retains across multiple conversation rounds:
    - backend / registry (unchanged)
    - ConversationHistory (core: lets the agent remember what it did previously)
    - Cumulative token / step statistics
    - repo_map cache (auto-invalidated when the repo changes)
    """

    def __init__(self, backend, registry, config, repo_path: str, log_dir: str, confirm_callback=None) -> None:
        from agent.core import Agent, AgentConfig
        from context.history import ConversationHistory

        self.repo_path = repo_path
        self.log_dir = log_dir
        self.config = config
        self._confirm_callback = confirm_callback

        # Streaming callbacks: flush each token to the terminal immediately
        _stream_started = [False]
        _thought_printed = [False]  # tracks whether a thought has been printed (for newline before message)
        _streamed_buf = []   # records streamed content for comparison with message

        def _thought_cb(text: str) -> None:
            """Reasoning process: dim color to indicate the model is thinking."""
            import sys
            if not _stream_started[0]:
                sys.stdout.write("\r  ")
                sys.stdout.flush()
                _stream_started[0] = True
            sys.stdout.write(dim(text))
            sys.stdout.flush()
            _thought_printed[0] = True

        def _stream_cb(text: str) -> None:
            """Final answer: normal bright color."""
            import sys
            if not _stream_started[0]:
                # First message token; nothing has been printed yet
                sys.stdout.write("\r  ")
                sys.stdout.flush()
                _stream_started[0] = True
            elif _thought_printed[0]:
                # A thought was printed previously; add two newlines as separator before the message
                sys.stdout.write("\n\n")
                sys.stdout.flush()
                _thought_printed[0] = False  # only insert separator once
            sys.stdout.write(text)
            sys.stdout.flush()
            _streamed_buf.append(text)
            _print_event_live._streamed_thought = "".join(_streamed_buf)

        def _reset_stream_state() -> None:
            """Reset streaming state at the start of each round to prevent cross-round accumulation causing duplicate message printing."""
            _stream_started[0] = False
            _thought_printed[0] = False
            _streamed_buf.clear()
            _print_event_live._streamed_thought = ""
            _print_event_live._pending_message = ""

        self._reset_stream_state = _reset_stream_state

        agent_cfg = AgentConfig(
            max_steps=config.agent.max_steps,
            budget_tokens=config.agent.budget_tokens,
            history_max_messages=config.context.history_window * 2,
            llm_max_retries=3,
            llm_retry_delay=1.0,
            stream=True,
            stream_callback=_stream_cb,
            thought_callback=_thought_cb,
            confirm_dangerous=confirm_callback is not None,
            confirm_callback=confirm_callback,
        )
        self.agent = Agent(backend, registry, agent_cfg)
        self._shared_history = ConversationHistory(
            max_messages=config.context.history_window * 2
        )

        # Cumulative statistics
        self.total_tokens = 0
        self.total_steps = 0
        self.round_count = 0

    def run_round(self, user_input: str) -> bool:
        """
        Execute one conversation round.

        Args:
            user_input: the user's input for this round

        Returns:
            True if the round succeeded or ended normally; False if it failed
        """
        from agent.core import AgentConfig
        from agent.event_log import EventLog
        from agent.task import Task
        from llm.base import LLMMessage

        self.round_count += 1

        # Reset streaming state: leftover streamed buffer / flags from the previous round
        # would cause the finish message to be compared against accumulated cross-round content
        # and printed again.
        self._reset_stream_state()

        # Append the user's input to the shared history
        self._shared_history.add(LLMMessage(role="user", content=user_input))

        # Build the Task for this round (repo_path is fixed; description is the user input)
        task = Task(
            description=user_input,
            repo_path=self.repo_path,
            max_steps=self.config.agent.max_steps,
            budget_tokens=self.config.agent.budget_tokens,
        )

        # Inject the shared history into the agent (replaces its internal history)
        # Cross-round continuation is achieved via monkey-patching _shared_history
        self.agent._shared_history = self._shared_history

        t0 = time.time()
        with EventLog.create(task, log_dir=self.log_dir) as log:
            # Real-time printing: each event is echoed immediately after being written
            result = self._run_with_live_print(task, log)

        elapsed = time.time() - t0
        self.total_tokens += result.total_tokens
        self.total_steps += result.steps_taken

        # Append the agent's reply from this round to the shared history
        # so the agent can see what it said in the previous round
        if result.summary:
            self._shared_history.add(LLMMessage(
                role="assistant",
                content=f"[Round {self.round_count} complete]\n{result.summary}",
            ))

        # Print a newline after streaming output; reset readline line state
        import sys as _sys
        _sys.stdout.write("\n")
        _sys.stdout.flush()

        # Print per-round statistics
        click.echo(dim(
            f"  ─── Round {self.round_count} · "
            f"{result.steps_taken} steps · "
            f"{result.total_tokens:,} tokens · "
            f"{elapsed:.1f}s ───"
        ))

        return result.is_success() or result.status.value == "gave_up"

    def _run_with_live_print(self, task, log):
        """
        Run the agent while printing events in real time.

        Since agent.run() is synchronous (returns only when done), real-time
        "print as you write" is achieved by monkey-patching EventLog._append.
        """
        original_append = log._append

        def live_append(event):
            original_append(event)
            _print_event_live(event)

        log._append = live_append

        # Pass the shared history to the agent.
        # agent.run() rebuilds history internally; we inject ours after initialization.
        # Implemented by overriding _build_messages history parameter.
        return self._run_injecting_history(task, log)

    def _run_injecting_history(self, task, log):
        """
        Run the agent and inject the shared history before the first LLM call.

        Core trick: patch the agent's run() method so that after it initializes
        ConversationHistory but before the first LLM call, we swap in our shared history.
        Simpler approach: set a _initial_history attribute on the agent that run() checks.
        """
        from context.history import ConversationHistory
        from agent.prompt import build_task_prompt
        from llm.base import LLMMessage

        agent = self.agent

        # Approach: set _pending_history on the agent; core.py's run() reads it.
        # (See the patch in core.py)
        agent._pending_history = self._shared_history

        result = agent.run(task, log)

        # Clean up to avoid affecting the next round
        if hasattr(agent, "_pending_history"):
            del agent._pending_history

        return result

    def print_stats(self) -> None:
        """Print cumulative session statistics."""
        click.echo(bold(f"\n{'─'*50}"))
        click.echo(f"  Session stats:")
        click.echo(f"    Rounds  : {self.round_count}")
        click.echo(f"    Steps   : {self.total_steps}")
        click.echo(f"    Tokens  : {self.total_tokens:,}")
        click.echo(bold(f"{'─'*50}\n"))
