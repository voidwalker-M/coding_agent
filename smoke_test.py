"""
smoke_test.py

End-to-end smoke test, run from the project root:
    python smoke_test.py

What it tests:
1. Uses a real LLM (reads configuration from config/default.yaml)
2. Gives the agent a simple task: create a hello.py file under /tmp
3. Prints each step's action and observation
4. Prints the final RunResult

No SWE-bench dependency, no real repo required — great for quickly verifying API connectivity and tool execution.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import yaml

# Add the project root to sys.path (already there when script is in the root; this is just a safety measure)
sys.path.insert(0, str(Path(__file__).parent))

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog, summarize_run
from agent.task import EventType, Task
from llm.router import create_backend_from_config
from tools.base import ToolRegistry
from tools.file_tool import FileReadTool, FileViewTool, FileWriteTool
from tools.git_tool import GitAddTool, GitCommitTool, GitDiffTool, GitStatusTool
from tools.search_tool import FindFilesTool, FindSymbolTool, SearchTextTool
from tools.shell_tool import ShellTool
from tools.test_tool import PytestTool

# ---------------------------------------------------------------------------
# Logging: print every agent step
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# Colored output helpers (colors when terminal supports it, degrades otherwise)
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

def load_config(path: str = "config/default.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        raw = f.read()

    # Expand ${VAR} environment variable placeholders
    import os, re
    def replace_env(m):
        var = m.group(1)
        val = os.environ.get(var, "")
        if not val:
            logger.warning("Environment variable %s is not set", var)
        return val
    raw = re.sub(r"\$\{(\w+)\}", replace_env, raw)

    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Assemble tool registry
# ---------------------------------------------------------------------------

def build_registry() -> ToolRegistry:
    return (
        ToolRegistry()
        .register(ShellTool())
        .register(FileReadTool())
        .register(FileViewTool())
        .register(FileWriteTool())
        .register(SearchTextTool())
        .register(FindFilesTool())
        .register(FindSymbolTool())
        .register(PytestTool())
        .register(GitStatusTool())
        .register(GitDiffTool())
        .register(GitAddTool())
        .register(GitCommitTool())
    )


# ---------------------------------------------------------------------------
# Print event log in real time
# ---------------------------------------------------------------------------

def print_events_realtime(log: EventLog, last_count: int) -> int:
    """
    Read the event log and print new events since last_count.
    Returns the current total event count (for incremental printing on next call).
    """
    events = log.replay()
    for event in events[last_count:]:
        _print_event(event)
    return len(events)


def _print_event(event) -> None:
    etype = event.event_type
    payload = event.payload

    if etype == EventType.TASK_START:
        task = payload["task"]
        print(bold(f"\n{'='*60}"))
        print(bold(f"  TASK: {task['description']}"))
        print(bold(f"  REPO: {task['repo_path']}"))
        print(bold(f"{'='*60}\n"))

    elif etype == EventType.ACTION:
        step = payload["step"]
        action = payload["action"]
        thought = action.get("thought", "")
        atype = action.get("action_type", "")
        tc = action.get("tool_call")

        print(cyan(f"[Step {step}] Action: {atype}"))
        if thought:
            # Truncate to 200 chars; thought content can be very long
            short_thought = thought[:200] + ("..." if len(thought) > 200 else "")
            print(f"  Thought: {short_thought}")
        if tc:
            print(f"  Tool:    {tc['name']}")
            params_str = str(tc['params'])
            if len(params_str) > 120:
                params_str = params_str[:120] + "..."
            print(f"  Params:  {params_str}")

    elif etype == EventType.OBSERVATION:
        obs = payload["observation"]
        status = obs.get("status", "")
        tool = obs.get("tool_name", "")
        output = obs.get("output", "")
        error = obs.get("error")

        if status == "success":
            print(green(f"  ✓ [{tool}] {status}"))
        else:
            print(red(f"  ✗ [{tool}] {status}"))
            if error:
                print(red(f"    Error: {error}"))

        # Print first 10 lines of output
        lines = output.splitlines()[:10]
        for line in lines:
            print(f"    {line}")
        if len(output.splitlines()) > 10:
            print(f"    ... ({len(output.splitlines()) - 10} more lines)")
        print()

    elif etype == EventType.REFLECTION:
        reason = payload.get("reason", "")
        print(yellow(f"\n  ⟳ REFLECTION triggered: {reason}\n"))

    elif etype == EventType.TASK_COMPLETE:
        summary = payload.get("summary", "")
        print(green(bold(f"\n{'='*60}")))
        print(green(bold(f"  ✓ TASK COMPLETE")))
        print(green(f"  {summary}"))
        print(green(bold(f"{'='*60}\n")))

    elif etype == EventType.TASK_FAILED:
        reason = payload.get("reason", "")
        print(red(bold(f"\n{'='*60}")))
        print(red(bold(f"  ✗ TASK FAILED")))
        print(red(f"  {reason}"))
        print(red(bold(f"{'='*60}\n")))


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

TASK_DESCRIPTION = """\
Create a file at /tmp/hello_agent.py with the following content:

    print("Hello from coding agent!")

Then run it with `python /tmp/hello_agent.py` and confirm the output is correct.
"""


def main():
    print(bold("\n🤖 Coding Agent — Smoke Test\n"))

    # 1. Load configuration
    config = load_config()
    llm_config = config.get("llm", {})
    agent_config_dict = config.get("agent", {})

    provider = llm_config.get("provider", "?")
    model = llm_config.get("model", "?")
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Log dir  : {agent_config_dict.get('log_dir', './logs')}\n")

    # 2. Create backend
    try:
        backend = create_backend_from_config(llm_config)
        print(green("✓ LLM backend created\n"))
    except Exception as e:
        print(red(f"✗ Failed to create LLM backend: {e}"))
        sys.exit(1)

    # 3. Assemble registry
    registry = build_registry()
    print(f"Tools    : {', '.join(registry.tool_names)}\n")

    # 4. Create task
    task = Task(
        description=TASK_DESCRIPTION,
        repo_path=str(Path.cwd()),
        max_steps=agent_config_dict.get("max_steps", 15),
        budget_tokens=agent_config_dict.get("budget_tokens", 80000),
    )

    # 5. Create agent
    agent_config = AgentConfig(
        max_steps=task.max_steps,
    )
    agent = Agent(backend, registry, agent_config)

    # 6. Run
    log_dir = agent_config_dict.get("log_dir", "./logs")
    print(bold("Starting agent...\n"))
    t0 = time.time()

    event_count = 0
    with EventLog.create(task, log_dir=log_dir) as log:
        print(f"Log file : {log.path}\n")

        # agent.run() is synchronous; we print all events after it completes.
        # For true real-time printing, streaming was added in a later version.
        result = agent.run(task, log)
        print_events_realtime(log, 0)

    elapsed = time.time() - t0

    # 7. Print statistics
    print(bold("─" * 60))
    print(f"Status   : {green(result.status.value) if result.is_success() else red(result.status.value)}")
    print(f"Steps    : {result.steps_taken}")
    print(f"Tokens   : {result.total_tokens:,}")
    print(f"Time     : {elapsed:.1f}s")
    if result.error:
        print(f"Error    : {red(result.error)}")
    print(bold("─" * 60))

    # 8. Event summary
    with EventLog.open_existing(log.path) as log2:
        stats = summarize_run(log2)
    print(f"\nEvent summary:")
    print(f"  Actions     : {stats['actions']}")
    print(f"  Reflections : {stats['reflections']}")
    print(f"  Tool calls  : {stats['tool_calls']}")
    print(f"  Final status: {stats['final_status']}")

    return 0 if result.is_success() else 1


if __name__ == "__main__":
    sys.exit(main())
