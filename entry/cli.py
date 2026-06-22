"""
entry/cli.py

Command-line entry point.

Usage:
    # Pass task description directly
    python -m entry.cli run --repo /path/to/repo --task "Fix the failing test"

    # Read task description from file
    python -m entry.cli run --repo . --task-file task.txt

    # Override the model
    python -m entry.cli run --repo . --task "fix it" --model deepseek-chat

    # View event log statistics
    python -m entry.cli log show logs/abc123_20240101_120000.jsonl

After installing as a CLI tool (scripts configured in pyproject.toml):
    agent run --repo . --task "fix it"
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

# Add the project root to sys.path (needed when running the script directly)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.schema import load_config, merge_cli_overrides   # noqa: E402
from llm.router import create_backend_from_config            # noqa: E402

# Module-level imports (for patching in tests)
from config.schema import load_config, merge_cli_overrides  # noqa: E402
from llm.router import create_backend_from_config           # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: colored output
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def green(t: str) -> str:  return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def red(t: str) -> str:    return _c(t, "31")
def cyan(t: str) -> str:   return _c(t, "36")
def bold(t: str) -> str:   return _c(t, "1")
def dim(t: str) -> str:    return _c(t, "2")
def magenta(t: str) -> str: return _c(t, "35")


# ---------------------------------------------------------------------------
# Build agent components
# ---------------------------------------------------------------------------

def _build_registry(cfg, confirm_callback=None, runtime=None):
    """Assemble the tool registry from configuration."""
    from tools.base import ToolRegistry
    from tools.file_tool import FileReadTool, FileViewTool, FileWriteTool
    from tools.git_tool import GitAddTool, GitCommitTool, GitDiffTool, GitStatusTool
    from tools.search_tool import FindFilesTool, FindSymbolTool, SearchTextTool
    from tools.shell_tool import ShellTool
    from tools.test_tool import PytestTool

    return (
        ToolRegistry()
        .register(ShellTool(confirm_callback=confirm_callback, runtime=runtime))
        .register(FileReadTool())
        .register(FileViewTool())
        .register(FileWriteTool())
        .register(SearchTextTool())
        .register(FindFilesTool())
        .register(FindSymbolTool())
        .register(PytestTool(runtime=runtime))
        .register(GitStatusTool(runtime=runtime))
        .register(GitDiffTool(runtime=runtime))
        .register(GitAddTool(runtime=runtime))
        .register(GitCommitTool(runtime=runtime))
    )


def _build_retriever(repo_path: str, kind: str, rerank: str = "none", cache: bool = True,
                     extra_paths=None):
    """
    Build a RAG retriever based on --retriever / --rerank options. Returns None when kind='none'.

    Persistent caching (<repo>/.rag_cache) and incremental updates are enabled by default:
    unchanged files reuse cached vectors; only changed files are re-embedded.
    Hybrid retrieval (dense + BM25) is enabled by default.

    extra_paths: external dirs/files (docs, dependency source) indexed alongside the repo (#2).
    """
    if not kind or kind == "none":
        return None
    from pathlib import Path as _Path
    from context.rag import RagRetriever
    cache_dir = str(_Path(repo_path) / ".rag_cache") if cache else None
    retriever = RagRetriever(
        repo_path,
        hybrid=True,
        reranker=(rerank if rerank and rerank != "none" else None),
        cache_dir=cache_dir,
        extra_paths=list(extra_paths) if extra_paths else None,
    ).build()
    return retriever


def _print_step(event) -> None:
    """Print a single event in real time."""
    from agent.task import EventType
    etype = event.event_type
    payload = event.payload

    if etype == EventType.TASK_START:
        task = payload["task"]
        click.echo(bold(f"\n{'─'*60}"))
        click.echo(bold(f"  Task : {task['description'][:80]}"))
        click.echo(bold(f"  Repo : {task['repo_path']}"))
        click.echo(bold(f"{'─'*60}\n"))

    elif etype == EventType.ACTION:
        step = payload["step"]
        action = payload["action"]
        thought = action.get("thought", "")[:160]
        atype = action.get("action_type", "")
        tc = action.get("tool_call")
        click.echo(cyan(f"[Step {step}] {atype}"))
        if thought:
            click.echo(dim(f"  ↳ {thought}"))
        if tc:
            params_str = str(tc["params"])[:100]
            click.echo(f"  Tool: {tc['name']}  params: {params_str}")

    elif etype == EventType.OBSERVATION:
        obs = payload["observation"]
        status = obs.get("status", "")
        tool = obs.get("tool_name", "")
        output = obs.get("output", "")
        if status == "success":
            click.echo(green(f"  ✓ [{tool}]"))
        else:
            click.echo(red(f"  ✗ [{tool}] {obs.get('error', '')}"))
        # Print first 5 lines of output
        for line in output.splitlines()[:5]:
            click.echo(dim(f"    {line}"))
        if len(output.splitlines()) > 5:
            click.echo(dim(f"    ... ({len(output.splitlines())-5} more lines)"))
        click.echo()

    elif etype == EventType.REFLECTION:
        click.echo(yellow(f"\n  ⟳ Reflection: {payload.get('reason', '')}\n"))

    elif etype == EventType.TASK_COMPLETE:
        click.echo(green(bold(f"\n✓ COMPLETE: {payload.get('summary', '')}\n")))

    elif etype == EventType.TASK_FAILED:
        click.echo(red(bold(f"\n✗ FAILED: {payload.get('reason', '')}\n")))


# ---------------------------------------------------------------------------
# CLI main command group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "-c",
    default=None,
    help="Path to config YAML file (default: config/default.yaml)",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Coding Agent — autonomous code editing and bug fixing."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--repo", "-r", default=".", show_default=True, help="Path to the target repository (default: current directory)")
@click.option("--task", "-t", default=None, help="Task description (natural language)")
@click.option("--task-file", "-f", default=None, help="Read task description from file")
@click.option("--model", "-m", default=None, help="Override LLM model name")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--max-steps", default=None, type=int, help="Override max steps")
@click.option("--stream", "-s", is_flag=True, default=True, help="Enable streaming output (default: on)")
@click.option("--confirm", is_flag=True, default=False, help="Ask confirmation before running dangerous shell commands")
@click.option("--sandbox", is_flag=True, default=False, help="Run commands in Docker sandbox (requires Docker)")
@click.option("--retriever", "-R", type=click.Choice(["none", "rag"]), default="none", show_default=True, help="Context retriever: 'rag' enables hybrid (dense+BM25) code retrieval")
@click.option("--rerank", type=click.Choice(["none", "mmr", "cross-encoder"]), default="none", show_default=True, help="Rerank retrieved chunks: 'mmr' (numpy, diversity) or 'cross-encoder' (needs sentence-transformers)")
@click.option("--rag-extra", multiple=True, type=click.Path(), help="External dir/file to index with RAG (docs, dependency source). Repeatable. Implies --retriever rag")
@click.option("--engine", "-e", type=click.Choice(["native", "langgraph"]), default="native", show_default=True, help="Orchestration engine: 'langgraph' runs the LangGraph port")
@click.option("--cache", is_flag=True, default=False, help="Cache LLM responses (saves tokens on repeated/identical calls)")
@click.option("--cheap-model", default=None, help="Enable cost-aware routing: send easy steps to this cheaper model")
@click.option("--max-usd", default=None, type=float, help="Hard spend ceiling in USD for this run (stops when exceeded)")
@click.option("--rpm", default=None, type=int, help="Throttle LLM calls to at most this many requests per minute")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def run(
    ctx: click.Context,
    repo: str,
    task: str | None,
    task_file: str | None,
    model: str | None,
    provider: str | None,
    max_steps: int | None,
    stream: bool,
    confirm: bool,
    sandbox: bool,
    retriever: str,
    rerank: str,
    rag_extra: tuple,
    engine: str,
    cache: bool,
    cheap_model: str | None,
    max_usd: float | None,
    rpm: int | None,
    verbose: bool,
) -> None:
    """Run the coding agent on a repository."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configuration
    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(
        config, provider=provider, model=model, max_steps=max_steps
    )

    # Parse task description
    if task_file:
        description = Path(task_file).read_text(encoding="utf-8").strip()
    elif task:
        description = task
    else:
        click.echo(red("Error: provide --task or --task-file"), err=True)
        sys.exit(1)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    # Print run info
    click.echo(bold(f"\n🤖 Coding Agent"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Repo     : {repo_path}")
    click.echo(f"  Max steps: {config.agent.max_steps}\n")

    # Build components
    try:
        backend = create_backend_from_config({
            "provider": config.llm.provider,
            "model":    config.llm.model,
            "api_key":  config.llm.api_key or None,
            "base_url": config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(red(f"Error: {e}"), err=True)
        sys.exit(1)

    # Token-efficiency layers (#4): cost-aware routing, response cache, rate/$ limit.
    if cache or cheap_model or max_usd is not None or rpm is not None:
        from llm.compose import compose_backend
        cheap_backend = None
        if cheap_model:
            cheap_backend = create_backend_from_config({
                "provider": config.llm.provider, "model": cheap_model,
                "api_key": config.llm.api_key or None, "base_url": config.llm.base_url or None,
                "max_tokens": config.llm.max_tokens,
            })
        backend = compose_backend(
            backend, cheap=cheap_backend, cache=cache,
            rpm=rpm, max_usd=max_usd, model_for_cost=config.llm.model,
        )
        extras = [n for n, on in (("cache", cache), ("router", bool(cheap_model)),
                                  ("max_usd", max_usd is not None), ("rpm", rpm is not None)) if on]
        click.echo(dim(f"  Token-saving: {', '.join(extras)}"))

    from tools.shell_tool import terminal_confirm
    from tools.runtime import create_runtime
    confirm_cb = terminal_confirm if confirm else None
    runtime = create_runtime(sandbox=sandbox, repo_path=str(repo_path)) if sandbox else None
    if sandbox:
        click.echo(dim(f"  Sandbox: Docker ({runtime.name})"))
    # --rag-extra implies enabling RAG even if --retriever wasn't passed.
    if rag_extra and retriever == "none":
        retriever = "rag"
    rag_retriever = _build_retriever(str(repo_path), retriever, rerank=rerank, extra_paths=rag_extra)
    if rag_retriever is not None:
        click.echo(dim(f"  Retriever: RAG ({rag_retriever.chunk_count} chunks, {rag_retriever.backend_info})"))
        if rag_extra:
            click.echo(dim(f"  RAG external: {', '.join(rag_extra)}"))
    registry = _build_registry(config, confirm_callback=confirm_cb, runtime=runtime)

    from agent.core import Agent, AgentConfig
    from agent.event_log import EventLog, summarize_run
    from agent.task import Task
    try:
        from context.token_budget import is_tiktoken_available
    except ImportError:
        is_tiktoken_available = lambda: False

    # Streaming callback: final answer in normal bright color
    def _stream_cb(text: str) -> None:
        import sys
        sys.stdout.write(text)
        sys.stdout.flush()

    # Reasoning callback: thinking process in dim color
    def _thought_cb(text: str) -> None:
        import sys
        sys.stdout.write(dim(text))
        sys.stdout.flush()

    agent_config = AgentConfig(
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
        history_max_messages=config.context.history_window * 2,
        stream=stream,
        stream_callback=_stream_cb if stream else None,
        thought_callback=_thought_cb if stream else None,
        confirm_dangerous=confirm,
        confirm_callback=confirm_cb,
        retriever=rag_retriever,
    )
    if engine == "langgraph":
        from agent.langgraph_loop import LangGraphAgent
        click.echo(dim("  Engine: LangGraph (streaming disabled)"))
        agent = LangGraphAgent(backend, registry, agent_config)
    else:
        agent = Agent(backend, registry, agent_config)

    task_obj = Task(
        description=description,
        repo_path=str(repo_path),
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
    )

    if verbose:
        click.echo(dim(
            f"  tiktoken: {'yes' if is_tiktoken_available() else 'no (char estimate)'}\n"
        ))

    # Run
    t0 = time.time()
    with EventLog.create(task_obj, log_dir=config.agent.log_dir) as log:
        click.echo(dim(f"  Log: {log.path}\n"))
        result = agent.run(task_obj, log)
        # Print all events
        for event in log.replay():
            _print_step(event)

    elapsed = time.time() - t0

    # Print results
    click.echo(bold("─" * 60))
    status_str = green("SUCCESS") if result.is_success() else red(result.status.value.upper())
    click.echo(f"Status  : {status_str}")
    click.echo(f"Steps   : {result.steps_taken}")
    click.echo(f"Tokens  : {result.total_tokens:,}")
    click.echo(f"Time    : {elapsed:.1f}s")
    if result.error:
        click.echo(red(f"Error   : {result.error}"))
    click.echo(bold("─" * 60) + "\n")

    sys.exit(0 if result.is_success() else 1)



# ---------------------------------------------------------------------------
# chat subcommand — interactive conversation mode
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--repo", "-r", default=".", show_default=True, help="Path to the target repository (default: current directory)")
@click.option("--model", "-m", default=None, help="Override LLM model name")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--max-steps", default=None, type=int, help="Max steps per round")
@click.option("--sandbox", is_flag=True, default=False, help="Run commands in Docker sandbox (requires Docker)")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def chat(
    ctx: click.Context,
    repo: str,
    model: str | None,
    provider: str | None,
    max_steps: int | None,
    sandbox: bool,
    verbose: bool,
) -> None:
    """Interactive chat mode — continuous conversation with the agent."""
    import logging
    from entry.chat import ChatSession

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(config, provider=provider, model=model, max_steps=max_steps)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    try:
        backend = create_backend_from_config({
            "provider":   config.llm.provider,
            "model":      config.llm.model,
            "api_key":    config.llm.api_key or None,
            "base_url":   config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(red(f"Error: {e}"), err=True)
        sys.exit(1)

    registry = _build_registry(config)
    from tools.shell_tool import terminal_confirm
    from tools.runtime import create_runtime
    runtime = create_runtime(sandbox=sandbox, repo_path=str(repo_path)) if sandbox else None
    if sandbox:
        click.echo(dim(f"  Sandbox: Docker ({runtime.name})"))
    session = ChatSession(
        backend=backend,
        registry=registry,
        config=config,
        repo_path=str(repo_path),
        log_dir=config.agent.log_dir,
        confirm_callback=terminal_confirm,   # confirmation enabled by default in chat mode
    )

    # Welcome message
    click.echo(bold(f"\n🤖 Coding Agent — Chat Mode"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Repo     : {repo_path}")
    click.echo(dim(f"  Type your task. Commands: /exit /stats /clear /help\n"))

    # Enable line editing: backspace, arrow keys, Ctrl+A/E, history (↑↓)
    try:
        import readline as _rl
        import sys as _sys
        # Detect backend: libedit (some Linux/macOS) vs GNU readline
        _is_libedit = "libedit" in getattr(_rl, "__doc__", "") or (
            hasattr(_rl, "parse_and_bind") and _sys.platform == "darwin"
        )
        # More reliable detection: try a libedit-specific binding syntax
        try:
            _rl.parse_and_bind("bind -e")   # enable Emacs mode in libedit
            _is_libedit = True
        except Exception:
            _is_libedit = False

        if _is_libedit:
            _rl.parse_and_bind("bind -e")           # Emacs mode: Ctrl+A/E/K etc.
            _rl.parse_and_bind("bind ^I rl_complete")  # Tab completion
        else:
            _rl.parse_and_bind("set editing-mode emacs")  # GNU readline Emacs mode
            _rl.parse_and_bind("tab: complete")

        _rl.set_history_length(500)   # up to 500 history entries
    except ImportError:
        pass  # no readline on Windows; degrade to plain input

    # Main REPL loop
    while True:
        try:
            # Clear the current line (readline doesn't know about leftover chars from streaming output)
            # \r returns to line start; \033[2K clears the entire line; then show the prompt
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
            user_input = input(magenta("you") + " > ").strip()
        except EOFError:
            click.echo()
            break
        except KeyboardInterrupt:
            click.echo()
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit", "/q"):
                break
            elif cmd == "/stats":
                session.print_stats()
            elif cmd == "/clear":
                session._shared_history.clear_except_first()
                click.echo(dim("  History cleared (kept initial context)."))
            elif cmd == "/help":
                click.echo(dim(
                    "  Commands:\n"
                    "    /exit   — quit\n"
                    "    /stats  — show session statistics\n"
                    "    /clear  — clear conversation history\n"
                    "    /help   — show this help\n"
                    "  Anything else is sent to the agent."
                ))
            else:
                click.echo(dim(f"  Unknown command: {user_input}. Type /help for help."))
            continue

        # Run one agent round
        click.echo(dim(f"\n  Agent working..."))
        try:
            session.run_round(user_input)
        except KeyboardInterrupt:
            click.echo(yellow("\n  Interrupted. Type /exit to quit or continue with a new task."))
        except Exception as e:
            click.echo(red(f"\n  Error: {e}"))
            if verbose:
                import traceback
                traceback.print_exc()

    session.print_stats()
    click.echo(dim("  Bye!\n"))


# ---------------------------------------------------------------------------
# web subcommand — browser chat box (#1)
# ---------------------------------------------------------------------------

@cli.command("web")
@click.option("--repo", "-r", default=".", show_default=True, help="Path to the target repository")
@click.option("--model", "-m", default=None, help="Override LLM model name")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8765, show_default=True, type=int, help="Bind port")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def web(
    ctx: click.Context,
    repo: str,
    model: str | None,
    provider: str | None,
    host: str,
    port: int,
    verbose: bool,
) -> None:
    """Serve a browser chat box for the agent (stdlib-only web server)."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(config, provider=provider, model=model)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    try:
        backend = create_backend_from_config({
            "provider":   config.llm.provider,
            "model":      config.llm.model,
            "api_key":    config.llm.api_key or None,
            "base_url":   config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(red(f"Error: {e}"), err=True)
        sys.exit(1)

    registry = _build_registry(config)
    from entry.web import ChatWebApp, serve
    app = ChatWebApp(backend, registry, config, str(repo_path), config.agent.log_dir)

    click.echo(bold(f"\n🌐 Coding Agent — Web Chat"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Repo     : {repo_path}")
    click.echo(green(f"  Open     : http://{host}:{port}\n"))
    click.echo(dim("  Ctrl+C to stop.\n"))
    serve(app, host=host, port=port)


# ---------------------------------------------------------------------------
# multi subcommand — multi-agent orchestrator (#3)
# ---------------------------------------------------------------------------

@cli.command("multi")
@click.option("--repo", "-r", default=".", show_default=True, help="Path to the target repository")
@click.option("--task", "-t", default=None, help="Task description (natural language)")
@click.option("--task-file", "-f", default=None, help="Read task description from file")
@click.option("--model", "-m", default=None, help="Override LLM model name")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--max-steps", default=None, type=int, help="Max steps per role")
@click.option("--iterations", "-i", default=2, show_default=True, type=int, help="Max coder/reviewer iterations")
@click.option("--sandbox", is_flag=True, default=False, help="Run commands in Docker sandbox")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def multi(
    ctx: click.Context,
    repo: str,
    task: str | None,
    task_file: str | None,
    model: str | None,
    provider: str | None,
    max_steps: int | None,
    iterations: int,
    sandbox: bool,
    verbose: bool,
) -> None:
    """Run the planner → coder → reviewer multi-agent pipeline on a task."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(config, provider=provider, model=model, max_steps=max_steps)

    if task_file:
        description = Path(task_file).read_text(encoding="utf-8").strip()
    elif task:
        description = task
    else:
        click.echo(red("Error: provide --task or --task-file"), err=True)
        sys.exit(1)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    try:
        backend = create_backend_from_config({
            "provider": config.llm.provider, "model": config.llm.model,
            "api_key": config.llm.api_key or None, "base_url": config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(red(f"Error: {e}"), err=True)
        sys.exit(1)

    from tools.runtime import create_runtime
    runtime = create_runtime(sandbox=sandbox, repo_path=str(repo_path)) if sandbox else None
    registry = _build_registry(config, runtime=runtime)

    from agent.core import AgentConfig
    from agent.orchestrator import Orchestrator
    from agent.task import Task

    agent_cfg = AgentConfig(max_steps=config.agent.max_steps, budget_tokens=config.agent.budget_tokens)

    click.echo(bold(f"\n🤝 Coding Agent — Multi-Agent (planner → coder → reviewer)"))
    click.echo(f"  Model    : {config.llm.model}   Iterations: {iterations}")
    click.echo(f"  Repo     : {repo_path}\n")

    def _on_role(role: str) -> None:
        click.echo(cyan(f"  ▶ {role}…"))

    orch = Orchestrator(backend, registry, agent_cfg,
                        max_iterations=iterations, on_role_start=_on_role)
    task_obj = Task(description=description, repo_path=str(repo_path),
                    max_steps=config.agent.max_steps)
    result = orch.run(task_obj, log_dir=config.agent.log_dir)

    click.echo(bold("\n" + "─" * 60))
    verdict = green("APPROVED") if result.approved else red("NOT APPROVED")
    click.echo(f"Verdict   : {verdict}  (after {result.iterations} iteration(s))")
    click.echo(f"Steps     : {result.total_steps}   Tokens: {result.total_tokens:,}")
    for r in result.roles:
        click.echo(dim(f"    {r.role:<9} {r.status:<10} {r.steps} steps  {r.tokens} tok"))
    click.echo(bold("─" * 60) + "\n")
    sys.exit(0 if result.is_success() else 1)


# ---------------------------------------------------------------------------
# eval subcommand — benchmark harness
# ---------------------------------------------------------------------------

@cli.command("eval")
@click.option("--model", "-m", default=None, help="Override LLM model name")
@click.option("--provider", "-p", default=None, help="Override LLM provider")
@click.option("--max-steps", default=None, type=int, help="Override per-task max steps")
@click.option("--attempts", "-k", default=1, type=int, show_default=True, help="Run each task k times for pass@1 / pass@k")
@click.option("--retriever", "-R", type=click.Choice(["none", "rag"]), default="none", show_default=True, help="Context retriever for each task")
@click.option("--engine", "-e", type=click.Choice(["native", "langgraph"]), default="native", show_default=True, help="Orchestration engine")
@click.option("--output", "-o", default=None, help="Save the JSON report to this path")
@click.option("--keep", is_flag=True, default=False, help="Keep per-task working directories for debugging")
@click.option("--results-dir", default="./eval_runs", show_default=True, help="Root dir for task workdirs and logs")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    model: str | None,
    provider: str | None,
    max_steps: int | None,
    attempts: int,
    retriever: str,
    engine: str,
    output: str | None,
    keep: bool,
    results_dir: str,
    verbose: bool,
) -> None:
    """Run the benchmark suite and report success rate / steps / tokens."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(config, provider=provider, model=model, max_steps=max_steps)

    from agent.core import Agent, AgentConfig
    from eval.harness import EvalHarness
    from eval.suite import default_suite

    try:
        backend = create_backend_from_config({
            "provider":   config.llm.provider,
            "model":      config.llm.model,
            "api_key":    config.llm.api_key or None,
            "base_url":   config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as e:
        click.echo(red(f"Error: {e}"), err=True)
        sys.exit(1)

    def make_agent(spec, repo_path):
        registry = _build_registry(config)
        rag = _build_retriever(repo_path, retriever)
        agent_cfg = AgentConfig(
            max_steps=spec.max_steps if max_steps is None else max_steps,
            budget_tokens=config.agent.budget_tokens,
            retriever=rag,
        )
        if engine == "langgraph":
            from agent.langgraph_loop import LangGraphAgent
            return LangGraphAgent(backend, registry, agent_cfg)
        return Agent(backend, registry, agent_cfg)

    suite = default_suite()
    click.echo(bold(f"\n🧪 Forge Agent — Eval Harness"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Engine   : {engine}   Retriever: {retriever}")
    click.echo(f"  Tasks    : {len(suite)}   Attempts: {attempts}\n")

    def _progress(r) -> None:
        verdict = green("PASS") if r.passed else red("FAIL")
        click.echo(f"  [{verdict}] {r.task_id:<24} "
                   + dim(f"agent={r.agent_status} steps={r.steps} "
                         f"tokens={r.tokens} {r.elapsed:.1f}s — {r.detail}"))

    harness = EvalHarness(
        agent_factory=make_agent,
        results_dir=results_dir,
        keep_workdirs=keep,
        on_result=_progress,
        model_name=config.llm.model,
    )
    report = harness.run_suite(suite, attempts=attempts)

    click.echo("\n" + report.format_table() + "\n")

    if output:
        report.save_json(output)
        click.echo(dim(f"  Report saved to {output}\n"))

    # Exit code 0 only if all tasks passed
    sys.exit(0 if report.passed == report.total else 1)


# ---------------------------------------------------------------------------
# log subcommand group
# ---------------------------------------------------------------------------

@cli.group()
def log() -> None:
    """Inspect event logs."""


@log.command("show")
@click.argument("log_file")
def log_show(log_file: str) -> None:
    """Show a summary of an event log file."""
    from agent.event_log import EventLog, summarize_run

    path = Path(log_file)
    if not path.exists():
        click.echo(red(f"File not found: {path}"), err=True)
        sys.exit(1)

    with EventLog.open_existing(path) as elog:
        events = elog.replay()
        stats = summarize_run(elog)

    click.echo(bold(f"\nEvent Log: {path.name}"))
    click.echo(f"  Total events : {stats['total_events']}")
    click.echo(f"  Actions      : {stats['actions']}")
    click.echo(f"  Reflections  : {stats['reflections']}")
    click.echo(f"  Tool calls   : {stats['tool_calls']}")
    click.echo(f"  Final status : {stats['final_status']}\n")

    click.echo(bold("Events:"))
    for event in events:
        ts = event.timestamp[11:19]   # HH:MM:SS
        etype = event.event_type.value
        detail = ""
        if event.event_type.value == "action":
            tc = event.payload.get("action", {}).get("tool_call")
            detail = f"  tool={tc['name']}" if tc else ""
        elif event.event_type.value == "observation":
            obs = event.payload.get("observation", {})
            detail = f"  status={obs.get('status')}"
        click.echo(f"  {ts}  {etype:<16}{detail}")


@log.command("list")
@click.option("--dir", "log_dir", default="./logs", help="Log directory")
def log_list(log_dir: str) -> None:
    """List all event log files."""
    log_path = Path(log_dir)
    if not log_path.exists():
        click.echo(f"Log directory not found: {log_path}")
        return

    files = sorted(log_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        click.echo("No log files found.")
        return

    click.echo(bold(f"\nLog files in {log_path}:\n"))
    for f in files:
        size_kb = f.stat().st_size / 1024
        click.echo(f"  {f.name}  ({size_kb:.1f} KB)")
    click.echo()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
