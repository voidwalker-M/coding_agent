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

from config.schema import load_config, merge_cli_overrides, agent_compaction_kwargs  # noqa: E402
from llm.router import create_backend_from_config            # noqa: E402


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

def _setup_logging(verbose: bool, log_file: "Path | None" = None,
                   verbose_level: int = logging.DEBUG) -> None:
    """Configure root logging: stderr always, plus a file handler when given.

    force=True so repeated in-process invocations (Click's CliRunner in tests)
    reconfigure instead of silently keeping the first call's handlers.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=verbose_level if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _build_registry(cfg, confirm_callback=None, runtime=None, memory=None, symbol_index=None,
                    undo_manager=None, repo_path: str = ".",
                    block_test_edits: bool = False):
    """Assemble the tool registry from configuration.

    memory:       a LongTermMemory instance; when given, the remember/recall tools
                  are registered so the agent can manage its own persistent memory.
    symbol_index: a prebuilt SymbolIndex; when given, find_symbol answers from it
                  (O(1) indexed lookup) instead of re-scanning the repo per call.
    undo_manager: shared UndoManager backing file snapshots and the undo tool.
                  Defaults to an in-memory-only one (no sidecar file on disk).
    repo_path:    used by SkillTool to discover `.agent/skills/*/SKILL.md`.
    block_test_edits: when True, file_write/file_edit refuse tests/ paths (SWE-bench).
    """
    from tools.base import ToolRegistry
    from tools.file_tool import FileEditTool, FileReadTool, FileViewTool, FileWriteTool
    from tools.git_tool import GitAddTool, GitCommitTool, GitDiffTool, GitStatusTool
    from tools.search_tool import FindFilesTool, FindSymbolTool, SearchTextTool
    from tools.shell_tool import ShellTool
    from tools.test_tool import PytestTool
    from tools.undo_tool import UndoManager, UndoTool

    if undo_manager is None:
        undo_manager = UndoManager()

    registry = (
        ToolRegistry()
        .register(ShellTool(confirm_callback=confirm_callback, runtime=runtime))
        .register(FileReadTool())
        .register(FileViewTool())
        .register(FileWriteTool(undo_manager=undo_manager, block_test_edits=block_test_edits))
        .register(FileEditTool(undo_manager=undo_manager, block_test_edits=block_test_edits))
        .register(UndoTool(undo_manager))
        .register(SearchTextTool())
        .register(FindFilesTool())
        .register(FindSymbolTool(index=symbol_index))
        .register(PytestTool(runtime=runtime))
        .register(GitStatusTool(runtime=runtime))
        .register(GitDiffTool(runtime=runtime))
        .register(GitAddTool(runtime=runtime))
        .register(GitCommitTool(runtime=runtime))
    )
    from tools.plan_tool import PlanTool
    from tools.skill_tool import SkillTool
    from tools.browser_tool import WebFetchTool
    registry.register(PlanTool())
    registry.register(SkillTool(repo_path=repo_path))
    registry.register(WebFetchTool())
    if memory is not None:
        from tools.memory_tool import RecallTool, RememberTool
        registry.register(RememberTool(memory)).register(RecallTool(memory))
    return registry


def _build_symbol_index(repo_path: str):
    """Build a persistent, incremental SymbolIndex for the repo (best-effort).

    Cached under <repo>/.symbol_cache so only changed files are re-parsed on
    subsequent runs. Returns None on any failure (find_symbol then falls back to
    its regex scan), so this never blocks a run.
    """
    from pathlib import Path as _Path
    from context.symbol_index import SymbolIndex
    try:
        return SymbolIndex(repo_path, cache_dir=str(_Path(repo_path) / ".symbol_cache")).build()
    except Exception:
        return None


def _build_memory(repo_path: str, cfg, *, enable: bool):
    """Build a LongTermMemory when memory is enabled (config or --memory), else None.

    Stores SQLite (`memory.db`) + a markdown cache/index under `memory.dir`
    (default <repo>/.agent_memory). Recall is filtered by the configured user/role.
    Uses the RAG embedding backend for dense recall when numpy is available;
    otherwise degrades to the built-in lexical engine.
    """
    from context.store_factory import memory_wanted, open_memory_store
    if not enable and not memory_wanted():
        return None
    from pathlib import Path as _Path
    from context.memory import LongTermMemory

    mem_dir = cfg.memory.dir or str(_Path(repo_path) / ".agent_memory")
    store = open_memory_store(_Path(mem_dir) / "memory.db")
    embeddings = None
    try:
        from context.rag import create_embedding_backend
        embeddings = create_embedding_backend()  # OpenAI if keyed, else hashing (offline)
    except Exception:
        embeddings = None
    from context.auth import load_session
    sess = load_session()
    if sess:
        user_id = sess["user_id"]
        role = sess.get("role") or "user"
    else:
        user_id = getattr(cfg.memory, "default_user", "default")
        role = getattr(cfg.memory, "default_role", "agent")
    return LongTermMemory(
        mem_dir,
        embeddings=embeddings,
        max_records=cfg.memory.max_records,
        user_id=user_id,
        role=role,
        auto_approve=getattr(cfg.memory, "auto_approve", True),
        store=store,
    ).load()


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


def _open_user_store(ctx: click.Context, repo: str):
    """Same memory store chat/run use, so registered users own LTM/STM there."""
    from context.store_factory import open_memory_store

    config = load_config(ctx.obj.get("config_path"))
    repo_path = Path(repo).resolve()
    mem_dir = config.memory.dir or str(repo_path / ".agent_memory")
    return open_memory_store(Path(mem_dir) / "memory.db")


# ---------------------------------------------------------------------------
# register / login / logout / whoami
# ---------------------------------------------------------------------------

@cli.command("register")
@click.option("--username", "-u", prompt=True, help="Login name (becomes memory user_id)")
@click.option(
    "--password", "-p",
    prompt=True, hide_input=True, confirmation_prompt=True,
    help="Password (hidden; not stored in plaintext)",
)
@click.option("--repo", "-r", default=".", show_default=True)
@click.pass_context
def register_cmd(ctx: click.Context, username: str, password: str, repo: str) -> None:
    """Create a local user. Password is stored as PBKDF2-SHA256."""
    from context.auth import AuthError, register_user, save_session

    store = _open_user_store(ctx, repo)
    try:
        user = register_user(store, username, password)
    except AuthError as exc:
        click.echo(red(f"Error: {exc}"), err=True)
        sys.exit(1)
    finally:
        store.close()
    save_session(user.id, user.role)
    click.echo(green(f"Registered and logged in as {user.id}"))
    click.echo(dim("  Memory for chat/run --memory will use this user_id."))


@cli.command("login")
@click.option("--username", "-u", prompt=True)
@click.option("--password", "-p", prompt=True, hide_input=True)
@click.option("--repo", "-r", default=".", show_default=True)
@click.pass_context
def login_cmd(ctx: click.Context, username: str, password: str, repo: str) -> None:
    """Log in; later commands use this user_id for memory."""
    from context.auth import AuthError, authenticate, save_session

    store = _open_user_store(ctx, repo)
    try:
        user = authenticate(store, username, password)
    except AuthError as exc:
        click.echo(red(f"Error: {exc}"), err=True)
        sys.exit(1)
    finally:
        store.close()
    save_session(user.id, user.role)
    click.echo(green(f"Logged in as {user.id}"))


@cli.command("logout")
def logout_cmd() -> None:
    """Clear the local login session."""
    from context.auth import clear_session, load_session

    sess = load_session()
    clear_session()
    if sess:
        click.echo(f"Logged out {sess['user_id']}")
    else:
        click.echo("Not logged in")


@cli.command("whoami")
def whoami_cmd() -> None:
    """Show the currently logged-in user."""
    from context.auth import load_session, session_path

    sess = load_session()
    if not sess:
        click.echo("Not logged in (memory user_id = config default, usually 'default')")
        return
    click.echo(f"{sess['user_id']}  role={sess['role']}")
    click.echo(dim(f"  session: {session_path()}"))


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
@click.option("--memory", is_flag=True, default=False, help="Enable persistent long-term memory (markdown store + recall/remember tools)")
@click.option("--memory-dir", default=None, type=click.Path(), help="Directory for long-term memory (default: <repo>/.agent_memory)")
@click.option("--compact", is_flag=True, default=False, help="Enable LLM history compaction when conversation exceeds trigger_tokens")
@click.option("--engine", "-e", type=click.Choice(["native", "langgraph"]), default="native", show_default=True, help="Orchestration engine: 'langgraph' runs the LangGraph port")
@click.option("--cache", is_flag=True, default=False, help="Cache LLM responses (saves tokens on repeated/identical calls)")
@click.option("--cheap-model", default=None, help="Enable cost-aware routing: cheapest tier model")
@click.option("--mid-model", default=None, help="Optional middle-tier model for 3-tier routing")
@click.option("--router", type=click.Choice(["heuristic", "difficulty", "cascade"]), default="heuristic", show_default=True, help="Routing policy when a cheap/mid model is set: keyword heuristic, up-front difficulty estimate, or confidence cascade")
@click.option("--min-confidence", default=0.35, show_default=True, type=float, help="Cascade router: escalate to a stronger model when confidence falls below this")
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
    memory: bool,
    memory_dir: str | None,
    compact: bool,
    engine: str,
    cache: bool,
    cheap_model: str | None,
    mid_model: str | None,
    router: str,
    min_confidence: float,
    max_usd: float | None,
    rpm: int | None,
    verbose: bool,
) -> None:
    """Run the coding agent on a repository."""
    from datetime import datetime, timezone

    from agent.task import Task

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

    # Build the task up front so this run's three artifacts — event log, undo
    # sidecar, text log — all share one {task_id}_{timestamp} stem.
    task_obj = Task(
        description=description,
        repo_path=str(repo_path),
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
    )
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir_path = Path(config.agent.log_dir)
    stem = f"{task_obj.task_id}_{run_ts}"
    _setup_logging(verbose, log_dir_path / f"{stem}.log")

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
    if cache or cheap_model or mid_model or max_usd is not None or rpm is not None:
        from llm.compose import compose_backend

        def _tier_backend(name: str):
            return create_backend_from_config({
                "provider": config.llm.provider, "model": name,
                "api_key": config.llm.api_key or None, "base_url": config.llm.base_url or None,
                "max_tokens": config.llm.max_tokens,
            })

        cheap_backend = _tier_backend(cheap_model) if cheap_model else None
        mid_backend = _tier_backend(mid_model) if mid_model else None
        backend = compose_backend(
            backend, cheap=cheap_backend, mid=mid_backend,
            route_mode=router, min_confidence=min_confidence, cache=cache,
            rpm=rpm, max_usd=max_usd, model_for_cost=config.llm.model,
        )
        routing_on = bool(cheap_model or mid_model)
        extras = [n for n, on in (("cache", cache), (f"router:{router}", routing_on),
                                  ("max_usd", max_usd is not None), ("rpm", rpm is not None)) if on]
        click.echo(dim(f"  Token-saving: {', '.join(extras)}"))
        if routing_on:
            tiers = [m for m in (cheap_model, mid_model, config.llm.model) if m]
            click.echo(dim(f"  Router tiers (cheap→strong): {' → '.join(tiers)}"))

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
    if config.context.compaction.enabled or compact:
        click.echo(dim(
            f"  Compaction: on (trigger {config.context.compaction.trigger_tokens:,} tokens, "
            f"keep recent {config.context.compaction.keep_recent_messages})"
        ))
    # Long-term memory (#2): enabled by --memory or config; --memory-dir overrides.
    if memory_dir:
        config.memory.dir = memory_dir
    memory_on = memory or config.memory.enabled
    from context.auth import load_session
    sess = load_session()
    if sess:
        memory_on = True
        click.echo(dim(f"  User: {sess['user_id']} (logged in)"))
    long_term = _build_memory(str(repo_path), config, enable=memory_on)
    if long_term is not None:
        click.echo(dim(f"  Memory: long-term on ({long_term.count} records, {long_term.backend_info})"))
    # Symbol index (#3): build once so find_symbol is an O(1) indexed lookup.
    symbol_index = _build_symbol_index(str(repo_path))
    if symbol_index is not None:
        click.echo(dim(f"  Symbol index: {symbol_index.symbol_count} symbols "
                       f"in {symbol_index.name_count} names "
                       f"({symbol_index.stats.get('reparsed_files', 0)} parsed, "
                       f"{symbol_index.stats.get('reused_files', 0)} cached)"))
    from tools.undo_tool import UndoManager
    undo_manager = UndoManager(sidecar_path=log_dir_path / f"{stem}.undo.jsonl")
    registry = _build_registry(config, confirm_callback=confirm_cb, runtime=runtime,
                               memory=long_term, symbol_index=symbol_index,
                               undo_manager=undo_manager, repo_path=str(repo_path))

    from agent.core import Agent, AgentConfig
    from agent.event_log import EventLog, summarize_run
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
        long_term_memory=long_term,
        memory_top_k=config.memory.top_k,
        memory_window_queries=config.memory.window_queries,
        capture_episodes=config.memory.capture_episodes,
        **agent_compaction_kwargs(config.context, enabled_override=(config.context.compaction.enabled or compact)),
    )
    if engine == "langgraph":
        from agent.langgraph_loop import LangGraphAgent
        click.echo(dim("  Engine: LangGraph (streaming disabled)"))
        agent = LangGraphAgent(backend, registry, agent_config)
    else:
        agent = Agent(backend, registry, agent_config)

    if verbose:
        click.echo(dim(
            f"  tiktoken: {'yes' if is_tiktoken_available() else 'no (char estimate)'}\n"
        ))

    # Run
    t0 = time.time()
    with EventLog.create(task_obj, log_dir=config.agent.log_dir, timestamp=run_ts) as log:
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
    overhead = max(0.0, elapsed - result.llm_time - result.tool_time)
    click.echo(
        f"Time    : {elapsed:.1f}s "
        f"(llm={result.llm_time:.1f}s tool={result.tool_time:.1f}s overhead={overhead:.1f}s)"
    )
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
    import uuid
    from datetime import datetime, timezone

    from entry.chat import ChatSession

    config = load_config(ctx.obj.get("config_path"))
    config = merge_cli_overrides(config, provider=provider, model=model, max_steps=max_steps)

    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    # Undo and the text log are scoped to the whole session, not one round: the
    # registry (and so the agent's tools) is shared across every round here.
    session_stem = (f"chat_{uuid.uuid4().hex[:8]}_"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    log_dir_path = Path(config.agent.log_dir)
    _setup_logging(verbose, log_dir_path / f"{session_stem}.log")

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

    from tools.undo_tool import UndoManager
    undo_manager = UndoManager(sidecar_path=log_dir_path / f"{session_stem}.undo.jsonl")
    from context.auth import load_session
    from context.memory import ShortTermMemory

    sess = load_session()
    memory_on = bool(sess) or config.memory.enabled
    long_term = _build_memory(str(repo_path), config, enable=memory_on)
    stm = None
    if long_term is not None:
        user_id = (sess or {}).get("user_id") or getattr(config.memory, "default_user", "default")
        conv_id = long_term.new_conversation(title="chat")
        stm = ShortTermMemory(
            window_queries=getattr(config.memory, "window_queries", 10),
            store=long_term.store,
            user_id=user_id,
            conversation_id=conv_id,
            on_overflow=long_term.ingest_overflow,
        )
    registry = _build_registry(
        config, undo_manager=undo_manager, repo_path=str(repo_path), memory=long_term,
    )
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
        short_term_memory=stm,
        long_term_memory=long_term,
    )

    # Welcome message
    click.echo(bold(f"\n🤖 Coding Agent — Chat Mode"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Repo     : {repo_path}")
    if sess:
        click.echo(f"  User     : {sess['user_id']}")
    elif long_term is not None:
        click.echo(dim(f"  User     : {long_term.user_id} (not logged in)"))
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
    _setup_logging(verbose, verbose_level=logging.INFO)
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

    from context.auth import load_session
    from context.memory import ShortTermMemory

    sess = load_session()
    memory_on = bool(sess) or config.memory.enabled
    long_term = _build_memory(str(repo_path), config, enable=memory_on)
    stm = None
    if long_term is not None:
        user_id = (sess or {}).get("user_id") or getattr(config.memory, "default_user", "default")
        conv_id = long_term.new_conversation(title="web")
        stm = ShortTermMemory(
            window_queries=getattr(config.memory, "window_queries", 10),
            store=long_term.store,
            user_id=user_id,
            conversation_id=conv_id,
            on_overflow=long_term.ingest_overflow,
        )
    registry = _build_registry(config, repo_path=str(repo_path), memory=long_term)
    from entry.web import ChatWebApp, serve
    app = ChatWebApp(
        backend, registry, config, str(repo_path), config.agent.log_dir,
        short_term_memory=stm, long_term_memory=long_term,
    )

    click.echo(bold(f"\n🌐 Coding Agent — Web Chat"))
    click.echo(f"  Provider : {config.llm.provider}")
    click.echo(f"  Model    : {config.llm.model}")
    click.echo(f"  Repo     : {repo_path}")
    click.echo(green(f"  Open     : http://{host}:{port}\n"))
    click.echo(dim("  Ctrl+C to stop.\n"))
    serve(app, host=host, port=port)


# ---------------------------------------------------------------------------
# serve subcommand — FastAPI + optional gRPC
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option("--repo", "-r", default=".", show_default=True, help="Default repo path for tasks")
@click.option("--host", default="0.0.0.0", show_default=True, help="HTTP bind host")
@click.option("--port", default=8766, show_default=True, type=int, help="HTTP bind port")
@click.option("--grpc-port", default=0, show_default=True, type=int,
              help="gRPC bind port (0 = disabled)")
@click.option("--workers", default=4, show_default=True, type=int, help="Thread-pool workers")
@click.option("--checkpoint-dir", default="./checkpoints", show_default=True,
              help="Directory for resumable checkpoints")
@click.option("--verbose", "-v", is_flag=True, help="Show debug logs")
@click.pass_context
def serve_cmd(
    ctx: click.Context,
    repo: str,
    host: str,
    port: int,
    grpc_port: int,
    workers: int,
    checkpoint_dir: str,
    verbose: bool,
) -> None:
    """Serve the agent over FastAPI (REST) with optional gRPC."""
    _setup_logging(verbose, verbose_level=logging.INFO)
    config = load_config(ctx.obj.get("config_path"))
    repo_path = Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)

    try:
        import uvicorn
    except ImportError:
        click.echo(red("Error: uvicorn not installed. Run: pip install 'coding-agent[server]'"), err=True)
        sys.exit(1)

    from context.store_factory import memory_wanted
    from entry.api import app, configure_service
    from entry.runtime import build_runtime
    from llm.router import create_backend_from_config

    backend = None
    llm_note = f"{config.llm.provider}/{config.llm.model}"
    try:
        backend = create_backend_from_config({
            "provider": config.llm.provider,
            "model": config.llm.model,
            "api_key": config.llm.api_key or None,
            "base_url": config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })
    except ValueError as exc:
        from llm.base import OfflineBackend
        backend = OfflineBackend()
        llm_note = f"offline ({exc})"

    memory_on = config.memory.enabled or memory_wanted()
    runtime = build_runtime(
        config_path=ctx.obj.get("config_path"),
        repo_path=str(repo_path),
        backend=backend,
        stream=True,
        memory=memory_on,
    )
    configure_service(
        repo_path=str(repo_path),
        checkpoint_dir=checkpoint_dir,
        log_dir=config.agent.log_dir,
        workers=workers,
        runtime=runtime,
    )
    click.echo(bold(f"Agent API listening on http://{host}:{port}"))
    click.echo(dim(f"  repo={repo_path}  checkpoints={checkpoint_dir}  workers={workers}"))
    click.echo(dim(f"  llm={llm_note}  memory={memory_on}"))

    if grpc_port:
        import threading
        from entry.grpc_server import serve_grpc
        threading.Thread(
            target=serve_grpc, kwargs={"host": host, "port": grpc_port, "workers": workers},
            daemon=True,
        ).start()
        click.echo(dim(f"  gRPC on {host}:{grpc_port}"))

    uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")


@cli.command("load-test")
@click.option("--url", default="http://127.0.0.1:8766", show_default=True)
@click.option("--concurrency", "-c", default=4, show_default=True, type=int)
@click.option("--requests", "-n", default=20, show_default=True, type=int)
@click.option("--repo", "-r", default=".", show_default=True)
def load_test_cmd(url: str, concurrency: int, requests: int, repo: str) -> None:
    """Run a simple concurrent load test against agent serve."""
    import json
    from entry.load_test import run_load_test
    report = run_load_test(url, concurrency=concurrency, requests=requests, repo_path=repo)
    click.echo(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# mcp subcommand — expose tools via Model Context Protocol
# ---------------------------------------------------------------------------

@cli.command("mcp")
@click.option("--repo", "-r", default=".", show_default=True, help="Repository root path")
@click.option(
    "--transport", "-t", default="stdio",
    type=click.Choice(["stdio", "http", "streamable-http"]),
    show_default=True,
    help="MCP transport (stdio for Claude Desktop / Cursor)",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8767, show_default=True, type=int)
@click.option("--include-writes", is_flag=True, help="Expose file_edit/file_write/pytest")
@click.pass_context
def mcp_cmd(
    ctx: click.Context,
    repo: str,
    transport: str,
    host: str,
    port: int,
    include_writes: bool,
) -> None:
    """Run the agent tool registry as an MCP server."""
    import os
    from pathlib import Path as _Path

    repo_path = _Path(repo).resolve()
    if not repo_path.exists():
        click.echo(red(f"Error: repo path does not exist: {repo_path}"), err=True)
        sys.exit(1)
    os.chdir(repo_path)

    from context.mcp_bridge import build_mcp_server, run_mcp_server_blocking
    cfg = load_config(ctx.obj.get("config_path"))
    symbol_index = _build_symbol_index(str(repo_path))
    registry = _build_registry(cfg, symbol_index=symbol_index, repo_path=str(repo_path))
    server = build_mcp_server(
        registry, repo_path=str(repo_path), include_writes=include_writes,
    )
    names = [t.name for t in __import__("asyncio").run(server.list_tools())]
    click.echo(dim(f"  MCP tools ({len(names)}): {', '.join(names)}"))
    if transport == "stdio":
        click.echo(dim("  Transport: stdio (ready for IDE MCP host)"))
    else:
        click.echo(dim(f"  Transport: {transport} on {host}:{port}"))
    run_mcp_server_blocking(server, transport=transport, host=host, port=port)


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
@click.option("--topology", type=click.Choice(["pipeline", "pair", "debate", "autonomous"]),
              default="pipeline", show_default=True,
              help="Multi-agent topology")
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
    topology: str,
    sandbox: bool,
    verbose: bool,
) -> None:
    """Run a multi-agent topology (pipeline / pair / debate / autonomous) on a task."""
    _setup_logging(verbose)
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
    registry = _build_registry(config, runtime=runtime, repo_path=str(repo_path))

    from agent.core import AgentConfig
    from agent.orchestrator import Orchestrator
    from agent.task import Task

    agent_cfg = AgentConfig(max_steps=config.agent.max_steps, budget_tokens=config.agent.budget_tokens)

    click.echo(bold(f"\n🤝 Coding Agent — Multi-Agent ({topology})"))
    click.echo(f"  Model    : {config.llm.model}   Iterations: {iterations}")
    click.echo(f"  Repo     : {repo_path}\n")

    def _on_role(role: str) -> None:
        click.echo(cyan(f"  ▶ {role}…"))

    orch = Orchestrator(backend, registry, agent_cfg,
                        max_iterations=iterations, on_role_start=_on_role,
                        topology=topology)
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
    from datetime import datetime, timezone

    eval_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _setup_logging(verbose, Path(results_dir) / f"eval_{eval_ts}.log")

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
        registry = _build_registry(config, repo_path=repo_path)
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
    click.echo(f"  LLM time     : {stats['llm_time_total']:.1f}s")
    click.echo(f"  Tool time    : {stats['tool_time_total']:.1f}s")
    if stats["tool_time_by_name"]:
        slowest = sorted(stats["tool_time_by_name"].items(), key=lambda kv: -kv[1])
        breakdown = "  ".join(f"{n}={s:.1f}s" for n, s in slowest)
        click.echo(f"  Time by tool : {breakdown}")
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

    files = sorted((p for p in log_path.glob("*.jsonl")
                    if not p.name.endswith(".undo.jsonl")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        click.echo("No log files found.")
        return

    click.echo(bold(f"\nLog files in {log_path}:\n"))
    for f in files:
        size_kb = f.stat().st_size / 1024
        undo = " [undo available]" if f.with_suffix(".undo.jsonl").exists() else ""
        click.echo(f"  {f.name}  ({size_kb:.1f} KB){dim(undo)}")
    click.echo()


# ---------------------------------------------------------------------------
# undo subcommand — roll back a previous run's file changes
# ---------------------------------------------------------------------------

@cli.command("undo")
@click.argument("target")
@click.option("--log-dir", default=None, help="Where to look when TARGET is a bare task id")
@click.option("--file", "only_path", default=None, help="Only restore this one file")
@click.option("--list", "list_only", is_flag=True, default=False,
              help="Preview the rollback without changing anything")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt")
@click.pass_context
def undo_cmd(ctx: click.Context, target: str, log_dir: str | None,
             only_path: str | None, list_only: bool, yes: bool) -> None:
    """Roll back file changes made by a previous run.

    TARGET is a task id, an event log (.jsonl), or an undo sidecar (.undo.jsonl).
    Each file is restored to its content from before the run first touched it.
    """
    import difflib

    from tools.undo_tool import UndoManager

    config = load_config(ctx.obj.get("config_path"))
    t = Path(target)
    if t.name.endswith(".undo.jsonl"):
        sidecar = t
    elif t.suffix == ".jsonl":
        sidecar = UndoManager.sidecar_for_log_path(t)
    else:
        sidecar = UndoManager.find_sidecar(Path(log_dir or config.agent.log_dir), target)

    if sidecar is None or not sidecar.exists():
        click.echo(red(f"No undo snapshot found for '{target}'"), err=True)
        sys.exit(1)

    plan = UndoManager.rollback_plan(UndoManager.load_sidecar(sidecar), path=only_path)
    if not plan:
        click.echo(dim("Nothing to roll back."))
        return

    click.echo(bold(f"\nRollback plan ({sidecar.name}):"))
    for m in plan:
        p = Path(m.path)
        if not m.existed_before:
            if p.exists():
                click.echo(red(f"  delete   {m.path}") + dim("  (created during this run)"))
            else:
                click.echo(dim(f"  ok       {m.path}  (already absent)"))
            continue
        current = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        if current == (m.content_before or ""):
            click.echo(dim(f"  ok       {m.path}  (already at pre-run content)"))
            continue
        changed = sum(
            1 for line in difflib.unified_diff(
                current.splitlines(), (m.content_before or "").splitlines(), lineterm="")
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )
        click.echo(yellow(f"  restore  {m.path}") + dim(f"  ({changed} lines differ)"))

    if list_only:
        return
    if not yes and not click.confirm("\nApply this rollback?"):
        click.echo(dim("Aborted."))
        return

    restored = 0
    for m in plan:
        try:
            UndoManager._restore(m)
            restored += 1
        except OSError as exc:
            click.echo(red(f"  failed to restore {m.path}: {exc}"), err=True)
    click.echo(green(f"Restored {restored} file(s).\n"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
