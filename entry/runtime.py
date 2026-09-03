"""
entry/runtime.py

Shared runtime for CLI, FastAPI, and gRPC entry points: build Agent + registry
from AppConfig without duplicating cli.py wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.core import Agent, AgentConfig
from agent.event_log import EventLog
from agent.task import Task
from config.schema import AppConfig, load_config, agent_compaction_kwargs
from llm.base import LLMBackend
from llm.router import create_backend_from_config
from tools.base import ToolRegistry


@dataclass
class AgentRuntime:
    agent: Agent
    registry: ToolRegistry
    config: AppConfig
    backend: LLMBackend


def build_runtime(
    *,
    config_path: str | None = None,
    repo_path: str = ".",
    backend: LLMBackend | None = None,
    stream: bool = False,
    sandbox: bool = False,
    retriever: str = "none",
    memory: bool = False,
) -> AgentRuntime:
    """Construct Agent + ToolRegistry from config (or inject backend for tests)."""
    from entry.cli import (
        _build_memory,
        _build_registry,
        _build_retriever,
        _build_symbol_index,
    )
    from tools.runtime import create_runtime

    config = load_config(config_path)
    repo = str(Path(repo_path).resolve())

    if backend is None:
        backend = create_backend_from_config({
            "provider": config.llm.provider,
            "model": config.llm.model,
            "api_key": config.llm.api_key or None,
            "base_url": config.llm.base_url or None,
            "max_tokens": config.llm.max_tokens,
        })

    from context.kv_cache import open_kv_cache, redis_url
    if redis_url():
        kv = open_kv_cache()
        if getattr(kv, "kind", "") == "redis":
            from llm.cache import CachingBackend
            backend = CachingBackend(backend, kv=kv)

    runtime = create_runtime(sandbox=sandbox, repo_path=repo) if sandbox else None
    rag = _build_retriever(repo, retriever)
    ltm = _build_memory(repo, config, enable=memory or config.memory.enabled)
    symbol_index = _build_symbol_index(repo)
    from tools.undo_tool import UndoManager
    undo = UndoManager()
    registry = _build_registry(
        config, runtime=runtime, memory=ltm, symbol_index=symbol_index,
        undo_manager=undo, repo_path=repo,
    )

    agent_config = AgentConfig(
        max_steps=config.agent.max_steps,
        budget_tokens=config.agent.budget_tokens,
        history_max_messages=config.context.history_window * 2,
        stream=stream,
        retriever=rag,
        long_term_memory=ltm,
        memory_top_k=config.memory.top_k,
        memory_window_queries=config.memory.window_queries,
        capture_episodes=config.memory.capture_episodes,
        **agent_compaction_kwargs(config.context),
    )
    agent = Agent(backend, registry, agent_config)
    return AgentRuntime(agent=agent, registry=registry, config=config, backend=backend)


def new_task(description: str, repo_path: str, *, max_steps: int | None = None) -> Task:
    cfg = load_config()
    return Task(
        description=description,
        repo_path=str(Path(repo_path).resolve()),
        max_steps=max_steps or cfg.agent.max_steps,
        budget_tokens=cfg.agent.budget_tokens,
    )


def new_event_log(task: Task, log_dir: str | None = None) -> EventLog:
    cfg = load_config()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return EventLog.create(task, log_dir=log_dir or cfg.agent.log_dir, timestamp=ts)
