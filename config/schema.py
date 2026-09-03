"""
config/schema.py

Configuration file loading and validation.
Parses config/default.yaml into type-safe dataclasses.

Supports:
- Environment variable expansion: ${VAR} syntax
- Multi-layer config merging: default.yaml < user-specified yaml < CLI arguments
- Clear error messages when required fields are missing
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096


@dataclass
class AgentCfg:
    max_steps: int = 40
    budget_tokens: int = 80_000
    log_dir: str = "./logs"


@dataclass
class ShellToolConfig:
    timeout: int = 30
    max_output_tokens: int = 8_000


@dataclass
class FileToolConfig:
    max_view_lines: int = 100


@dataclass
class ToolsConfig:
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)
    file: FileToolConfig = field(default_factory=FileToolConfig)


@dataclass
class CompactionConfig:
    enabled: bool = False
    trigger_tokens: int = 24_000
    keep_recent_messages: int = 8
    min_messages_to_compact: int = 4


@dataclass
class ContextConfig:
    repo_map_budget: int = 8_000
    history_window: int = 20
    compaction: CompactionConfig = field(default_factory=CompactionConfig)


@dataclass
class MemoryConfig:
    enabled: bool = False
    dir: str = ""                 # empty → <repo>/.agent_memory (resolved by the CLI)
    top_k: int = 4                # long-term records recalled per task
    max_records: int = 500        # cap before consolidation evicts the weakest
    capture_episodes: bool = True # store an episodic memory at end of each run
    window_queries: int = 10      # short-term conversation window (n user queries)
    default_user: str = "default"
    default_role: str = "agent"   # guest | user | agent | admin
    auto_approve: bool = True     # False → proposed memories stay pending (Cursor-style)


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentCfg = field(default_factory=AgentCfg)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(text: str) -> str:
    """Expand ${VAR} environment variable placeholders."""
    def replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_RE.sub(replace, text)


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load configuration file and return an AppConfig.

    Args:
        path: YAML file path; when None, automatically searches for config/default.yaml

    Returns:
        AppConfig instance
    """
    if path is None:
        # Auto-discovery: current directory → project root
        candidates = [
            Path("config/default.yaml"),
            Path(__file__).parent / "default.yaml",
        ]
        for p in candidates:
            if p.exists():
                path = p
                break
        else:
            return AppConfig()   # no config file found; use all defaults

    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    raw = config_path.read_text(encoding="utf-8")
    raw = _expand_env(raw)
    data: dict[str, Any] = yaml.safe_load(raw) or {}
    return _parse(data)


def _parse(data: dict[str, Any]) -> AppConfig:
    """Parse a yaml dict into an AppConfig."""
    llm_raw = data.get("llm", {})
    agent_raw = data.get("agent", {})
    tools_raw = data.get("tools", {})
    context_raw = data.get("context", {})
    memory_raw = data.get("memory", {})

    llm = LLMConfig(
        provider=llm_raw.get("provider", "anthropic"),
        model=llm_raw.get("model", "claude-sonnet-4-5"),
        api_key=llm_raw.get("api_key", ""),
        base_url=llm_raw.get("base_url", "") or "",
        max_tokens=int(llm_raw.get("max_tokens", 4096)),
    )

    agent = AgentCfg(
        max_steps=int(agent_raw.get("max_steps", 40)),
        budget_tokens=int(agent_raw.get("budget_tokens", 80_000)),
        log_dir=agent_raw.get("log_dir", "./logs"),
    )

    shell_raw = tools_raw.get("shell", {})
    file_raw = tools_raw.get("file", {})
    tools = ToolsConfig(
        shell=ShellToolConfig(
            timeout=int(shell_raw.get("timeout", 30)),
            max_output_tokens=int(shell_raw.get("max_output_tokens", 8_000)),
        ),
        file=FileToolConfig(
            max_view_lines=int(file_raw.get("max_view_lines", 100)),
        ),
    )

    context = ContextConfig(
        repo_map_budget=int(context_raw.get("repo_map_budget", 8_000)),
        history_window=int(context_raw.get("history_window", 20)),
        compaction=_parse_compaction(context_raw.get("compaction", {})),
    )

    memory = MemoryConfig(
        enabled=bool(memory_raw.get("enabled", False)),
        dir=memory_raw.get("dir", "") or "",
        top_k=int(memory_raw.get("top_k", 4)),
        max_records=int(memory_raw.get("max_records", 500)),
        capture_episodes=bool(memory_raw.get("capture_episodes", True)),
        window_queries=int(memory_raw.get("window_queries", 10)),
        default_user=str(memory_raw.get("default_user", "default") or "default"),
        default_role=str(memory_raw.get("default_role", "agent") or "agent"),
        auto_approve=bool(memory_raw.get("auto_approve", True)),
    )

    return AppConfig(llm=llm, agent=agent, tools=tools, context=context, memory=memory)


def _parse_compaction(raw: dict[str, Any]) -> CompactionConfig:
    if not isinstance(raw, dict):
        raw = {}
    return CompactionConfig(
        enabled=bool(raw.get("enabled", False)),
        trigger_tokens=int(raw.get("trigger_tokens", 24_000)),
        keep_recent_messages=int(raw.get("keep_recent_messages", 8)),
        min_messages_to_compact=int(raw.get("min_messages_to_compact", 4)),
    )


def agent_compaction_kwargs(context: ContextConfig, *, enabled_override: bool | None = None) -> dict[str, int | bool]:
    """Map ContextConfig.compaction → AgentConfig keyword arguments."""
    c = getattr(context, "compaction", None)
    if c is None:
        enabled = False if enabled_override is None else enabled_override
        return {
            "compaction_enabled": enabled,
            "compaction_trigger_tokens": 24_000,
            "compaction_keep_recent": 8,
            "compaction_min_messages": 4,
        }
    enabled = c.enabled if enabled_override is None else enabled_override
    return {
        "compaction_enabled": enabled,
        "compaction_trigger_tokens": c.trigger_tokens,
        "compaction_keep_recent": c.keep_recent_messages,
        "compaction_min_messages": c.min_messages_to_compact,
    }


def merge_cli_overrides(
    config: AppConfig,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
) -> AppConfig:
    """
    Apply CLI argument overrides onto an already-loaded config.
    CLI arguments have the highest priority.
    """
    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    if max_steps is not None:
        config.agent.max_steps = max_steps
    return config
