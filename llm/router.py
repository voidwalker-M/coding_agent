"""
llm/router.py

Select and instantiate the correct LLMBackend based on configuration.

Supported providers:
    anthropic  → AnthropicBackend
    openai     → OpenAICompatBackend (base_url=None)
    deepseek   → OpenAICompatBackend (base_url=https://api.deepseek.com)
    groq       → OpenAICompatBackend (base_url=https://api.groq.com/openai/v1)
    ollama     → OpenAICompatBackend (base_url=http://localhost:11434/v1)
    vllm       → OpenAICompatBackend (base_url=http://localhost:8000/v1)

To add a new provider, add one entry to _PROVIDER_BASE_URLS.
"""

from __future__ import annotations

import os

from llm.base import LLMBackend

# provider → base_url mapping (None means use the SDK default)
_PROVIDER_BASE_URLS: dict[str, str | None] = {
    "anthropic": None,      # routes to AnthropicBackend, not used in this table
    "openai":    None,
    "deepseek":  "https://api.deepseek.com",
    "groq":      "https://api.groq.com/openai/v1",
    "ollama":    "http://localhost:11434/v1",
    "vllm":      "http://localhost:8000/v1",
}

# provider → environment variable name (fallback when api_key is not configured)
_ENV_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
    "groq":      "GROQ_API_KEY",
    "ollama":    "OLLAMA_API_KEY",   # Ollama local server typically doesn't need one; leave empty
    "vllm":      "VLLM_API_KEY",     # vLLM OpenAI-compat server; leave empty for local
}


def create_backend(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
    max_empty_retries: int | None = None,
) -> LLMBackend:
    """
    Factory function that creates the appropriate LLMBackend for a given provider.

    Args:
        provider:   "anthropic" | "openai" | "deepseek" | "groq" | "ollama" | "vllm"
        model:      model name, e.g. "claude-sonnet-4-5", "gpt-4o", "deepseek-chat"
        api_key:    API key; read from environment variable when None
        base_url:   override the default base_url (usually not needed)
        max_tokens: maximum output token count

    Returns:
        The corresponding LLMBackend instance

    Raises:
        ValueError: unsupported provider, or api_key is missing
    """
    provider = provider.lower().strip()

    if provider not in _PROVIDER_BASE_URLS:
        supported = ", ".join(sorted(_PROVIDER_BASE_URLS))
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported: {supported}"
        )

    # Resolve api_key
    resolved_key = api_key or os.environ.get(_ENV_KEY_MAP.get(provider, ""), "")
    if not resolved_key and provider not in ("ollama", "vllm"):
        env_var = _ENV_KEY_MAP.get(provider, "")
        raise ValueError(
            f"API key for '{provider}' not provided. "
            f"Set it via config or environment variable {env_var!r}."
        )
    # Ollama local server doesn't need a real key; use a placeholder
    if not resolved_key:
        resolved_key = provider  # local servers don't need a real key

    if provider == "anthropic":
        from llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(
            model=model,
            api_key=resolved_key,
            max_tokens=max_tokens,
        )

    # All OpenAI-compatible providers
    from llm.openai_compat import OpenAICompatBackend

    # base_url priority: explicit caller argument > provider default
    resolved_base_url = base_url or _PROVIDER_BASE_URLS[provider]

    # Only pass max_empty_retries when the caller overrode it, so the backend's
    # own default stands otherwise.
    extra = {} if max_empty_retries is None else {"max_empty_retries": max_empty_retries}
    return OpenAICompatBackend(
        model=model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        max_tokens=max_tokens,
        **extra,
    )


def create_backend_from_config(config: dict) -> LLMBackend:
    """
    Create a backend from a config dict, corresponding to the 'llm' section of config/default.yaml.

    Config format:
        provider: anthropic
        model: claude-sonnet-4-5
        api_key: sk-...        # optional; falls back to environment variable
        base_url:              # optional
        max_tokens: 4096       # optional
        max_empty_retries: 6   # optional; resamples of empty reasoning-model turns
    """
    mer = config.get("max_empty_retries")
    return create_backend(
        provider=config.get("provider", "anthropic"),
        model=config.get("model", "claude-sonnet-4-5"),
        api_key=config.get("api_key") or None,
        base_url=config.get("base_url") or None,
        max_tokens=int(config.get("max_tokens", 4096)),
        max_empty_retries=(int(mer) if mer is not None else None),
    )
