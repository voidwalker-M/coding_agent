"""
eval/pricing.py

Rough USD cost estimation for an eval run.

The agent only tracks a single combined `total_tokens` figure per run (it does not
split prompt vs completion tokens), so this uses a *blended* $/1M-token rate per
model family — a deliberate approximation, not a billing-accurate number. Rates
move frequently; edit the table below to match current provider pricing.

Usage:
    from eval.pricing import estimate_cost
    usd = estimate_cost("claude-sonnet-4-6", total_tokens=12_000)
"""

from __future__ import annotations

# Blended (input+output averaged) USD per 1,000,000 tokens. Approximate.
# Keys are matched as case-insensitive substrings of the model name.
_BLENDED_USD_PER_MTOK: dict[str, float] = {
    "claude-opus":   30.0,
    "claude-sonnet":  9.0,
    "claude-haiku":   3.0,
    "gpt-4o":         7.5,
    "gpt-4":         45.0,
    "o1":            30.0,
    "deepseek":       0.6,
    "groq":           0.5,
    "ollama":         0.0,   # local
    "qwen":           0.5,
}

# Used when no model-family key matches.
DEFAULT_BLENDED_USD_PER_MTOK = 10.0


def blended_rate(model: str | None) -> float:
    """Return the blended $/1M-token rate for a model name (substring match)."""
    m = (model or "").lower()
    for key, rate in _BLENDED_USD_PER_MTOK.items():
        if key in m:
            return rate
    return DEFAULT_BLENDED_USD_PER_MTOK


def estimate_cost(model: str | None, total_tokens: int) -> float:
    """Estimate USD cost for a run. Returns 0.0 when tokens are unknown/zero."""
    if not total_tokens or total_tokens <= 0:
        return 0.0
    return (total_tokens / 1_000_000.0) * blended_rate(model)
