"""
llm/compose.py

Convenience composition of the token-efficiency decorator backends (#4).

The decorators stack in a deliberate order (outermost first):

    RateLimitedBackend            # spend ceiling / throttle — outermost, sees every real call
      └─ CachingBackend           # serve repeats for free (a cache hit costs nothing,
                                  #   and correctly bypasses the limiter's token accounting)
           └─ RoutingBackend      # pick cheap vs strong for genuine calls
                └─ base backend(s)

Wrapping order matters: cache *inside* the limiter means a cache hit doesn't
consume the rate/$ budget; routing *inside* the cache means we only route on a
genuine miss.
"""

from __future__ import annotations

from llm.base import LLMBackend
from llm.cache import CachingBackend, EmbedFn
from llm.model_router import RoutePolicy, RoutingBackend, default_policy
from llm.rate_limit import RateLimiter, RateLimitedBackend


def compose_backend(
    base: LLMBackend,
    *,
    cheap: LLMBackend | None = None,
    route_policy: RoutePolicy = default_policy,
    cache: bool = False,
    embed_fn: EmbedFn | None = None,
    rpm: int | None = None,
    tpm: int | None = None,
    max_usd: float | None = None,
    model_for_cost: str | None = None,
) -> LLMBackend:
    """
    Wrap `base` with the requested token-efficiency layers. Each layer is opt-in;
    with all options off this returns `base` unchanged.

    Args:
        base:           the strong/primary backend.
        cheap:          if given, enables cost-aware routing (cheap vs base).
        route_policy:   routing policy (default: escalate on first call / trouble).
        cache:          enable response caching (exact; semantic if embed_fn given).
        embed_fn:       embedding fn enabling the semantic cache layer.
        rpm/tpm/max_usd: rate-limit dimensions (None = unlimited).
        model_for_cost: model name used for $ accounting in the limiter.
    """
    backend = base
    if cheap is not None:
        backend = RoutingBackend(strong=base, cheap=cheap, policy=route_policy)
    if cache:
        backend = CachingBackend(backend, embed_fn=embed_fn)
    if rpm is not None or tpm is not None or max_usd is not None:
        limiter = RateLimiter(rpm=rpm, tpm=tpm, max_usd=max_usd, model=model_for_cost)
        backend = RateLimitedBackend(backend, limiter)
    return backend
