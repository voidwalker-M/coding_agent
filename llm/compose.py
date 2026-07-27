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
from llm.model_router import (
    CascadingBackend,
    RoutePolicy,
    RoutingBackend,
    TieredRoutingBackend,
    default_policy,
    difficulty_policy,
    difficulty_tier_policy,
)
from llm.rate_limit import RateLimiter, RateLimitedBackend


def _build_router(
    base: LLMBackend,
    cheap: LLMBackend | None,
    mid: LLMBackend | None,
    route_mode: str,
    route_policy: RoutePolicy,
    min_confidence: float,
) -> LLMBackend:
    """Pick the routing layer for the given mode. `base` is the strongest tier.

    Tiers, cheapest→strongest, are the non-None subset of (cheap, mid, base).
    """
    tiers = [(name, b) for name, b in (("cheap", cheap), ("mid", mid), ("strong", base)) if b is not None]
    if len(tiers) == 1:            # only `base` present → nothing to route
        return base

    mode = (route_mode or "heuristic").lower()
    # The cheapest tier for the 2-tier routers — derived from the tier list, so it
    # is correct whether it came from `cheap` or `mid` (mid-only is a valid setup).
    cheapest = tiers[0][1]

    if mode == "cascade":
        # Confidence/uncertainty cascade over the tier backends (cheap→strong).
        return CascadingBackend([b for _, b in tiers], min_confidence=min_confidence)

    if mode == "difficulty":
        if len(tiers) > 2:
            return TieredRoutingBackend(
                tiers, policy=difficulty_tier_policy([n for n, _ in tiers])
            )
        return RoutingBackend(strong=base, cheap=cheapest, policy=difficulty_policy())

    # "heuristic" (default): the original 2-tier keyword policy. If a mid tier
    # was supplied without an explicit mode, fold it into difficulty tiering so
    # the extra model is actually used.
    if len(tiers) > 2:
        return TieredRoutingBackend(tiers, policy=difficulty_tier_policy([n for n, _ in tiers]))
    return RoutingBackend(strong=base, cheap=cheapest, policy=route_policy)


def compose_backend(
    base: LLMBackend,
    *,
    cheap: LLMBackend | None = None,
    mid: LLMBackend | None = None,
    route_mode: str = "heuristic",
    route_policy: RoutePolicy = default_policy,
    min_confidence: float = 0.35,
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
        base:           the strong/primary backend (the strongest tier).
        cheap:          cheapest tier; if given (or `mid`), enables routing.
        mid:            optional middle tier for 3-tier "multi-router" setups.
        route_mode:     "heuristic" (keyword policy), "difficulty" (route up front
                        by estimated difficulty), or "cascade" (run cheap first and
                        escalate on low confidence/uncertainty).
        route_policy:   policy used by the heuristic 2-tier router.
        min_confidence: confidence floor below which the cascade escalates.
        cache:          enable response caching (exact; semantic if embed_fn given).
        embed_fn:       embedding fn enabling the semantic cache layer.
        rpm/tpm/max_usd: rate-limit dimensions (None = unlimited).
        model_for_cost: model name used for $ accounting in the limiter.
    """
    backend = base
    if cheap is not None or mid is not None:
        backend = _build_router(base, cheap, mid, route_mode, route_policy, min_confidence)
    if cache:
        backend = CachingBackend(backend, embed_fn=embed_fn)
    if rpm is not None or tpm is not None or max_usd is not None:
        limiter = RateLimiter(rpm=rpm, tpm=tpm, max_usd=max_usd, model=model_for_cost)
        backend = RateLimitedBackend(backend, limiter)
    return backend
