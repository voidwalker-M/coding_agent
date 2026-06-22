"""
tests/test_token_efficiency.py

Tests for the token-saving decorator backends (#4): response cache, cost-aware
model router, and multi-dimensional rate / $ limiting. All use MockBackend, so
no API is consumed.
"""

import pytest

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMBackend, LLMMessage, LLMResponse, MockBackend
from llm.cache import CachingBackend
from llm.compose import compose_backend
from llm.model_router import RoutingBackend, default_policy
from llm.rate_limit import BudgetExceeded, RateLimiter, RateLimitedBackend


def _msgs(*texts):
    return [LLMMessage(role="user", content=t) for t in texts]


def _tool_action(name="shell"):
    return Action(ActionType.TOOL_CALL, "do it", tool_call=ToolCall(name, {"cmd": "ls"}))


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_cache_serves_repeat_for_free():
    inner = MockBackend([_tool_action(), _tool_action()], input_tokens=100, output_tokens=50)
    cached = CachingBackend(inner)

    r1 = cached.complete(_msgs("same prompt"), [])
    assert r1.total_tokens == 150          # miss → real call
    assert inner.call_count == 1

    r2 = cached.complete(_msgs("same prompt"), [])
    assert r2.total_tokens == 0            # hit → free
    assert inner.call_count == 1           # inner NOT called again
    assert r2.action.tool_call.name == "shell"
    assert cached.stats()["hits"] == 1


def test_cache_miss_on_different_prompt():
    inner = MockBackend([_tool_action(), _tool_action()])
    cached = CachingBackend(inner)
    cached.complete(_msgs("prompt A"), [])
    cached.complete(_msgs("prompt B"), [])
    assert inner.call_count == 2
    assert cached.stats()["misses"] == 2


def test_semantic_cache_hit_with_embed_fn():
    # Embedding fn: two near-identical prompts map to near-identical vectors.
    def embed(text):
        return [1.0, 0.0] if "budget" in text else [0.0, 1.0]

    inner = MockBackend([_tool_action(), _tool_action()])
    cached = CachingBackend(inner, embed_fn=embed, similarity_threshold=0.95)
    cached.complete(_msgs("fix the budget code"), [])      # miss, stored
    cached.complete(_msgs("please fix budget logic"), [])  # semantic hit
    assert inner.call_count == 1
    assert cached.stats()["hits"] == 1


# ---------------------------------------------------------------------------
# Model router
# ---------------------------------------------------------------------------

def test_default_policy_first_call_is_strong():
    assert default_policy(_msgs("anything"), [], call_index=0) == "strong"


def test_default_policy_escalates_on_trouble():
    assert default_policy(_msgs("AssertionError: traceback ..."), [], call_index=3) == "strong"
    assert default_policy(_msgs("file contents look fine"), [], call_index=3) == "cheap"


def test_router_sends_first_call_to_strong_then_cheap():
    strong = MockBackend([_tool_action("strong_tool")] * 5)
    cheap = MockBackend([_tool_action("cheap_tool")] * 5)
    router = RoutingBackend(strong, cheap)

    r0 = router.complete(_msgs("plan the work"), [])      # call 0 → strong
    assert r0.action.tool_call.name == "strong_tool"
    r1 = router.complete(_msgs("ordinary observation"), [])  # call 1 → cheap
    assert r1.action.tool_call.name == "cheap_tool"
    r2 = router.complete(_msgs("test_x failed: traceback"), [])  # trouble → strong
    assert r2.action.tool_call.name == "strong_tool"
    assert router.stats() == {"strong": 2, "cheap": 1}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_budget_ceiling_raises():
    # claude-sonnet blended = 9.0 $/Mtok → 1M tokens = $9, ceiling $0.001 trips fast.
    inner = MockBackend([_tool_action()] * 5, input_tokens=100_000, output_tokens=100_000)
    limiter = RateLimiter(max_usd=0.001, model="claude-sonnet-4-6")
    limited = RateLimitedBackend(inner, limiter)

    limited.complete(_msgs("first"), [])             # first call allowed; spends > ceiling
    with pytest.raises(BudgetExceeded):
        limited.complete(_msgs("second"), [])        # now over ceiling → raise


def test_rpm_throttle_sleeps_when_window_full():
    slept = []
    t = {"now": 0.0}
    limiter = RateLimiter(
        rpm=2,
        sleep_fn=lambda s: (slept.append(s), t.__setitem__("now", t["now"] + s)),
        clock=lambda: t["now"],
    )
    inner = MockBackend([_tool_action()] * 5, input_tokens=10, output_tokens=10)
    limited = RateLimitedBackend(inner, limiter)

    limited.complete(_msgs("1"), [])
    limited.complete(_msgs("2"), [])
    assert slept == []                  # first two within rpm=2, no sleep
    limited.complete(_msgs("3"), [])    # third must wait for the window
    assert len(slept) >= 1
    assert limiter.stats()["waits"] >= 1


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def test_compose_no_options_returns_base():
    base = MockBackend([_tool_action()])
    assert compose_backend(base) is base


def test_compose_stacks_layers():
    base = MockBackend([_tool_action()] * 5)
    cheap = MockBackend([_tool_action("cheap")] * 5)
    composed = compose_backend(base, cheap=cheap, cache=True, max_usd=100.0,
                               model_for_cost="claude-haiku-4-5")
    # Outermost is the rate limiter; a normal call still flows through to an action.
    resp = composed.complete(_msgs("hello"), [])
    assert resp.action.action_type == ActionType.TOOL_CALL
