"""
tests/test_model_router.py

Tests for the multi-router LLM (feature #1): difficulty-aware up-front routing
and confidence/uncertainty cascade routing. All use MockBackend, so no API is
consumed.
"""

import pytest

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMMessage, LLMResponse, MockBackend
from llm.compose import compose_backend
from llm.model_router import (
    CascadingBackend,
    DifficultyEstimator,
    TieredRoutingBackend,
    default_policy,
    difficulty_policy,
    difficulty_tier_policy,
    estimate_confidence,
)


def _msgs(*texts, roles=None):
    roles = roles or ["user"] * len(texts)
    return [LLMMessage(role=r, content=t) for r, t in zip(roles, texts)]


def _tool_action(name="shell", thought="do it", params=None):
    return Action(ActionType.TOOL_CALL, thought, tool_call=ToolCall(name, params or {"cmd": "ls"}))


def _give_up(msg="cannot solve"):
    return Action(ActionType.GIVE_UP, thought=msg, message=msg)


def _finish(msg="done"):
    return Action(ActionType.FINISH, thought="", message=msg)


# ---------------------------------------------------------------------------
# Difficulty estimator
# ---------------------------------------------------------------------------

def test_difficulty_hard_task_scores_high():
    est = DifficultyEstimator()
    hard = _msgs("Refactor the concurrency model and fix the race condition in the scheduler")
    sig = est.estimate(hard, [], call_index=0)
    assert sig.score >= 0.5
    assert sig.reasons  # explains itself


def test_difficulty_easy_task_scores_low():
    est = DifficultyEstimator()
    easy = _msgs("Fix a typo in the README comment")
    sig = est.estimate(easy, [], call_index=0)
    assert sig.score < 0.4


def test_difficulty_escalates_on_recent_trouble():
    est = DifficultyEstimator()
    msgs = _msgs(
        "add a helper function",
        "[Tool: pytest | ERROR]\nAssertionError: traceback ... test_x failed",
        roles=["user", "user"],
    )
    low = est.estimate(_msgs("add a helper function"), [], call_index=3).score
    high = est.estimate(msgs, [], call_index=3).score
    assert high > low
    assert high >= 0.5


def test_difficulty_band_mapping():
    est = DifficultyEstimator()
    sig = est.estimate(_msgs("simple typo rename"), [], call_index=0)
    # thresholds [0.34, 0.67] -> a low score lands in band 0
    assert sig.band([0.34, 0.67]) == 0


# ---------------------------------------------------------------------------
# Difficulty policy (2-tier) and tier policy (N-tier)
# ---------------------------------------------------------------------------

def test_difficulty_policy_routes_by_threshold():
    policy = difficulty_policy(threshold=0.55)
    assert policy(_msgs("Redesign the architecture and debug the deadlock"), [], 0) == "strong"
    assert policy(_msgs("fix a small typo"), [], 0) == "cheap"


def test_difficulty_tier_policy_three_tiers():
    policy = difficulty_tier_policy(["cheap", "mid", "strong"])
    assert policy(_msgs("fix a trivial typo"), [], 0) == "cheap"
    hard = policy(_msgs("Redesign the concurrency architecture; investigate the race condition root cause"), [], 0)
    assert hard == "strong"


def test_tiered_routing_backend_dispatches_and_counts():
    cheap = MockBackend([_tool_action("cheap_tool")] * 5)
    mid = MockBackend([_tool_action("mid_tool")] * 5)
    strong = MockBackend([_tool_action("strong_tool")] * 5)
    router = TieredRoutingBackend(
        [("cheap", cheap), ("mid", mid), ("strong", strong)],
        policy=difficulty_tier_policy(["cheap", "mid", "strong"]),
    )
    r_easy = router.complete(_msgs("fix a trivial one-line typo"), [])
    assert r_easy.action.tool_call.name == "cheap_tool"
    r_hard = router.complete(
        _msgs("Redesign the architecture, debug the concurrency deadlock, investigate root cause"), [])
    assert r_hard.action.tool_call.name == "strong_tool"
    assert router.routed["cheap"] == 1 and router.routed["strong"] == 1


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

def test_confidence_explicit_value_wins():
    resp = LLMResponse(action=_tool_action(), raw_content="", confidence=0.9)
    assert estimate_confidence(resp) == pytest.approx(0.9)


def test_confidence_give_up_is_low_tool_call_is_high():
    assert estimate_confidence(LLMResponse(_give_up(), "")) < 0.3
    assert estimate_confidence(LLMResponse(_tool_action(), "")) > 0.6


def test_confidence_hedging_lowers_score():
    plain = LLMResponse(_finish("The fix is applied and tests pass"), "")
    hedged = LLMResponse(_finish("I'm not sure but maybe this could be the fix"), "")
    assert estimate_confidence(hedged) < estimate_confidence(plain)


def test_confidence_unparsed_tool_args_are_low():
    bad = LLMResponse(_tool_action(params={"raw": "{not json"}), "")
    assert estimate_confidence(bad) < 0.5


def test_confidence_folds_in_logprob():
    import math
    high_lp = LLMResponse(_finish("done"), "", logprob_avg=math.log(0.99))
    low_lp = LLMResponse(_finish("done"), "", logprob_avg=math.log(0.10))
    assert estimate_confidence(high_lp) > estimate_confidence(low_lp)


# ---------------------------------------------------------------------------
# Cascade (confidence / uncertainty escalation)
# ---------------------------------------------------------------------------

def test_cascade_accepts_confident_cheap_answer():
    cheap = MockBackend([_tool_action("cheap_tool")], input_tokens=10, output_tokens=5, confidence=0.9)
    strong = MockBackend([_tool_action("strong_tool")], input_tokens=100, output_tokens=50)
    cascade = CascadingBackend([cheap, strong], min_confidence=0.5)

    r = cascade.complete(_msgs("easy step"), [])
    assert r.action.tool_call.name == "cheap_tool"   # accepted cheap
    assert strong.call_count == 0                     # strong never touched
    assert r.total_tokens == 15                        # only cheap's tokens
    assert cascade.stats()["accepted_per_stage"] == [1, 0]


def test_cascade_escalates_on_low_confidence():
    cheap = MockBackend([_tool_action("cheap_tool")], input_tokens=10, output_tokens=5, confidence=0.2)
    strong = MockBackend([_tool_action("strong_tool")], input_tokens=100, output_tokens=50, confidence=0.95)
    cascade = CascadingBackend([cheap, strong], min_confidence=0.5)

    r = cascade.complete(_msgs("hard step"), [])
    assert r.action.tool_call.name == "strong_tool"  # escalated
    assert cheap.call_count == 1 and strong.call_count == 1
    # honest accounting: both attempts' tokens are summed
    assert r.total_tokens == 15 + 150
    assert cascade.stats()["escalations"] == 1
    assert cascade.stats()["accepted_per_stage"] == [0, 1]


def test_cascade_counts_multi_hop_escalations():
    # 3 stages, first two unconfident → 2 escalation hops (2 extra calls paid for).
    s0 = MockBackend([_tool_action("s0")], input_tokens=10, output_tokens=5, confidence=0.1)
    s1 = MockBackend([_tool_action("s1")], input_tokens=20, output_tokens=10, confidence=0.2)
    s2 = MockBackend([_tool_action("s2")], input_tokens=100, output_tokens=50, confidence=0.9)
    cascade = CascadingBackend([s0, s1, s2], min_confidence=0.5)

    r = cascade.complete(_msgs("hard"), [])
    assert r.action.tool_call.name == "s2"
    assert cascade.stats()["escalations"] == 2          # hops, not "1 call escalated"
    assert cascade.stats()["accepted_per_stage"] == [0, 0, 1]
    assert r.total_tokens == 15 + 30 + 150             # every attempt is billed


def test_cascade_last_stage_always_accepted():
    # Even a low-confidence final stage is accepted (no stronger tier left).
    cheap = MockBackend([_give_up()], input_tokens=10, output_tokens=5)
    strong = MockBackend([_give_up()], input_tokens=100, output_tokens=50)
    cascade = CascadingBackend([cheap, strong], min_confidence=0.9)
    r = cascade.complete(_msgs("impossible"), [])
    assert r.action.action_type == ActionType.GIVE_UP
    assert cheap.call_count == 1 and strong.call_count == 1


# ---------------------------------------------------------------------------
# compose_backend wiring for the new modes
# ---------------------------------------------------------------------------

def test_compose_difficulty_mode_routes_hard_to_strong():
    strong = MockBackend([_tool_action("strong_tool")] * 3)
    cheap = MockBackend([_tool_action("cheap_tool")] * 3)
    composed = compose_backend(strong, cheap=cheap, route_mode="difficulty")
    r = composed.complete(_msgs("Refactor the architecture and debug the deadlock root cause"), [])
    assert r.action.tool_call.name == "strong_tool"


def test_compose_cascade_mode_escalates():
    strong = MockBackend([_tool_action("strong_tool")], confidence=0.95)
    cheap = MockBackend([_tool_action("cheap_tool")], confidence=0.1)
    composed = compose_backend(strong, cheap=cheap, route_mode="cascade", min_confidence=0.5)
    r = composed.complete(_msgs("anything"), [])
    assert r.action.tool_call.name == "strong_tool"


def test_compose_three_tier_difficulty():
    strong = MockBackend([_tool_action("strong_tool")] * 3)
    mid = MockBackend([_tool_action("mid_tool")] * 3)
    cheap = MockBackend([_tool_action("cheap_tool")] * 3)
    composed = compose_backend(strong, cheap=cheap, mid=mid, route_mode="difficulty")
    assert isinstance(composed, TieredRoutingBackend)
    r = composed.complete(_msgs("fix a trivial typo"), [])
    assert r.action.tool_call.name == "cheap_tool"


def test_compose_mid_only_does_not_pass_none_cheap():
    # Regression: mid set, cheap absent — the 2-tier router must use `mid` as the
    # cheaper tier, never None.
    strong = MockBackend([_tool_action("strong_tool")] * 3)
    mid = MockBackend([_tool_action("mid_tool")] * 3)
    composed = compose_backend(strong, mid=mid, route_mode="difficulty")
    r = composed.complete(_msgs("fix a trivial typo"), [])
    assert r.action.tool_call.name == "mid_tool"      # routed to the cheaper tier, no crash


def test_existing_heuristic_default_unchanged():
    # Regression: no route_mode given still yields the original heuristic policy.
    strong = MockBackend([_tool_action("strong_tool")] * 3)
    cheap = MockBackend([_tool_action("cheap_tool")] * 3)
    composed = compose_backend(strong, cheap=cheap)
    r0 = composed.complete(_msgs("plan the work"), [])
    assert r0.action.tool_call.name == "strong_tool"   # first call → strong (default_policy)
