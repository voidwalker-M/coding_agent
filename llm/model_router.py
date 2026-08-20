"""
llm/model_router.py

Cost-aware model routing (token/cost-saving feature #4.3).

Most steps an agent takes are mechanical (read a file, run a search, inspect a
git diff) and don't need a frontier model; a few (planning, fixing a failing
test, recovering from an error) do. Routing the easy steps to a cheaper model
and reserving the expensive model for the hard ones cuts cost substantially with
little quality loss.

`RoutingBackend` is a decorator `LLMBackend` wrapping a STRONG and a CHEAP
backend. A pluggable policy inspects the conversation and returns "strong" or
"cheap" per call. The default heuristic escalates to the strong model when:
  - it's the very first call (initial planning), or
  - the latest context shows trouble: a test/tool failure, a traceback, or an
    injected [REFLECTION] prompt.

No agent-core changes: wrap and pass to Agent.

    router = RoutingBackend(strong=create_backend("anthropic","claude-opus-4-8",...),
                            cheap=create_backend("anthropic","claude-haiku-4-5",...))
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from agent.task import ActionType
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)

# policy(messages, tools, call_index) -> "strong" | "cheap"
RoutePolicy = Callable[["list[LLMMessage]", "list[LLMToolSchema]", int], str]

_TROUBLE_MARKERS = (
    "[reflection]", "traceback", "error", "failed", "failure",
    "exception", "assertionerror", "test_", "did not pass",
)


def default_policy(messages: list[LLMMessage], tools: list[LLMToolSchema], call_index: int) -> str:
    """Escalate to the strong model on the first call or when trouble is detected."""
    if call_index == 0:
        return "strong"   # initial planning benefits from the strong model
    # Inspect the most recent non-system message (the latest observation/reflection).
    recent = ""
    for m in reversed(messages):
        if m.role != "system":
            recent = (m.content or "").lower()
            break
    if any(marker in recent for marker in _TROUBLE_MARKERS):
        return "strong"
    return "cheap"


class RoutingBackend(LLMBackend):
    """Route each call to a strong or cheap backend based on a policy."""

    def __init__(
        self,
        strong: LLMBackend,
        cheap: LLMBackend,
        policy: RoutePolicy = default_policy,
    ) -> None:
        self._strong = strong
        self._cheap = cheap
        self._policy = policy
        self._calls = 0
        self.routed = {"strong": 0, "cheap": 0}

    @property
    def model_name(self) -> str:
        return f"router({self._strong.model_name}|{self._cheap.model_name})"

    @property
    def supports_function_calling(self) -> bool:
        # Conservative: only claim FC support if BOTH backends support it, since a
        # call may be routed to either.
        return self._strong.supports_function_calling and self._cheap.supports_function_calling

    def _pick(self, messages, tools) -> LLMBackend:
        choice = self._policy(messages, tools, self._calls)
        self._calls += 1
        self.routed[choice] = self.routed.get(choice, 0) + 1
        logger.debug("model router → %s (call %d)", choice, self._calls)
        return self._strong if choice == "strong" else self._cheap

    def complete(self, messages, tools) -> LLMResponse:
        return self._pick(messages, tools).complete(messages, tools)

    def stream(self, messages, tools, on_text=None, on_thought=None) -> LLMResponse:
        backend = self._pick(messages, tools)
        if hasattr(backend, "stream"):
            return backend.stream(messages, tools, on_text=on_text, on_thought=on_thought)
        return backend.complete(messages, tools)

    def stats(self) -> dict:
        return dict(self.routed)


# ===========================================================================
# Difficulty-aware routing (route BEFORE the call, by estimated difficulty)
# ===========================================================================
#
# The default heuristic above is a coarse difficulty proxy ("first call or
# trouble → strong"). `DifficultyEstimator` makes the notion explicit and
# graded: it maps the current step to a score in [0, 1] from cheap, explainable
# lexical/structural signals (no model call, no network). Two consumers:
#   * difficulty_policy(...)      → a 2-tier RoutePolicy for RoutingBackend
#   * TieredRoutingBackend(...)   → N tiers (cheap / mid / strong / …)
# ---------------------------------------------------------------------------

# Words that signal an inherently hard task/step (raise difficulty).
_HARD_MARKERS: tuple[str, ...] = (
    "refactor", "redesign", "architecture", "concurren", "race condition",
    "deadlock", "root cause", "debug", "root-cause", "investigate", "algorithm",
    "optimize", "performance", "security", "vulnerab", "migrat", "thread",
    "async", "regression", "edge case", "multiple files", "multi-file",
    "across files", "backward compat", "design", "trade-off", "tradeoff",
)
# Words that signal a trivial task/step (lower difficulty).
_EASY_MARKERS: tuple[str, ...] = (
    "typo", "rename", "one line", "one-line", "trivial", "simple", "comment",
    "docstring", "formatting", "lint", "import", "print statement", "readme",
    "whitespace", "small change",
)

# Trouble that appears mid-run (a superset of _TROUBLE_MARKERS, weighted).
_STUCK_MARKERS: tuple[str, ...] = _TROUBLE_MARKERS + (
    "no such file", "not found", "cannot", "unable", "still failing",
    "does not pass", "stack trace",
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass
class DifficultySignal:
    """The graded difficulty of the current step plus the reasons behind it."""
    score: float                       # in [0, 1]; higher = harder
    reasons: list[str] = field(default_factory=list)

    def band(self, thresholds: Sequence[float]) -> int:
        """Return the tier index for ascending `thresholds` (cheap→strong).

        With thresholds=[0.34, 0.67]: score<0.34→0, <0.67→1, else→2.
        """
        idx = 0
        for t in thresholds:
            if self.score >= t:
                idx += 1
            else:
                break
        return idx


class DifficultyEstimator:
    """Score how hard the current step is, from lexical + structural signals.

    Signals (all cheap, all local):
      * the task description (first user message): length + hard/easy keywords;
      * the most recent observation/reflection: trouble markers (tests failing,
        tracebacks, "cannot"…) push difficulty up — a stuck agent needs muscle;
      * depth: long runs without resolution drift upward (escalate when stuck);
      * breadth: many tools available is a weak "complex environment" signal.

    Returns a `DifficultySignal` so callers can log *why* a step was escalated.
    """

    def __init__(
        self,
        *,
        long_task_chars: int = 600,
        depth_ramp_start: int = 8,
        depth_ramp_full: int = 24,
    ) -> None:
        self._long_task = long_task_chars
        self._depth_start = depth_ramp_start
        self._depth_full = depth_ramp_full

    @staticmethod
    def _first_user(messages: list[LLMMessage]) -> str:
        for m in messages:
            if m.role == "user":
                return m.content or ""
        return ""

    @staticmethod
    def _latest_nonsystem(messages: list[LLMMessage]) -> str:
        for m in reversed(messages):
            if m.role != "system":
                return m.content or ""
        return ""

    def estimate(
        self, messages: list[LLMMessage], tools: list[LLMToolSchema], call_index: int
    ) -> DifficultySignal:
        reasons: list[str] = []
        score = 0.30  # neutral prior: default to the middle, let signals move it

        task = self._first_user(messages).lower()
        recent = self._latest_nonsystem(messages).lower()

        hard_hits = [w for w in _HARD_MARKERS if w in task]
        easy_hits = [w for w in _EASY_MARKERS if w in task]
        if hard_hits:
            # ~3 distinct hard signals is enough to reach the top (strong) band.
            score += min(0.45, 0.15 * len(hard_hits))
            reasons.append(f"hard task keywords: {', '.join(sorted(set(hard_hits))[:3])}")
        if easy_hits:
            score -= min(0.25, 0.12 * len(easy_hits))
            reasons.append(f"easy task keywords: {', '.join(sorted(set(easy_hits))[:3])}")

        # Long, detailed tasks tend to be harder.
        if len(task) >= self._long_task:
            score += 0.10
            reasons.append("long task description")

        # Mid-run trouble → escalate (only when it's actually recent context,
        # i.e. not the very first planning call which has no observation yet).
        if call_index > 0 and any(w in recent for w in _STUCK_MARKERS):
            score += 0.30
            reasons.append("recent trouble (test failure / traceback / stuck)")

        # Depth ramp: a long-running, unresolved step is probably a hard one.
        if call_index >= self._depth_start:
            span = max(1, self._depth_full - self._depth_start)
            ramp = min(1.0, (call_index - self._depth_start) / span)
            score += 0.15 * ramp
            reasons.append(f"deep into run (step {call_index})")

        # Weak breadth signal.
        if len(tools) >= 10:
            score += 0.03

        score = _clamp(score)
        return DifficultySignal(score=score, reasons=reasons)


def difficulty_policy(
    estimator: DifficultyEstimator | None = None,
    *,
    threshold: float = 0.55,
) -> RoutePolicy:
    """A 2-tier RoutePolicy for RoutingBackend based on estimated difficulty.

    Escalates to "strong" when the estimated difficulty ≥ threshold, else "cheap".
    Drop-in replacement for `default_policy`:

        RoutingBackend(strong, cheap, policy=difficulty_policy(threshold=0.5))
    """
    est = estimator or DifficultyEstimator()

    def policy(messages: list[LLMMessage], tools: list[LLMToolSchema], call_index: int) -> str:
        sig = est.estimate(messages, tools, call_index)
        choice = "strong" if sig.score >= threshold else "cheap"
        if choice == "strong" and sig.reasons:
            logger.debug("difficulty=%0.2f → strong (%s)", sig.score, "; ".join(sig.reasons))
        return choice

    return policy


# tier_policy(messages, tools, call_index) -> tier name
TierPolicy = Callable[["list[LLMMessage]", "list[LLMToolSchema]", int], str]


def difficulty_tier_policy(
    tier_names: Sequence[str],
    estimator: DifficultyEstimator | None = None,
    thresholds: Sequence[float] | None = None,
) -> TierPolicy:
    """Map estimated difficulty to one of N ordered tier names (cheap→strong).

    `thresholds` has len(tier_names)-1 ascending cut points; when omitted the
    band edges are spread evenly across [0, 1].
    """
    est = estimator or DifficultyEstimator()
    names = list(tier_names)
    if thresholds is None:
        n = len(names)
        thresholds = [i / n for i in range(1, n)]
    thr = list(thresholds)

    def policy(messages, tools, call_index) -> str:
        sig = est.estimate(messages, tools, call_index)
        idx = min(sig.band(thr), len(names) - 1)
        return names[idx]

    return policy


class TieredRoutingBackend(LLMBackend):
    """Route each call to one of N ordered tiers by a pluggable tier policy.

    `tiers` is an ordered list of (name, backend) from cheapest to strongest.
    The generalization of RoutingBackend to more than two models — the
    "multi-router" proper. Defaults to difficulty-based tiering.
    """

    def __init__(
        self,
        tiers: list[tuple[str, LLMBackend]],
        policy: TierPolicy | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("TieredRoutingBackend requires at least one tier")
        self._tiers = tiers
        self._by_name = {name: backend for name, backend in tiers}
        self._policy = policy or difficulty_tier_policy([n for n, _ in tiers])
        self._calls = 0
        self.routed = {name: 0 for name, _ in tiers}

    @property
    def model_name(self) -> str:
        return "tiered(" + "|".join(n for n, _ in self._tiers) + ")"

    @property
    def supports_function_calling(self) -> bool:
        return all(b.supports_function_calling for _, b in self._tiers)

    def _pick(self, messages, tools) -> LLMBackend:
        name = self._policy(messages, tools, self._calls)
        self._calls += 1
        if name not in self._by_name:  # policy returned an unknown tier → strongest
            name = self._tiers[-1][0]
        self.routed[name] = self.routed.get(name, 0) + 1
        logger.debug("tiered router → %s (call %d)", name, self._calls)
        return self._by_name[name]

    def complete(self, messages, tools) -> LLMResponse:
        return self._pick(messages, tools).complete(messages, tools)

    def stream(self, messages, tools, on_text=None, on_thought=None) -> LLMResponse:
        backend = self._pick(messages, tools)
        if hasattr(backend, "stream"):
            return backend.stream(messages, tools, on_text=on_text, on_thought=on_thought)
        return backend.complete(messages, tools)

    def stats(self) -> dict:
        return dict(self.routed)


# ===========================================================================
# Confidence / uncertainty routing (cascade: escalate AFTER a cheap attempt)
# ===========================================================================
#
# Difficulty routing decides up front. The other axis the user asked for is the
# *model's own confidence*: run the cheap model first, and only pay for a
# stronger model when the cheap answer looks uncertain. This is the FrugalGPT
# "LLM cascade" — often the biggest real cost saver, because most steps are easy
# and the cheap tier is accepted outright.
# ---------------------------------------------------------------------------

# Hedging phrases that betray low confidence in free text.
_HEDGE_MARKERS: tuple[str, ...] = (
    "not sure", "unsure", "i think", "i believe", "maybe", "perhaps",
    "possibly", "might be", "could be", "unclear", "don't know", "dont know",
    "hard to say", "cannot tell", "not certain", "guess", "i'm not confident",
    "difficult to", "it seems",
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def estimate_confidence(response: LLMResponse) -> float:
    """Best-effort confidence in [0, 1] for a response.

    Priority: an explicit `response.confidence` (self-report / provider) wins;
    otherwise derive a heuristic from the parsed Action and any logprob. The
    heuristic is deliberately conservative — it exists to catch *clearly*
    uncertain steps (give-ups, unparsed tool args, hedged prose), not to be a
    calibrated probability.
    """
    if response.confidence is not None:
        return _clamp(float(response.confidence))

    action = response.action
    text = f"{getattr(action, 'thought', '') or ''} {getattr(action, 'message', '') or ''}"
    tl = text.lower()

    atype = action.action_type
    if atype == ActionType.GIVE_UP:
        base = 0.15
    elif atype == ActionType.TOOL_CALL:
        tc = action.tool_call
        params = (tc.params if tc else None) or {}
        # A "raw" key is our marker for tool-args that failed to parse as JSON.
        base = 0.35 if "raw" in params else 0.78
    elif atype == ActionType.FINISH:
        base = 0.70
    else:
        base = 0.50

    if any(h in tl for h in _HEDGE_MARKERS):
        base -= 0.25
    if tl.strip() in ("", "(no thought)"):
        base -= 0.10

    # Fold in a logprob-derived probability when the provider supplied one.
    if response.logprob_avg is not None:
        try:
            p = math.exp(response.logprob_avg)          # mean token prob in (0,1]
            base = 0.5 * base + 0.5 * _clamp(p)
        except (OverflowError, ValueError):
            pass

    return _clamp(base)


# confidence_fn(response) -> float in [0, 1]
ConfidenceFn = Callable[[LLMResponse], float]


class CascadingBackend(LLMBackend):
    """Confidence/uncertainty cascade over stages ordered cheap→strong.

    On each call, run the cheapest stage; if its estimated confidence is below
    `min_confidence`, escalate to the next stage, and so on. The last stage is
    always accepted. The returned response carries the accepted action but the
    *summed* token usage of every attempt made — cascades cost more on the hard
    steps they escalate, and the accounting must not hide that.
    """

    def __init__(
        self,
        stages: list[LLMBackend],
        *,
        min_confidence: float = 0.35,
        confidence_fn: ConfidenceFn = estimate_confidence,
    ) -> None:
        if not stages:
            raise ValueError("CascadingBackend requires at least one stage")
        self._stages = stages
        self._min_conf = min_confidence
        self._conf = confidence_fn
        self.accepted = [0] * len(stages)   # per-stage acceptance histogram
        self.escalations = 0

    @property
    def model_name(self) -> str:
        return "cascade(" + "→".join(b.model_name for b in self._stages) + ")"

    @property
    def supports_function_calling(self) -> bool:
        return all(b.supports_function_calling for b in self._stages)

    def _run(self, messages, tools, streaming, on_text, on_thought) -> LLMResponse:
        spent_in = spent_out = 0
        last: LLMResponse | None = None
        for i, backend in enumerate(self._stages):
            is_last = i == len(self._stages) - 1
            # Only stream the accepted stage; earlier probes run non-streaming
            # because we need the whole response to judge confidence first.
            if streaming and is_last and hasattr(backend, "stream"):
                resp = backend.stream(messages, tools, on_text=on_text, on_thought=on_thought)
            else:
                resp = backend.complete(messages, tools)
            spent_in += resp.input_tokens
            spent_out += resp.output_tokens
            last = resp
            conf = self._conf(resp)
            if is_last or conf >= self._min_conf:
                self.accepted[i] += 1
                # Count escalation *hops*, not "calls that escalated": accepting at
                # stage i means i extra model calls were paid for. (accepted_per_stage
                # already reports where calls landed.)
                self.escalations += i
                logger.debug(
                    "cascade accepted stage %d/%d (conf=%.2f, min=%.2f)",
                    i + 1, len(self._stages), conf, self._min_conf,
                )
                # Emit text for a non-streamed accepted stage so the UI still updates.
                if streaming and not (is_last and hasattr(backend, "stream")) and on_text and resp.raw_content:
                    on_text(resp.raw_content)
                break
            logger.debug(
                "cascade escalating past stage %d (conf=%.2f < %.2f)",
                i + 1, conf, self._min_conf,
            )

        assert last is not None
        # Rebuild with summed usage + the accepted confidence.
        return LLMResponse(
            action=last.action,
            raw_content=last.raw_content,
            input_tokens=spent_in,
            output_tokens=spent_out,
            confidence=self._conf(last),
            logprob_avg=last.logprob_avg,
        )

    def complete(self, messages, tools) -> LLMResponse:
        return self._run(messages, tools, streaming=False, on_text=None, on_thought=None)

    def stream(self, messages, tools, on_text=None, on_thought=None) -> LLMResponse:
        return self._run(messages, tools, streaming=True, on_text=on_text, on_thought=on_thought)

    def stats(self) -> dict:
        """accepted_per_stage: calls accepted at each stage.
        escalations: total escalation *hops* (== extra model calls paid for)."""
        return {
            "accepted_per_stage": list(self.accepted),
            "escalations": self.escalations,
            "stages": len(self._stages),
        }
