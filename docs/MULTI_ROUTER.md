# Multi-Router LLM — difficulty- and confidence-aware model selection (#1)

**Goal.** Automatically decide *which* model handles each step, so cheap models do
the easy work and the expensive model is spent only where it earns its cost. Two
independent axes are supported, matching the two natural notions of "which model":

1. **Difficulty** (decide *before* the call) — estimate how hard the step is and
   route up front.
2. **Confidence / uncertainty** (decide *after* a cheap call) — run the cheap model
   first, measure how sure it is, and escalate only when it looks uncertain.

Everything is built as **decorator `LLMBackend`s** (like the existing token-efficiency
layers), so the ReAct core never changes — you wrap the backend at construction.

Files: [llm/model_router.py](../llm/model_router.py), [llm/compose.py](../llm/compose.py),
[llm/base.py](../llm/base.py), [llm/openai_compat.py](../llm/openai_compat.py).
Tests: [tests/test_model_router.py](../tests/test_model_router.py) (19).

---

## Backends & how they route

| Backend | Axis | When it decides | Tiers |
|---|---|---|---|
| `RoutingBackend` (existing) | heuristic keyword | before call | 2 (strong/cheap) |
| `RoutingBackend` + `difficulty_policy` | **difficulty** | before call | 2 |
| `TieredRoutingBackend` | **difficulty** | before call | **N** (cheap/mid/strong/…) |
| `CascadingBackend` | **confidence / uncertainty** | after cheap call | N (cheap→strong) |

All are `LLMBackend`s with a `.stats()` method for observability.

### Difficulty estimation (up-front routing)

`DifficultyEstimator.estimate(messages, tools, call_index) -> DifficultySignal` maps a
step to a score in `[0, 1]` plus the **reasons** behind it (so an escalation is
explainable in the logs). It is intentionally cheap — pure lexical/structural
signals, no model call, no network:

- **Task keywords** — hard markers (`refactor`, `concurrency`, `race condition`,
  `architecture`, `debug`, `security`, `migrate`, …) push the score up; easy markers
  (`typo`, `rename`, `one-line`, `docstring`, …) push it down. ~3 distinct hard
  signals is enough to reach the top band.
- **Recent trouble** — a test failure / traceback / "cannot" in the latest
  observation escalates a mid-run step (a stuck agent needs muscle).
- **Depth ramp** — a long-running, still-unresolved step drifts upward.
- **Task length + tool breadth** — weak secondary signals.

`DifficultySignal.band(thresholds)` turns the score into a tier index. `difficulty_policy(threshold=…)`
wraps it as a 2-tier `RoutePolicy`; `difficulty_tier_policy(names, thresholds=…)` maps
it to N ordered tiers (default band edges spread evenly across `[0,1]`).

```python
# 3-tier "multi-router" by estimated difficulty
router = TieredRoutingBackend(
    [("cheap", haiku), ("mid", sonnet), ("strong", opus)],
    policy=difficulty_tier_policy(["cheap", "mid", "strong"]),
)
```

### Confidence / uncertainty cascade (escalate on doubt)

`CascadingBackend(stages, min_confidence, confidence_fn)` runs the cheapest stage,
estimates its confidence, and escalates to the next stage only when confidence is
below `min_confidence`; the last stage is always accepted. This is the FrugalGPT
"LLM cascade" — usually the biggest real saver, because most steps are easy and the
cheap tier is accepted outright.

Honest accounting: the returned `LLMResponse` carries the **accepted action** but the
**summed token usage of every attempt made**, so escalations are never hidden from
the cost meter. Streaming only streams the *accepted* stage (earlier probes must be
read whole to judge confidence first).

**Confidence signal** — `estimate_confidence(response) -> float`:
1. an explicit `response.confidence` (a self-report or provider value) wins;
2. otherwise a heuristic from the parsed `Action`: `GIVE_UP`→low, a clean
   `TOOL_CALL`→high, unparsed tool args (`{"raw": …}`)→low, hedging prose
   (`"not sure"`, `"maybe"`, `"I think"`)→lowered;
3. if the provider returned token logprobs, `exp(mean_logprob)` is blended in.

`OpenAICompatBackend(request_logprobs=True)` opts into real logprobs
(`_mean_logprob`), off by default because not every OpenAI-compatible proxy
supports them; the field simply stays `None` and the heuristic takes over.

---

## Composition & CLI

`compose_backend(base, cheap=…, mid=…, route_mode=…, min_confidence=…)` picks the
right layer and stacks it with the cache / rate-limit layers in the correct order
(router innermost). `route_mode ∈ {"heuristic", "difficulty", "cascade"}`.

```bash
# up-front difficulty routing across three tiers
agent run -t "refactor the scheduler" \
  --cheap-model gpt-oss-20b --mid-model gpt-oss-120b --router difficulty

# confidence cascade: try cheap, escalate only when unsure
agent run -t "fix the failing test" \
  --cheap-model gpt-oss-20b --router cascade --min-confidence 0.4
```

The default (`--cheap-model` alone, no `--router`) preserves the original 2-tier
keyword heuristic, so existing behavior is unchanged.

---

## Design choices & limits

- **Two axes, not one** — difficulty (proactive) and confidence (reactive) answer
  different questions and compose: a difficulty router can feed a cascade tier.
- **Explainable** — the estimator returns reasons; escalations are auditable.
- **Graceful** — no logprobs → heuristic confidence; a single tier → no routing;
  all-off `compose_backend` returns `base` unchanged.
- **Heuristic difficulty** — lexical, so it won't understand semantics a keyword
  misses; supply a custom `RoutePolicy`/`TierPolicy` (even an LLM-judge) for more.
- **Cascade latency** — an escalated step pays two calls; `min_confidence` tunes
  the cost/quality trade-off. Cascades shine when the cheap acceptance rate is high.
