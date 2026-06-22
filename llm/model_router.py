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
from typing import Callable

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
