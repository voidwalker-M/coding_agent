"""
llm/rate_limit.py

Multi-dimensional rate limiting and a hard cost ceiling for LLM calls
(token-saving / spend-control feature #4.1).

Three independent dimensions, any of which may be left unset (None = unlimited):
  - requests per minute  (avoid provider 429s)
  - tokens   per minute  (avoid TPM limits)
  - max USD per session  (a HARD spend ceiling — the most useful dimension:
                          once exceeded, the next call raises BudgetExceeded)

The first two use a sliding-window throttle: if the next call would exceed a
per-minute budget, the limiter *sleeps* until the window frees up. The cost
ceiling does not sleep — it raises, so a runaway agent stops instead of burning
money.

Implemented as a decorator `LLMBackend` (no agent-core changes):

    limited = RateLimitedBackend(create_backend(...),
                                 RateLimiter(rpm=20, tpm=40_000, max_usd=1.0,
                                             model="claude-sonnet-4-6"))
"""

from __future__ import annotations

import logging
import time
from collections import deque

from eval.pricing import estimate_cost
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when the session USD ceiling is exceeded. Not retryable by core."""


class RateLimiter:
    """Sliding-window limiter over requests/min and tokens/min + a hard $ ceiling."""

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        max_usd: float | None = None,
        model: str | None = None,
        sleep_fn=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._max_usd = max_usd
        self._model = model
        self._sleep = sleep_fn
        self._clock = clock
        self._req_times: deque[float] = deque()              # request timestamps
        self._tok_events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self.spent_usd = 0.0
        self.total_tokens = 0
        self.waits = 0

    def _purge(self, now: float) -> None:
        cutoff = now - 60.0
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._tok_events and self._tok_events[0][0] < cutoff:
            self._tok_events.popleft()

    def _window_tokens(self) -> int:
        return sum(t for _, t in self._tok_events)

    def before_call(self, est_tokens: int = 0) -> None:
        """Block until a call is allowed; raise BudgetExceeded if over the $ ceiling."""
        if self._max_usd is not None and self.spent_usd >= self._max_usd:
            raise BudgetExceeded(
                f"Session cost ${self.spent_usd:.4f} reached ceiling ${self._max_usd:.2f}"
            )
        # Throttle on requests-per-minute.
        while self._rpm is not None:
            now = self._clock()
            self._purge(now)
            if len(self._req_times) < self._rpm:
                break
            wait = 60.0 - (now - self._req_times[0])
            self.waits += 1
            logger.info("rate limit: rpm reached, sleeping %.1fs", max(wait, 0))
            self._sleep(max(wait, 0.01))
        # Throttle on tokens-per-minute.
        while self._tpm is not None and est_tokens > 0:
            now = self._clock()
            self._purge(now)
            if self._window_tokens() + est_tokens <= self._tpm:
                break
            wait = 60.0 - (now - self._tok_events[0][0]) if self._tok_events else 1.0
            self.waits += 1
            logger.info("rate limit: tpm reached, sleeping %.1fs", max(wait, 0))
            self._sleep(max(wait, 0.01))

    def after_call(self, tokens: int) -> None:
        now = self._clock()
        self._req_times.append(now)
        if tokens > 0:
            self._tok_events.append((now, tokens))
        self.total_tokens += tokens
        if self._max_usd is not None or self._model is not None:
            self.spent_usd += estimate_cost(self._model, tokens)

    def stats(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "total_tokens": self.total_tokens,
            "waits": self.waits,
        }


class RateLimitedBackend(LLMBackend):
    """Decorator backend that enforces a RateLimiter around each call."""

    def __init__(self, inner: LLMBackend, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def supports_function_calling(self) -> bool:
        return self._inner.supports_function_calling

    def _estimate_request_tokens(self, messages: list[LLMMessage]) -> int:
        from context.token_budget import estimate_tokens
        return sum(estimate_tokens(m.content) for m in messages)

    def complete(self, messages, tools) -> LLMResponse:
        self._limiter.before_call(self._estimate_request_tokens(messages))
        resp = self._inner.complete(messages, tools)
        self._limiter.after_call(resp.total_tokens)
        return resp

    def stream(self, messages, tools, on_text=None, on_thought=None) -> LLMResponse:
        self._limiter.before_call(self._estimate_request_tokens(messages))
        if hasattr(self._inner, "stream"):
            resp = self._inner.stream(messages, tools, on_text=on_text, on_thought=on_thought)
        else:
            resp = self._inner.complete(messages, tools)
        self._limiter.after_call(resp.total_tokens)
        return resp
