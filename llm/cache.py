"""
llm/cache.py

Response caching for LLM calls (token-saving feature #4.2).

Two layers, both opt-in:
  1. Exact cache  — key = hash of (model, messages, tools). Safe and lossless:
     a repeated identical request (loops, retries, re-runs) reuses the prior
     response instead of paying for it again.
  2. Semantic cache (optional) — if an embedding function is provided, a request
     whose prompt embedding is within `similarity_threshold` of a cached one
     reuses that response. This is more aggressive and *approximate*: a wrong
     reuse can corrupt an agent run, so the threshold defaults high (0.97) and it
     is OFF unless an embed_fn is supplied.

Implemented as a decorator `LLMBackend` so it requires no changes to the agent
core — wrap any backend:

    cached = CachingBackend(create_backend(...))
    agent = Agent(cached, registry)

The cache stores the parsed Action via a lightweight (de)serialization so a hit
returns a fresh, correct LLMResponse.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable

from agent.task import Action, ActionType, ToolCall
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], "list[float]"]


# ---------------------------------------------------------------------------
# Action (de)serialization — so cached entries reconstruct a real Action
# ---------------------------------------------------------------------------

def _action_to_dict(a: Action) -> dict:
    tc = None
    if a.tool_call is not None:
        tc = {"name": a.tool_call.name, "params": a.tool_call.params}
    return {
        "action_type": a.action_type.value,
        "thought": a.thought,
        "tool_call": tc,
        "message": getattr(a, "message", None),
    }


def _action_from_dict(d: dict) -> Action:
    tc = None
    if d.get("tool_call"):
        tc = ToolCall(d["tool_call"]["name"], d["tool_call"]["params"])
    return Action(
        action_type=ActionType(d["action_type"]),
        thought=d.get("thought", ""),
        tool_call=tc,
        message=d.get("message"),
    )


def _request_key(model: str, messages: list[LLMMessage], tools: list[LLMToolSchema]) -> str:
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "tools": sorted(t.name for t in tools),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# CachingBackend
# ---------------------------------------------------------------------------

class CachingBackend(LLMBackend):
    """Decorator backend that caches responses by exact (and optionally semantic) match."""

    def __init__(
        self,
        inner: LLMBackend,
        embed_fn: EmbedFn | None = None,
        similarity_threshold: float = 0.97,
    ) -> None:
        self._inner = inner
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._exact: dict[str, dict] = {}                  # key -> action dict
        self._semantic: list[tuple[list[float], dict]] = []  # (embedding, action dict)
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def supports_function_calling(self) -> bool:
        return self._inner.supports_function_calling

    def _hit_response(self, action_dict: dict, est_tokens: int) -> LLMResponse:
        self.hits += 1
        self.tokens_saved += est_tokens
        # A cache hit costs 0 tokens.
        return LLMResponse(
            action=_action_from_dict(action_dict),
            raw_content="[cache hit]",
            input_tokens=0,
            output_tokens=0,
        )

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSchema],
    ) -> LLMResponse:
        key = _request_key(self.model_name, messages, tools)

        # 1. Exact cache.
        if key in self._exact:
            logger.debug("LLM cache: exact hit")
            return self._hit_response(self._exact[key], est_tokens=self._avg_tokens())

        # 2. Semantic cache (optional).
        query_emb = None
        if self._embed_fn is not None and self._semantic:
            try:
                query_emb = self._embed_fn(self._semantic_text(messages))
                best_sim, best = 0.0, None
                for emb, act in self._semantic:
                    sim = _cosine(query_emb, emb)
                    if sim > best_sim:
                        best_sim, best = sim, act
                if best is not None and best_sim >= self._threshold:
                    logger.debug("LLM cache: semantic hit (sim=%.3f)", best_sim)
                    return self._hit_response(best, est_tokens=self._avg_tokens())
            except Exception as exc:  # never let caching break a run
                logger.debug("semantic cache lookup failed: %s", exc)

        # 3. Miss → call inner and store.
        self.misses += 1
        resp = self._inner.complete(messages, tools)
        action_dict = _action_to_dict(resp.action)
        self._exact[key] = action_dict
        if self._embed_fn is not None:
            try:
                emb = query_emb if query_emb is not None else self._embed_fn(self._semantic_text(messages))
                self._semantic.append((emb, action_dict))
            except Exception:
                pass
        return resp

    def stream(self, messages, tools, on_text=None, on_thought=None):
        # On a cache hit there is nothing to stream; emit the cached text once.
        key = _request_key(self.model_name, messages, tools)
        if key in self._exact:
            resp = self._hit_response(self._exact[key], est_tokens=self._avg_tokens())
            if on_text:
                on_text(resp.raw_content)
            return resp
        resp = self._inner.stream(messages, tools, on_text=on_text, on_thought=on_thought) \
            if hasattr(self._inner, "stream") else self._inner.complete(messages, tools)
        self._exact[key] = _action_to_dict(resp.action)
        return resp

    # -- helpers --
    @staticmethod
    def _semantic_text(messages: list[LLMMessage]) -> str:
        # Embed the last couple of messages (the volatile part of the prompt).
        return "\n".join(m.content for m in messages[-2:])

    def _avg_tokens(self) -> int:
        # Rough per-call token estimate used only for the "tokens_saved" stat.
        return 1500

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "tokens_saved_est": self.tokens_saved,
        }
