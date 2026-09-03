"""
Turn a chat turn into durable memories.

ChatGPT / Claude do *not* dump the whole transcript into the prompt. They
keep the thread, and *extract* short facts (preferences, identity, durable
project notes). Overflowing the STM window must not delete those facts.

This extractor is heuristic (no extra LLM call) so it runs offline in tests.
It is conservative: better to miss a fact than to pollute LTM with "ok thanks".
"""

from __future__ import annotations

import re
from typing import Iterable

# Durable user/project statements — same family ChatGPT Memory looks for.
_FACT_RE = re.compile(
    r"(?is)\b("
    r"i (prefer|like|hate|always|never|usually|don't like|do not like)|"
    r"please always|please never|"
    r"remember (that|this)|don't forget|"
    r"my name is|call me|"
    r"i(?:'m| am) (a |an )|"
    r"i work (on|as|at|with)|"
    r"this (repo|project|codebase) "
    r")\b"
)

_NOISE_RE = re.compile(
    r"(?i)^(ok|okay|thanks|thank you|yes|no|yep|nah|cool|sure|got it)[.!\s]*$"
)


def is_noise(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or len(t) < 12 or bool(_NOISE_RE.match(t))


def extract_durable_facts(user_text: str, assistant_text: str = "") -> list[tuple[str, str, float]]:
    """Return (text, kind, importance) facts worth keeping across sessions."""
    user_text = (user_text or "").strip()
    if is_noise(user_text):
        return []
    if not _FACT_RE.search(user_text):
        return []
    kind = "user"
    lower = user_text.lower()
    if "this repo" in lower or "this project" in lower or "this codebase" in lower:
        kind = "project"
    elif lower.startswith("remember") or "don't forget" in lower:
        kind = "semantic"
    importance = 0.86 if kind == "user" else 0.78
    return [(user_text, kind, importance)]


def archive_snippet(user_text: str, assistant_text: str = "", *, limit: int = 800) -> str | None:
    """Searchable conversation snippet for overflow (ChatGPT 'chat history')."""
    user_text = (user_text or "").strip()
    if is_noise(user_text):
        return None
    assistant_text = (assistant_text or "").strip().replace("\n", " ")
    body = f"User: {user_text}"
    if assistant_text:
        body += f"\nAssistant: {assistant_text[:400]}"
    if len(body) > limit:
        body = body[: limit - 1] + "…"
    return body


def distill_queries(queries: Iterable) -> list[tuple[str, str, float]]:
    """Facts + conversation archives from STM queries falling out of the window."""
    out: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for q in queries:
        user = getattr(q, "user_text", "") or ""
        replies = getattr(q, "responses", None) or []
        assistant = " ".join(str(r) for r in replies)
        for text, kind, importance in extract_durable_facts(user, assistant):
            key = text.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append((text, kind, importance))
        snippet = archive_snippet(user, assistant)
        if snippet:
            key = snippet.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append((snippet, "conversation", 0.42))
    return out
