"""
context/compaction.py

LLM-based conversation history compaction.

When history grows beyond a token threshold, middle turns are summarized by the
LLM into a single user message. The task description (index 0) and the most
recent messages are always preserved verbatim.

Falls back silently (no mutation) if summarization fails or returns empty text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from context.history import ConversationHistory
from context.token_budget import TokenBudget, estimate_tokens
from llm.base import LLMBackend, LLMMessage, LLMResponse, LLMToolSchema

logger = logging.getLogger(__name__)

COMPACTION_MARKER = "[Compacted context summary — earlier steps summarized by LLM]"

_COMPACTION_SYSTEM = """\
You compress coding-agent conversation history to save context window space.

Summarize the agent turns below into a concise brief. Preserve:
- What was tried and what happened (tool calls, outcomes)
- Files read, edited, or tested
- Test pass/fail results and error messages (short form)
- Key decisions and any unfinished work

Omit verbatim long tool output, repeated exploration, and filler.
Use bullet points. Be factual — do not invent actions not present in the log.\
"""


@dataclass
class CompactionSettings:
    enabled: bool = False
    trigger_tokens: int = 24_000
    keep_recent_messages: int = 8
    min_messages_to_compact: int = 4
    max_summary_tokens: int = 1_500


@dataclass
class CompactionResult:
    compacted: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    messages_before: int = 0
    messages_after: int = 0


SummarizeFn = Callable[[str], tuple[str, int, int]]


class HistoryCompactor:
    """
    Compacts ConversationHistory in place when over the token trigger.

    Usage:
        compactor = HistoryCompactor(backend, CompactionSettings(enabled=True))
        result = compactor.maybe_compact(history)
    """

    def __init__(
        self,
        backend: LLMBackend,
        settings: CompactionSettings | None = None,
        summarizer: SummarizeFn | None = None,
    ) -> None:
        self._backend = backend
        self._settings = settings or CompactionSettings()
        self._summarizer = summarizer

    @property
    def settings(self) -> CompactionSettings:
        return self._settings

    def history_token_count(self, history: ConversationHistory) -> int:
        return sum(
            estimate_tokens(m.content) for m in history.to_list()
        )

    def should_compact(self, history: ConversationHistory) -> bool:
        if not self._settings.enabled:
            return False
        msgs = history.to_list()
        if len(msgs) < self._settings.min_messages_to_compact + 2:
            return False
        return self.history_token_count(history) > self._settings.trigger_tokens

    def maybe_compact(self, history: ConversationHistory) -> CompactionResult:
        """Compact history when over threshold. Mutates history on success."""
        if not self.should_compact(history):
            return CompactionResult()

        msgs = history.to_list()
        before = len(msgs)
        keep = min(self._settings.keep_recent_messages, max(0, len(msgs) - 1))
        pinned = msgs[0]
        recent = msgs[-keep:] if keep else []
        middle_end = len(msgs) - keep if keep else len(msgs)
        middle = msgs[1:middle_end]

        if len(middle) < self._settings.min_messages_to_compact:
            return CompactionResult(messages_before=before, messages_after=before)

        transcript = _format_transcript(middle)
        try:
            if self._summarizer is not None:
                summary, in_tok, out_tok = self._summarizer(transcript)
            else:
                summary, in_tok, out_tok = _llm_summarize(
                    self._backend, transcript, self._settings.max_summary_tokens,
                )
        except Exception as exc:
            logger.warning("History compaction failed, continuing without it: %s", exc)
            return CompactionResult(messages_before=before, messages_after=before)

        summary = (summary or "").strip()
        if not summary:
            logger.warning("History compaction returned empty summary; skipping")
            return CompactionResult(
                messages_before=before,
                messages_after=before,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        summary = TokenBudget().trim_to(summary, self._settings.max_summary_tokens)
        summary_msg = LLMMessage(
            role="user",
            content=f"{COMPACTION_MARKER}\n{summary}",
        )
        history.replace_messages([pinned, summary_msg, *recent])

        after = history.message_count
        logger.info(
            "Compacted history: %d → %d messages (%d tokens in, %d out)",
            before, after, in_tok, out_tok,
        )
        return CompactionResult(
            compacted=True,
            input_tokens=in_tok,
            output_tokens=out_tok,
            messages_before=before,
            messages_after=after,
        )


def _format_transcript(messages: list[LLMMessage]) -> str:
    parts: list[str] = []
    for i, msg in enumerate(messages, 1):
        role = msg.role.upper()
        content = (msg.content or "").strip()
        if len(content) > 4_000:
            content = content[:4_000] + "\n… [truncated for compaction input]"
        parts.append(f"--- Turn {i} ({role}) ---\n{content}")
    return "\n\n".join(parts)


def _extract_summary_text(response: LLMResponse) -> str:
    text = (response.raw_content or "").strip()
    if text.startswith("[mock]"):
        text = ""
    if not text and response.action.message:
        text = response.action.message.strip()
    return text


def _llm_summarize(
    backend: LLMBackend,
    transcript: str,
    max_summary_tokens: int,
) -> tuple[str, int, int]:
    messages = [
        LLMMessage(role="system", content=_COMPACTION_SYSTEM),
        LLMMessage(
            role="user",
            content=(
                f"Summarize the following agent history "
                f"(target ≤ {max_summary_tokens} tokens):\n\n{transcript}"
            ),
        ),
    ]
    response = backend.complete(messages, tools=[])
    return (
        _extract_summary_text(response),
        response.input_tokens,
        response.output_tokens,
    )
