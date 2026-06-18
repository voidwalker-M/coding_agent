"""
context/history.py

Sliding-window conversation history management.

Responsibilities:
- Maintain a list of LLMMessages
- Automatically discard oldest (non-first) messages when the window is exceeded
- Cooperate with TokenBudget: first limit by message count, then by token count
- Provide a clean interface for use by core.py

Design:
- The first message (task description) is never discarded
- Reflection prompts and regular observations are treated equally as history
- to_dicts() is used by TokenBudget.trim_history()
"""

from __future__ import annotations

from llm.base import LLMMessage


class ConversationHistory:
    """
    Conversation history manager with a sliding window.

    Usage:
        history = ConversationHistory(max_messages=20)
        history.add(LLMMessage(role="user", content="Fix the bug"))
        history.add(LLMMessage(role="assistant", content="..."))
        msgs = history.to_list()   # pass to LLMBackend
    """

    def __init__(self, max_messages: int = 40) -> None:
        """
        Args:
            max_messages: maximum number of messages to retain (including the first task description).
                          The actual token count sent to the LLM is further trimmed by TokenBudget.
        """
        self._messages: list[LLMMessage] = []
        self._max = max_messages

    def add(self, message: LLMMessage) -> None:
        """Add one message; discard the oldest non-first message if the window is exceeded."""
        self._messages.append(message)
        self._trim()

    def add_many(self, messages: list[LLMMessage]) -> None:
        """Add multiple messages; trim once after all are added."""
        self._messages.extend(messages)
        self._trim()

    def to_list(self) -> list[LLMMessage]:
        """Return the full history list (shallow copy)."""
        return list(self._messages)

    def to_dicts(self) -> list[dict]:
        """Convert to a list of dicts for use by TokenBudget.trim_history()."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    @classmethod
    def from_dicts(cls, dicts: list[dict], max_messages: int = 40) -> "ConversationHistory":
        """Restore from a list of dicts (used when resuming from a checkpoint)."""
        h = cls(max_messages=max_messages)
        h._messages = [LLMMessage(role=d["role"], content=d["content"]) for d in dicts]
        return h

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def last_message(self) -> LLMMessage | None:
        return self._messages[-1] if self._messages else None

    def clear_except_first(self) -> None:
        """Keep only the first task description and discard everything else (emergency reset)."""
        if self._messages:
            self._messages = [self._messages[0]]

    def _trim(self) -> None:
        """When max_messages is exceeded, discard the oldest message starting from index 1."""
        while len(self._messages) > self._max:
            # Preserve index 0 (task description); discard from index 1 onward
            if len(self._messages) > 1:
                self._messages.pop(1)
            else:
                break

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationHistory(messages={len(self._messages)}, max={self._max})"
