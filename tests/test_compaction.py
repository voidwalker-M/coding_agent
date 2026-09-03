"""Tests for LLM history compaction (context/compaction.py)."""

from __future__ import annotations

from context.compaction import (
    COMPACTION_MARKER,
    CompactionSettings,
    HistoryCompactor,
)
from context.history import ConversationHistory
from llm.base import LLMMessage


def _long_content(n: int = 500) -> str:
    return "x" * n


def test_should_compact_when_over_trigger():
    hist = ConversationHistory(max_messages=100)
    hist.add(LLMMessage(role="user", content="Fix the bug in parser.py"))
    for i in range(12):
        hist.add(LLMMessage(role="assistant", content=f"step {i} " + _long_content(800)))
        hist.add(LLMMessage(role="user", content=f"obs {i} " + _long_content(800)))

    compactor = HistoryCompactor(
        backend=None,  # type: ignore[arg-type]
        settings=CompactionSettings(enabled=True, trigger_tokens=2_000, keep_recent_messages=4),
    )
    assert compactor.should_compact(hist)


def test_maybe_compact_replaces_middle_with_summary():
    hist = ConversationHistory(max_messages=100)
    hist.add(LLMMessage(role="user", content="Task: implement factorial"))
    for i in range(10):
        hist.add(LLMMessage(role="assistant", content=f"Thought: explore {i}"))
        hist.add(LLMMessage(role="user", content=f"[Tool: file_view | SUCCESS] file{i}.py"))

    def fake_summarizer(transcript: str) -> tuple[str, int, int]:
        assert "file0.py" in transcript
        return "- Read file0.py\n- Edited utils.py", 100, 50

    compactor = HistoryCompactor(
        backend=None,  # type: ignore[arg-type]
        settings=CompactionSettings(
            enabled=True,
            trigger_tokens=100,
            keep_recent_messages=4,
            min_messages_to_compact=4,
        ),
        summarizer=fake_summarizer,
    )
    result = compactor.maybe_compact(hist)

    assert result.compacted is True
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    msgs = hist.to_list()
    assert msgs[0].content.startswith("Task:")
    assert COMPACTION_MARKER in msgs[1].content
    assert "Read file0.py" in msgs[1].content
    assert len(msgs) == 1 + 1 + 4  # task + summary + 4 recent
    assert msgs[-1].content.startswith("[Tool:")


def test_compact_skips_when_below_threshold():
    hist = ConversationHistory(max_messages=100)
    hist.add(LLMMessage(role="user", content="short task"))
    hist.add(LLMMessage(role="assistant", content="hi"))

    calls = []

    def fake_summarizer(transcript: str) -> tuple[str, int, int]:
        calls.append(transcript)
        return "summary", 1, 1

    compactor = HistoryCompactor(
        backend=None,  # type: ignore[arg-type]
        settings=CompactionSettings(enabled=True, trigger_tokens=99_999),
        summarizer=fake_summarizer,
    )
    result = compactor.maybe_compact(hist)

    assert result.compacted is False
    assert calls == []
    assert len(hist.to_list()) == 2


def test_compact_skips_on_empty_summary():
    hist = ConversationHistory(max_messages=100)
    hist.add(LLMMessage(role="user", content="task"))
    for i in range(8):
        hist.add(LLMMessage(role="assistant", content="a " + _long_content(600)))
        hist.add(LLMMessage(role="user", content="u " + _long_content(600)))

    compactor = HistoryCompactor(
        backend=None,  # type: ignore[arg-type]
        settings=CompactionSettings(enabled=True, trigger_tokens=100, keep_recent_messages=2),
        summarizer=lambda _t: ("", 10, 5),
    )
    before = len(hist.to_list())
    result = compactor.maybe_compact(hist)

    assert result.compacted is False
    assert len(hist.to_list()) == before


def test_replace_messages_respects_max_window():
    hist = ConversationHistory(max_messages=5)
    hist.add(LLMMessage(role="user", content="task"))
    for i in range(8):
        hist.add(LLMMessage(role="assistant", content=str(i)))

    hist.replace_messages([
        LLMMessage(role="user", content="task"),
        LLMMessage(role="user", content="summary"),
        LLMMessage(role="assistant", content="recent"),
    ])
    assert hist.message_count <= 5
    assert hist.to_list()[0].content == "task"


def test_history_on_evict_still_works_with_replace():
    folded: list[str] = []

    def on_evict(msg: dict) -> None:
        folded.append(msg.get("content", ""))

    hist = ConversationHistory(max_messages=3, on_evict=on_evict)
    hist.add(LLMMessage(role="user", content="task"))
    hist.add(LLMMessage(role="assistant", content="one"))
    hist.add(LLMMessage(role="assistant", content="two"))
    hist.add(LLMMessage(role="assistant", content="three"))

    assert folded == ["one"]
    assert len(hist.to_list()) == 3
