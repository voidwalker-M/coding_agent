"""
tests/test_memory.py

Tests for the two-tier memory subsystem (feature #2): ShortTermMemory (working
memory + rolling summary), LongTermMemory (markdown store + MEMORY.md index +
lexical/decay retrieval + consolidation), the remember/recall tools, and the
ConversationHistory on_evict hook. Pure-Python — no numpy, no API.
"""

import pytest

from context.history import ConversationHistory
from context.memory import LongTermMemory, MemoryRecord, ShortTermMemory
from llm.base import LLMMessage
from tools.memory_tool import RecallTool, RememberTool


@pytest.fixture
def clock():
    t = {"now": 1_000_000.0}
    return t


def _mem(tmp_path, clock, **kw):
    return LongTermMemory(str(tmp_path), clock=lambda: clock["now"], **kw).load()


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

def test_short_term_render_includes_notes_and_files():
    stm = ShortTermMemory()
    stm.add_note("bug is in format.py")
    stm.note_file("store/format.py")
    out = stm.render()
    assert "bug is in format.py" in out
    assert "store/format.py" in out


def test_short_term_dedupes_files_and_bounds_notes():
    stm = ShortTermMemory(max_notes=2)
    stm.note_file("a.py")
    stm.note_file("a.py")
    assert stm._files == ["a.py"]
    for i in range(5):
        stm.add_note(f"note {i}")
    assert len(stm._notes) == 2          # bounded to the most recent


def test_short_term_rolling_summary_is_bounded():
    stm = ShortTermMemory(max_summary_chars=100)
    for i in range(50):
        stm.fold(f"evicted turn number {i} with some content")
    assert len(stm._summary) <= 101      # bounded (+1 for the leading ellipsis)


def test_history_on_evict_folds_into_short_term():
    stm = ShortTermMemory()
    hist = ConversationHistory(max_messages=3, on_evict=stm.make_evict_callback())
    hist.add(LLMMessage(role="user", content="task description"))     # index 0, never evicted
    hist.add(LLMMessage(role="assistant", content="FIRST step observation"))
    hist.add(LLMMessage(role="user", content="second"))
    hist.add(LLMMessage(role="assistant", content="third"))           # evicts index 1
    assert "FIRST step observation" in stm.render()
    assert hist.message_count == 3       # window still respected


# ---------------------------------------------------------------------------
# LongTermMemory — write / persist / dedup
# ---------------------------------------------------------------------------

def test_remember_and_recall_roundtrip(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("Repo runs slow tests with pytest -m slow", kind="reference", tags=["pytest"])
    block = mem.recall("how to run the slow pytest tests", k=3)
    assert "pytest" in block.lower()


def test_persistence_reload(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("The parser uses recursive descent in parser/core.py", kind="project")
    # A file per memory + the index exist on disk.
    assert (tmp_path / "MEMORY.md").exists()
    assert len(list(tmp_path.glob("*.md"))) == 2     # 1 record + MEMORY.md

    reloaded = _mem(tmp_path, clock)
    assert reloaded.count == 1
    hits = reloaded.select("recursive descent parser", k=1)
    assert hits and "parser" in hits[0][0].text.lower()


def test_dedup_on_identical_content(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("Use tabs not spaces in this repo", kind="feedback")
    mem.remember("Use tabs not spaces in this repo", kind="feedback")
    assert mem.count == 1                            # reinforced, not duplicated


def test_markdown_frontmatter_roundtrip():
    rec = MemoryRecord(
        name="x", description="d", kind="reference", text="body text",
        tags=["a", "b"], files=["f.py"], importance=0.6, created_at=123.0,
    )
    raw = rec.to_markdown()
    assert raw.startswith("---")
    back = MemoryRecord.from_markdown(raw, default_name="x")
    assert back.kind == "reference"
    assert back.tags == ["a", "b"]
    assert back.text == "body text"
    assert back.importance == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# LongTermMemory — retrieval quality
# ---------------------------------------------------------------------------

def test_lexical_ranking_prefers_relevant(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("Authentication uses JWT tokens in auth/login.py", kind="project")
    mem.remember("The CSS theme colors live in styles/theme.css", kind="project")
    hits = mem.select("how does login authentication work", k=1)
    assert hits[0][0].text.startswith("Authentication")


def test_recall_filters_and_touches_access_count(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    rec = mem.remember("deploy script is scripts/deploy.sh", kind="reference")
    assert rec.access_count == 0
    mem.select("how do I deploy", k=1)
    assert mem._by_name[rec.name].access_count == 1     # recall reinforces


def test_recall_reinforcement_survives_reload(tmp_path, clock):
    """access_count/last_access feed decay + eviction, so they must be persisted —
    otherwise a frequently-recalled memory looks never-used to the next session."""
    mem = _mem(tmp_path, clock)
    rec = mem.remember("deploy script is scripts/deploy.sh", kind="reference")
    mem.select("how do I deploy", k=1)
    mem.select("deploy the app", k=1)

    reloaded = _mem(tmp_path, clock)
    assert reloaded._by_name[rec.name].access_count == 2
    assert reloaded._by_name[rec.name].last_access == pytest.approx(clock["now"])


def test_episode_capture_marks_outcome(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    rec = mem.record_episode(
        "Fix the failing parser test", outcome="success",
        files=["parser/core.py"], summary="handled empty input")
    assert rec.kind == "episodic"
    assert rec.outcome == "success"
    assert "parser/core.py" in rec.text
    assert rec.importance == pytest.approx(0.7)          # wins are salient


# ---------------------------------------------------------------------------
# Consolidation (decay / dedup / cap)
# ---------------------------------------------------------------------------

def test_consolidate_caps_to_max_records(tmp_path, clock):
    mem = _mem(tmp_path, clock, max_records=3)
    for i in range(6):
        mem.remember(f"distinct fact number {i} about widget {i}", kind="semantic")
    stats = mem.consolidate()
    assert mem.count == 3
    assert stats["records"] == 3
    # index + surviving files only
    assert len(list(tmp_path.glob("*.md"))) == 3 + 1


def test_maybe_consolidate_is_gated(tmp_path, clock):
    mem = _mem(tmp_path, clock, consolidate_threshold=2,
               consolidate_min_interval_s=1000.0, consolidate_min_new=1, max_records=100)
    mem.remember("fact one about alpha", kind="semantic")
    mem.remember("fact two about beta", kind="semantic")   # count reaches threshold
    # First consolidation happens (last_consolidation was 0).
    mem.remember("fact three about gamma", kind="semantic")
    # Immediately after, the interval gate blocks a second consolidation.
    assert mem.maybe_consolidate() is None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_remember_tool_writes(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    tool = RememberTool(mem)
    res = tool.execute({"text": "prefer ruff over flake8 here", "kind": "feedback"})
    assert res.success
    assert mem.count == 1


def test_recall_tool_reads(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("the build command is make release", kind="reference")
    tool = RecallTool(mem)
    res = tool.execute({"query": "how do I build a release", "k": 3})
    assert res.success
    assert "make release" in res.output


def test_remember_tool_requires_text(tmp_path, clock):
    tool = RememberTool(_mem(tmp_path, clock))
    res = tool.execute({"text": ""})
    assert not res.success
