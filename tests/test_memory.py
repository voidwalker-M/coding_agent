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
# ShortTermMemory — conversation window of n queries
# ---------------------------------------------------------------------------

def test_short_term_query_window_trims_to_n():
    stm = ShortTermMemory(window_queries=2)
    stm.append_query("first")
    stm.append_response("did first")
    stm.append_query("second")
    stm.append_query("third")
    texts = [q.user_text for q in stm.queries]
    assert texts == ["second", "third"]
    rendered = stm.render()
    # Current query stays in ConversationHistory; STM only lists *prior* queries.
    assert "second" in rendered
    assert "first" not in rendered or "dropped" in rendered


def test_short_term_begin_query_is_idempotent():
    stm = ShortTermMemory()
    stm.begin_query("fix the parser")
    stm.begin_query("fix the parser")
    assert len(stm.queries) == 1


def test_short_term_persists_window_in_sqlite(tmp_path):
    from context.memory_store import MemoryStore
    store = MemoryStore(tmp_path / "memory.db")
    conv = store.create_conversation("alice", title="chat")
    stm = ShortTermMemory(window_queries=3, store=store, user_id="alice",
                          conversation_id=conv.id)
    stm.append_query("how do I parse JSON")
    stm.append_response("use json.loads")
    reloaded = ShortTermMemory(window_queries=3, store=store, user_id="alice",
                               conversation_id=conv.id)
    assert reloaded.queries[0].user_text == "how do I parse JSON"
    assert "json.loads" in reloaded.queries[0].responses[0]


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


# ---------------------------------------------------------------------------
# Multi-user / ACL / cache
# ---------------------------------------------------------------------------

def test_private_memory_is_invisible_to_other_users(tmp_path, clock):
    alice = LongTermMemory(str(tmp_path), user_id="alice", role="user",
                           clock=lambda: clock["now"]).load()
    alice.remember("Alice's API token lives in secrets.env", kind="user",
                   visibility="private")
    bob = LongTermMemory(str(tmp_path), user_id="bob", role="user",
                         clock=lambda: clock["now"]).load()
    assert bob.count == 0
    assert "secrets.env" not in bob.recall("API token")
    assert "secrets.env" in alice.recall("API token")


def test_public_memory_visible_to_guest(tmp_path, clock):
    agent = LongTermMemory(str(tmp_path), user_id="agent", role="agent",
                           clock=lambda: clock["now"]).load()
    agent.remember("The deploy command is make release", kind="reference",
                   visibility="public", scope="global")
    guest = LongTermMemory(str(tmp_path), user_id="visitor", role="guest",
                           clock=lambda: clock["now"]).load()
    assert "make release" in guest.recall("how do I deploy")


def test_shared_acl_grant(tmp_path, clock):
    alice = LongTermMemory(str(tmp_path), user_id="alice", role="user",
                           clock=lambda: clock["now"]).load()
    rec = alice.remember("staging host is staging.internal", kind="reference",
                         visibility="private")
    alice.grant(rec.name, "user:bob", perm="read")
    bob = LongTermMemory(str(tmp_path), user_id="bob", role="user",
                         clock=lambda: clock["now"]).load()
    assert "staging.internal" in bob.recall("staging host")


def test_sqlite_is_source_of_truth(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.remember("parser is recursive descent", kind="project")
    assert (tmp_path / "memory.db").exists()
    assert (tmp_path / "MEMORY.md").exists()
    reloaded = LongTermMemory(str(tmp_path), clock=lambda: clock["now"]).load()
    assert reloaded.count == 1
    assert reloaded.store.ltm_all()[0]["kind"] == "project"


def test_embed_cache_roundtrip(tmp_path, clock):
    store = LongTermMemory(str(tmp_path), clock=lambda: clock["now"]).load().store
    store.cache_set("embed", "abc", b"\x00\x01", ttl_s=60)
    assert store.cache_get("embed", "abc") == b"\x00\x01"
    store.cache_set("embed", "old", b"x", ttl_s=-1)
    assert store.cache_get("embed", "old") is None


def test_guest_cannot_write(tmp_path, clock):
    guest = LongTermMemory(str(tmp_path), user_id="g", role="guest",
                           clock=lambda: clock["now"]).load()
    with pytest.raises(PermissionError):
        guest.remember("should not persist", kind="semantic")


def test_pending_memory_is_hidden_until_approved(tmp_path, clock):
    mem = LongTermMemory(str(tmp_path), clock=lambda: clock["now"],
                         auto_approve=False).load()
    rec = mem.propose("prefer tabs in this repo", kind="feedback")
    assert rec.status == "pending"
    assert mem.pending()
    assert "tabs" not in mem.recall("prefer tabs")
    mem.approve(rec.name)
    assert "tabs" in mem.recall("prefer tabs")


def test_prompt_recall_skips_episodic_logs(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    mem.record_episode("Fix parser", outcome="success", summary="handled empty input")
    mem.remember("use ruff", kind="feedback")
    always_on = mem.recall("parser ruff", k=5, for_prompt=True)
    assert "ruff" in always_on.lower()
    assert "Task: Fix parser" not in always_on
    on_demand = mem.recall("how did we fix the parser", k=5)
    assert "empty input" in on_demand


# ---------------------------------------------------------------------------
# ChatGPT-shaped pipeline: extract facts, archive overflow, keep transcript
# ---------------------------------------------------------------------------

def test_observe_turn_saves_preference_not_chitchat(tmp_path, clock):
    mem = _mem(tmp_path, clock)
    written = mem.observe_turn("I prefer concise answers and I don't like long paragraphs")
    assert written
    assert "concise" in mem.recall("concise answers preference", k=3, for_prompt=True).lower()
    assert mem.observe_turn("ok thanks") == []
    assert mem.count == 1


def test_overflow_keeps_sqlite_transcript_and_distills(tmp_path, clock):
    mem = _mem(tmp_path, clock, user_id="alice")
    conv_id = mem.new_conversation(title="chat")
    stm = ShortTermMemory(
        window_queries=2,
        store=mem.store,
        user_id="alice",
        conversation_id=conv_id,
        on_overflow=mem.ingest_overflow,
    )
    stm.append_query("I prefer concise answers, please never write long paragraphs")
    stm.append_response("I'll keep replies short.")
    stm.append_query("look at parser.py next")
    stm.append_query("now fix the tests")

    texts = [q.user_text for q in stm.queries]
    assert texts == ["look at parser.py next", "now fix the tests"]
    rendered = stm.render()
    assert "parser.py" in rendered
    assert "prefer concise" not in rendered

    rows = mem.store.stm_load(conv_id)
    contents = [row["content"] for row in rows]
    assert any("prefer concise" in c for c in contents)
    assert any("fix the tests" in c for c in contents)

    prompt = mem.recall("concise answers preference", k=5, for_prompt=True)
    assert "concise" in prompt.lower()
    assert any(r.kind == "conversation" for r in mem._records)
