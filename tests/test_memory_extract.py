"""Heuristic durable-fact extractor (no LLM)."""

from context.memory_extract import archive_snippet, distill_queries, extract_durable_facts, is_noise
from context.memory import ConversationQuery


def test_noise_is_ignored():
    assert is_noise("ok")
    assert is_noise("thanks")
    assert extract_durable_facts("ok thanks") == []
    assert archive_snippet("yep") is None


def test_preference_is_extracted():
    hits = extract_durable_facts("I prefer concise answers")
    assert hits
    assert hits[0][1] == "user"


def test_project_note_is_extracted():
    hits = extract_durable_facts("this repo uses pytest -m slow")
    assert hits
    assert hits[0][1] == "project"


def test_distill_queries_archives_and_extracts():
    q = ConversationQuery(
        index=0,
        user_text="I prefer concise answers. Please look at parser.py",
        responses=["ok, short replies from now on"],
    )
    out = distill_queries([q])
    kinds = {k for _, k, _ in out}
    assert "user" in kinds
    assert "conversation" in kinds
