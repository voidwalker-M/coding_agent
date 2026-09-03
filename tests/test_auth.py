"""Local username/password auth for CLI memory isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from context.auth import (
    AuthError,
    authenticate,
    clear_session,
    hash_password,
    load_session,
    normalize_username,
    register_user,
    save_session,
    verify_password,
)
from context.memory import LongTermMemory
from context.memory_store import MemoryStore
from entry.cli import cli


@pytest.fixture
def iterations(monkeypatch):
    monkeypatch.setattr("context.auth.PBKDF2_ITERATIONS", 2_000)
    return 2_000


def test_hash_is_not_plaintext(iterations):
    stored = hash_password("s3cret!", iterations=iterations)
    assert "s3cret!" not in stored
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret!", stored)
    assert not verify_password("wrong", stored)


def test_same_password_different_salts(iterations):
    a = hash_password("s3cret!", iterations=iterations)
    b = hash_password("s3cret!", iterations=iterations)
    assert a != b
    assert verify_password("s3cret!", a)
    assert verify_password("s3cret!", b)


def test_register_and_login(tmp_path, iterations):
    store = MemoryStore(tmp_path / "memory.db")
    user = register_user(store, "Alice", "hunter2")
    assert user.id == "alice"
    assert store.get_password_hash("alice")
    again = authenticate(store, "alice", "hunter2")
    assert again.id == "alice"
    with pytest.raises(AuthError, match="invalid"):
        authenticate(store, "alice", "nope")
    with pytest.raises(AuthError, match="already"):
        register_user(store, "alice", "hunter2")
    with pytest.raises(AuthError, match="reserved"):
        register_user(store, "default", "hunter2")
    store.close()


def test_session_file(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    monkeypatch.setenv("AGENT_SESSION_FILE", str(path))
    assert load_session() is None
    save_session("alice", "user")
    sess = load_session()
    assert sess["user_id"] == "alice"
    assert path.stat().st_mode & 0o777 in (0o600, 0o644)  # 600 when chmod works
    clear_session()
    assert load_session() is None


def test_logged_in_user_owns_memories(tmp_path, monkeypatch, iterations):
    monkeypatch.setenv("AGENT_SESSION_FILE", str(tmp_path / "session.json"))
    db = tmp_path / "memory.db"
    store = MemoryStore(db)
    register_user(store, "alice", "hunter2")
    register_user(store, "bob", "hunter2")
    save_session("alice", "user")

    alice = LongTermMemory(tmp_path, user_id="alice", role="user", store=store).load()
    alice.remember("alice-only fact about the parser", kind="user", visibility="private")
    bob = LongTermMemory(tmp_path, user_id="bob", role="user", store=store).load()
    hits = bob.recall("parser")
    assert "alice-only" not in hits
    alice_hits = alice.recall("parser")
    assert "alice-only" in alice_hits
    store.close()


def test_cli_register_login_whoami(tmp_path, monkeypatch, iterations):
    monkeypatch.setenv("AGENT_SESSION_FILE", str(tmp_path / "session.json"))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["register", "--username", "carol", "--password", "hunter2", "--repo", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "carol" in result.output

    who = runner.invoke(cli, ["whoami"])
    assert who.exit_code == 0
    assert "carol" in who.output

    runner.invoke(cli, ["logout"])
    bad = runner.invoke(
        cli,
        ["login", "--username", "carol", "--password", "wrongpw", "--repo", str(tmp_path)],
    )
    assert bad.exit_code != 0

    ok = runner.invoke(
        cli,
        ["login", "--username", "carol", "--password", "hunter2", "--repo", str(tmp_path)],
    )
    assert ok.exit_code == 0, ok.output
    assert "carol" in ok.output


def test_normalize_username():
    assert normalize_username("Alice") == "alice"
    with pytest.raises(AuthError):
        normalize_username("a")
    with pytest.raises(AuthError):
        normalize_username("1bob")
