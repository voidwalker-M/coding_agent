"""Store factory falls back to SQLite; Postgres tests skip without a URL."""

from __future__ import annotations

import pytest

from context.store_factory import infra_status, memory_wanted, open_memory_store, ping_database


def test_open_memory_store_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    store = open_memory_store(tmp_path / "memory.db")
    assert store.kind == "sqlite"
    store.cache_set("demo", "k", b"hello")
    assert store.cache_get("demo", "k") == b"hello"
    store.ping()
    store.close()


def test_memory_wanted_from_env(monkeypatch):
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert memory_wanted() is False
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    assert memory_wanted() is True
    monkeypatch.delenv("MEMORY_ENABLED")
    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent")
    assert memory_wanted() is True


def test_ping_database_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    status = ping_database(tmp_path / "memory.db")
    assert status["ok"] is True
    assert status["backend"] == "sqlite"
    infra = infra_status(tmp_path / "memory.db")
    assert infra["ok"] is True


def test_postgres_store_optional():
    psycopg = pytest.importorskip("psycopg")
    import os
    url = os.environ.get("MEMORY_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url or not url.startswith(("postgres://", "postgresql://")):
        pytest.skip("no postgres URL")
    from context.pg_store import PostgresMemoryStore
    store = PostgresMemoryStore(url, retries=3, delay_s=0.2)
    store.ensure_user("pgtest")
    conv = store.create_conversation("pgtest", title="t")
    store.stm_append(conv.id, 0, "user", "hello pg")
    rows = store.stm_load(conv.id)
    assert rows[0]["content"] == "hello pg"
    store.cache_set("demo", "k", b"pg")
    assert store.cache_get("demo", "k") == b"pg"
    store.close()
    del psycopg
