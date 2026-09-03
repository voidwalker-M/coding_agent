"""
context/store_factory.py

Pick the durable memory backend from the environment:

  MEMORY_DATABASE_URL / DATABASE_URL
      postgresql://...  → PostgresMemoryStore
      anything else / unset → SQLite at the given path

  REDIS_URL / CACHE_URL
      optional hot cache in front of the store's cache table
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from context.kv_cache import open_kv_cache, ping_cache
from context.memory_store import MemoryStore

logger = logging.getLogger(__name__)


def database_url() -> str | None:
    return os.environ.get("MEMORY_DATABASE_URL") or os.environ.get("DATABASE_URL") or None


def memory_wanted() -> bool:
    flag = os.environ.get("MEMORY_ENABLED", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    url = database_url()
    return bool(url and url.startswith(("postgres://", "postgresql://")))


def open_memory_store(
    sqlite_path: str | Path,
    *,
    clock: Callable[[], float] = time.time,
    kv: Any | None = None,
    retries: int = 30,
) -> Any:
    """Open Postgres when a postgres URL is set, otherwise SQLite."""
    if kv is None:
        kv = open_kv_cache(retries=min(retries, 15))
    url = database_url()
    if url and url.startswith(("postgres://", "postgresql://")):
        from context.pg_store import PostgresMemoryStore
        logger.info("memory store: postgres")
        return PostgresMemoryStore(url, clock=clock, retries=retries, kv=kv)
    store = MemoryStore(sqlite_path, clock=clock)
    store._kv = kv
    return store


def ping_database(sqlite_path: str | Path | None = None) -> dict:
    url = database_url()
    if url and url.startswith(("postgres://", "postgresql://")):
        try:
            from context.pg_store import PostgresMemoryStore
            store = PostgresMemoryStore(url, retries=1, delay_s=0.2)
            store.ping()
            store.close()
            return {"ok": True, "backend": "postgres", "configured": True}
        except Exception as exc:
            return {"ok": False, "backend": "postgres", "configured": True, "error": str(exc)}
    path = Path(sqlite_path or "./.agent_memory/memory.db")
    try:
        store = MemoryStore(path)
        store.ping()
        store.close()
        return {"ok": True, "backend": "sqlite", "configured": False, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "backend": "sqlite", "configured": False, "error": str(exc)}


def infra_status(sqlite_path: str | Path | None = None) -> dict:
    memory = ping_database(sqlite_path)
    cache = ping_cache()
    ok = bool(memory.get("ok")) and bool(cache.get("ok"))
    return {"ok": ok, "memory": memory, "cache": cache}
