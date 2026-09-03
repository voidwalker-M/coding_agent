"""
context/kv_cache.py

Namespaced byte cache: in-process dict, or Redis when REDIS_URL / CACHE_URL is set.

Used as a hot layer in front of the durable memory-store cache table, and as
an optional L2 for LLM exact-match responses.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class KvCache(Protocol):
    kind: str

    def get(self, namespace: str, key: str) -> bytes | None: ...
    def set(self, namespace: str, key: str, value: bytes, *, ttl_s: float | None = None) -> None: ...
    def delete(self, namespace: str, key: str | None = None) -> None: ...
    def ping(self) -> bool: ...


def _wire_key(namespace: str, key: str) -> str:
    return f"agent:{namespace}:{key}"


class MemoryKvCache:
    """Process-local dict with optional TTL. Default when Redis is not configured."""

    kind = "memory"

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._data: dict[str, tuple[bytes, float | None]] = {}

    def get(self, namespace: str, key: str) -> bytes | None:
        slot = self._data.get(_wire_key(namespace, key))
        if slot is None:
            return None
        value, exp = slot
        if exp is not None and exp < self._clock():
            self._data.pop(_wire_key(namespace, key), None)
            return None
        return value

    def set(self, namespace: str, key: str, value: bytes, *, ttl_s: float | None = None) -> None:
        exp = (self._clock() + float(ttl_s)) if ttl_s else None
        self._data[_wire_key(namespace, key)] = (bytes(value), exp)

    def delete(self, namespace: str, key: str | None = None) -> None:
        if key is None:
            prefix = f"agent:{namespace}:"
            for k in [k for k in self._data if k.startswith(prefix)]:
                self._data.pop(k, None)
            return
        self._data.pop(_wire_key(namespace, key), None)

    def ping(self) -> bool:
        return True


class RedisKvCache:
    """Redis-backed namespaced KV. Requires the `redis` package."""

    kind = "redis"

    def __init__(self, client, *, clock: Callable[[], float] = time.time) -> None:
        self._client = client
        self._clock = clock

    @classmethod
    def connect(cls, url: str, *, retries: int = 15, delay_s: float = 1.0) -> "RedisKvCache":
        import redis as redis_lib

        last: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                client = redis_lib.from_url(url, decode_responses=False)
                client.ping()
                return cls(client)
            except Exception as exc:
                last = exc
                if attempt + 1 < retries:
                    time.sleep(delay_s)
        raise ConnectionError(f"Redis not reachable at {url}: {last}") from last

    def get(self, namespace: str, key: str) -> bytes | None:
        raw = self._client.get(_wire_key(namespace, key))
        if raw is None:
            return None
        return bytes(raw)

    def set(self, namespace: str, key: str, value: bytes, *, ttl_s: float | None = None) -> None:
        name = _wire_key(namespace, key)
        payload = bytes(value)
        if ttl_s:
            self._client.setex(name, int(max(1, ttl_s)), payload)
        else:
            self._client.set(name, payload)

    def delete(self, namespace: str, key: str | None = None) -> None:
        if key is None:
            for name in self._client.scan_iter(match=f"agent:{namespace}:*"):
                self._client.delete(name)
            return
        self._client.delete(_wire_key(namespace, key))

    def ping(self) -> bool:
        return bool(self._client.ping())


def redis_url() -> str | None:
    return os.environ.get("REDIS_URL") or os.environ.get("CACHE_URL") or None


def open_kv_cache(*, strict: bool = False, retries: int = 5, delay_s: float = 0.4) -> KvCache:
    """Return Redis when REDIS_URL is set, otherwise an in-memory cache."""
    url = redis_url()
    if not url:
        return MemoryKvCache()
    try:
        return RedisKvCache.connect(url, retries=retries, delay_s=delay_s)
    except Exception as exc:
        if strict:
            raise
        logger.warning("Redis unavailable (%s); falling back to in-memory cache", exc)
        return MemoryKvCache()


def ping_cache() -> dict:
    """Health payload. Redis is required only when REDIS_URL is configured."""
    url = redis_url()
    if not url:
        return {"ok": True, "backend": "memory", "configured": False}
    try:
        cache = RedisKvCache.connect(url, retries=1, delay_s=0.2)
        cache.ping()
        cache.set("health", "ping", b"1", ttl_s=30)
        got = cache.get("health", "ping")
        return {"ok": got == b"1", "backend": "redis", "configured": True}
    except Exception as exc:
        return {"ok": False, "backend": "redis", "configured": True, "error": str(exc)}
