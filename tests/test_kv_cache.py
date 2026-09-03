"""In-process KV cache (Redis path is skipped unless REDIS_URL is set)."""

from context.kv_cache import MemoryKvCache, open_kv_cache, ping_cache
from llm.base import MockBackend
from llm.cache import CachingBackend
from agent.task import Action, ActionType


def test_memory_kv_roundtrip_and_ttl():
    kv = MemoryKvCache(clock=lambda: 100.0)
    kv.set("ns", "k", b"v", ttl_s=10)
    assert kv.get("ns", "k") == b"v"
    kv._clock = lambda: 111.0
    assert kv.get("ns", "k") is None


def test_memory_kv_delete_namespace():
    kv = MemoryKvCache()
    kv.set("a", "1", b"x")
    kv.set("a", "2", b"y")
    kv.set("b", "1", b"z")
    kv.delete("a")
    assert kv.get("a", "1") is None
    assert kv.get("b", "1") == b"z"


def test_open_kv_cache_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("CACHE_URL", raising=False)
    kv = open_kv_cache()
    assert kv.kind == "memory"
    status = ping_cache()
    assert status["ok"] is True
    assert status["backend"] == "memory"


def test_caching_backend_uses_kv_as_l2():
    shared = MemoryKvCache()
    script = [Action(ActionType.FINISH, "one", message="done")]
    inner = MockBackend(script)
    a = CachingBackend(inner, kv=shared)
    r1 = a.complete([], [])
    assert r1.action.message == "done"
    assert inner.call_count == 1

    b = CachingBackend(MockBackend([]), kv=shared)
    r2 = b.complete([], [])
    assert r2.action.message == "done"
    assert r2.raw_content == "[cache hit]"
