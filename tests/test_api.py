"""Tests for FastAPI service entry."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.core import Agent, AgentConfig
from agent.task import Action, ActionType
from config.schema import load_config
from entry.api import _jobs, app, configure_service, inject_runtime
from entry.runtime import AgentRuntime
from llm.base import MockBackend
from tools.base import NoopTool, ToolRegistry


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("CACHE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MEMORY_DATABASE_URL", raising=False)
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    _jobs.clear()
    script = [Action(ActionType.FINISH, "ok", message="done")]
    backend = MockBackend(script)
    reg = ToolRegistry().register(NoopTool("noop"))
    rt = AgentRuntime(
        agent=Agent(backend, reg, AgentConfig(max_steps=5, stream=False)),
        registry=reg,
        config=load_config(),
        backend=backend,
    )
    inject_runtime(rt)
    configure_service(
        repo_path=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"), workers=2, runtime=rt,
    )
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["memory"]["ok"] is True
    assert body["cache"]["ok"] is True
    assert body["llm"]["backend"] == "mock-model"


def test_memory_and_cache_roundtrip(client):
    w = client.post("/v1/memory", json={"text": "repo uses pytest", "user_id": "demo"})
    assert w.status_code == 200
    assert w.json()["text"] == "repo uses pytest"

    g = client.get("/v1/memory", params={"q": "pytest", "user_id": "demo"})
    assert g.status_code == 200
    assert g.json()["count"] >= 1

    c = client.post("/v1/cache", json={"namespace": "demo", "key": "k", "value": "v"})
    assert c.status_code == 200
    got = client.get("/v1/cache", params={"namespace": "demo", "key": "k"})
    assert got.status_code == 200
    assert got.json()["value"] == "v"

    conv = client.post("/v1/conversations", json={"user_id": "demo", "title": "t"})
    cid = conv.json()["id"]
    client.post(f"/v1/conversations/{cid}/turns", json={"role": "user", "content": "hi", "query_index": 0})
    turns = client.get(f"/v1/conversations/{cid}/turns")
    assert turns.json()["items"][0]["content"] == "hi"


def test_submit_and_poll_task(client):
    r = client.post("/v1/tasks", json={"description": "hello", "repo_path": "."})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 10
    state = "queued"
    while time.time() < deadline:
        st = client.get(f"/v1/tasks/{job_id}").json()
        state = st["state"]
        if state in ("success", "failed"):
            break
        time.sleep(0.05)

    assert state == "success"
    st = client.get(f"/v1/tasks/{job_id}").json()
    assert st["summary"] == "done"
