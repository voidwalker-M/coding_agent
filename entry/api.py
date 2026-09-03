"""
entry/api.py

FastAPI service entry for the coding agent.

Endpoints:
  POST /v1/tasks          — submit a task (runs in a thread-pool worker)
  GET  /v1/tasks/{id}     — poll status + latency stats
  POST /v1/tasks/{id}/resume — resume from latest checkpoint
  GET  /health            — liveness + postgres/redis
  POST/GET /v1/memory     — long-term facts (Postgres or SQLite)
  POST/GET /v1/conversations — STM turns
  POST/GET /v1/cache      — namespaced KV (Redis when configured)

Usage:
  agent serve --port 8766
  # or: uvicorn entry.api:app --host 0.0.0.0 --port 8766
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI not installed. Run: pip install 'coding-agent[server]'"
    ) from exc

from agent.checkpoint import checkpoint_path_for, load_checkpoint
from agent.task import RunResult, RunStatus, Task
from entry.runtime import AgentRuntime, build_runtime, new_event_log, new_task

# ---------------------------------------------------------------------------
# In-memory job store (demo / single-node; swap for Redis in production)
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class JobRecord:
    job_id: str
    task_id: str
    description: str
    repo_path: str
    state: JobState = JobState.QUEUED
    result: RunResult | None = None
    log_path: str | None = None
    checkpoint_path: str | None = None
    submit_ms: float = 0.0
    complete_ms: float | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_jobs: dict[str, JobRecord] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_runtime_lock = threading.Lock()
_runtime = None
_checkpoint_dir = Path("./checkpoints")
_log_dir = Path("./logs")
_repo_path = Path(".")
_memory_store = None
_kv_cache = None
_ltm = None


def _get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_runtime(repo_path=".", stream=True)
        return _runtime


def _open_infra(repo_path: str) -> None:
    """Open Postgres/SQLite + Redis for memory APIs (independent of the LLM)."""
    global _memory_store, _kv_cache, _ltm, _repo_path
    from context.kv_cache import open_kv_cache
    from context.memory import LongTermMemory
    from context.store_factory import open_memory_store

    _repo_path = Path(repo_path).resolve()
    mem_dir = _repo_path / ".agent_memory"
    _kv_cache = open_kv_cache()
    _memory_store = open_memory_store(mem_dir / "memory.db", kv=_kv_cache)
    _ltm = LongTermMemory(mem_dir, store=_memory_store).load()


def configure_service(*, repo_path: str = ".", checkpoint_dir: str = "./checkpoints",
                      log_dir: str = "./logs", workers: int = 4,
                      runtime: AgentRuntime | None = None) -> None:
    """Called by CLI before uvicorn starts."""
    global _runtime, _checkpoint_dir, _log_dir, _executor
    _checkpoint_dir = Path(checkpoint_dir)
    _log_dir = Path(log_dir)
    _checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _log_dir.mkdir(parents=True, exist_ok=True)
    _executor = ThreadPoolExecutor(max_workers=max(1, workers))
    _open_infra(repo_path)
    with _runtime_lock:
        _runtime = runtime or build_runtime(repo_path=repo_path, stream=True)


def inject_runtime(runtime: AgentRuntime) -> None:
    """Test hook: inject a pre-built runtime (typically MockBackend)."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TaskSubmitRequest(BaseModel):
    description: str = Field(..., min_length=1)
    repo_path: str = "."
    max_steps: int | None = None


class TaskSubmitResponse(BaseModel):
    job_id: str
    task_id: str
    state: str


class TaskStatusResponse(BaseModel):
    job_id: str
    task_id: str
    state: str
    steps_taken: int | None = None
    summary: str | None = None
    checkpoint_path: str | None = None
    avg_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    submit_to_complete_ms: float | None = None
    log_path: str | None = None
    error: str | None = None


class ResumeRequest(BaseModel):
    checkpoint_path: str | None = None


class MemoryWriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str = "default"
    kind: str = "semantic"
    visibility: str = "private"
    scope: str = "user"


class ConversationCreateRequest(BaseModel):
    user_id: str = "default"
    title: str = ""


class TurnWriteRequest(BaseModel):
    role: str = "user"
    content: str = Field(..., min_length=1)
    query_index: int = 0


class CacheWriteRequest(BaseModel):
    namespace: str = "demo"
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    ttl_s: float | None = 3600.0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_job(job: JobRecord, *, resume_from: str | None = None) -> None:
    t0 = time.perf_counter()
    with job.lock:
        job.state = JobState.RUNNING

    try:
        rt = _get_runtime()
        if resume_from:
            cp = load_checkpoint(resume_from)
            task = Task(**cp.task)
        else:
            task = new_task(job.description, job.repo_path)
        task.task_id = job.task_id

        log = new_event_log(task, str(_log_dir))
        with job.lock:
            job.log_path = str(log.path)

        result = rt.agent.run(
            task, log,
            checkpoint_dir=str(_checkpoint_dir),
            resume_from=resume_from,
        )
        log.close()

        with job.lock:
            job.result = result
            job.checkpoint_path = result.checkpoint_path or str(
                checkpoint_path_for(task.task_id, _checkpoint_dir)
            )
            if result.is_success():
                job.state = JobState.SUCCESS
            elif result.status == RunStatus.INTERRUPTED:
                job.state = JobState.INTERRUPTED
            else:
                job.state = JobState.FAILED
            job.complete_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        with job.lock:
            job.state = JobState.FAILED
            job.error = str(exc)
            job.complete_ms = (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Coding Agent API", version="0.1.0")


def _llm_status() -> dict[str, Any]:
    rt = _runtime
    if rt is None:
        return {"ok": False, "backend": "none"}
    name = getattr(rt.backend, "model_name", "unknown")
    if name == "offline":
        return {
            "ok": False,
            "backend": "offline",
            "detail": "set GPT_OSS_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY",
        }
    return {"ok": True, "backend": name}


@app.get("/health")
def health():
    from context.store_factory import infra_status

    mem_path = (_repo_path / ".agent_memory" / "memory.db")
    infra = infra_status(mem_path)
    body: dict[str, Any] = {
        "status": "ok" if infra["ok"] else "degraded",
        "memory": infra["memory"],
        "cache": infra["cache"],
        "llm": _llm_status(),
    }
    if infra["ok"]:
        return body
    if JSONResponse is None:
        return body
    return JSONResponse(status_code=503, content=body)


@app.post("/v1/tasks", response_model=TaskSubmitResponse)
def submit_task(req: TaskSubmitRequest) -> TaskSubmitResponse:
    job_id = str(uuid.uuid4())[:8]
    task_id = str(uuid.uuid4())[:8]
    job = JobRecord(
        job_id=job_id,
        task_id=task_id,
        description=req.description,
        repo_path=str(Path(req.repo_path).resolve()),
        submit_ms=time.perf_counter() * 1000,
    )
    _jobs[job_id] = job
    _executor.submit(_run_job, job)
    return TaskSubmitResponse(job_id=job_id, task_id=task_id, state=job.state.value)


@app.get("/v1/tasks/{job_id}", response_model=TaskStatusResponse)
def get_task(job_id: str) -> TaskStatusResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    with job.lock:
        r = job.result
        return TaskStatusResponse(
            job_id=job.job_id,
            task_id=job.task_id,
            state=job.state.value,
            steps_taken=r.steps_taken if r else None,
            summary=r.summary if r else None,
            checkpoint_path=job.checkpoint_path or (r.checkpoint_path if r else None),
            avg_ttft_ms=r.avg_ttft_ms if r else None,
            p95_ttft_ms=r.p95_ttft_ms if r else None,
            submit_to_complete_ms=job.complete_ms,
            log_path=job.log_path,
            error=job.error,
        )


@app.post("/v1/tasks/{job_id}/resume", response_model=TaskSubmitResponse)
def resume_task(job_id: str, body: ResumeRequest | None = None) -> TaskSubmitResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    ckpt = (body.checkpoint_path if body and body.checkpoint_path else job.checkpoint_path)
    if not ckpt or not Path(ckpt).exists():
        raise HTTPException(status_code=400, detail="no checkpoint available")
    with job.lock:
        job.state = JobState.QUEUED
        job.result = None
        job.error = None
    _executor.submit(_run_job, job, resume_from=ckpt)
    return TaskSubmitResponse(job_id=job.job_id, task_id=job.task_id, state=JobState.QUEUED.value)


def _require_store():
    if _memory_store is None or _ltm is None:
        raise HTTPException(status_code=503, detail="memory store not initialized")
    return _memory_store, _ltm


@app.post("/v1/memory")
def write_memory(req: MemoryWriteRequest) -> dict[str, Any]:
    store, ltm = _require_store()
    store.ensure_user(req.user_id)
    rec = ltm.as_actor(req.user_id, role="user").remember(
        req.text, kind=req.kind, visibility=req.visibility, scope=req.scope,
        owner_user_id=req.user_id,
    )
    return {
        "name": rec.name,
        "text": rec.text,
        "kind": rec.kind,
        "user_id": rec.owner_user_id,
        "visibility": rec.visibility,
        "store": getattr(store, "kind", "sqlite"),
    }


@app.get("/v1/memory")
def read_memory(q: str = "", user_id: str = "default", k: int = 5) -> dict[str, Any]:
    store, ltm = _require_store()
    view = ltm.as_actor(user_id, role="user")
    if q.strip():
        hits = view.select(q, k=max(1, min(k, 20)))
        items = [{"name": r.name, "text": r.text, "kind": r.kind, "score": score} for r, score in hits]
    else:
        items = [
            {"name": r.name, "text": r.text, "kind": r.kind, "user_id": r.owner_user_id}
            for r in view._records[: max(1, min(k, 50))]
        ]
    return {"count": len(items), "items": items, "store": getattr(store, "kind", "sqlite")}


@app.post("/v1/conversations")
def create_conversation(req: ConversationCreateRequest) -> dict[str, Any]:
    store, _ = _require_store()
    conv = store.create_conversation(req.user_id, title=req.title)
    return {
        "id": conv.id,
        "user_id": conv.user_id,
        "title": conv.title,
        "created_at": conv.created_at,
    }


@app.get("/v1/conversations")
def list_conversations(user_id: str = "default") -> dict[str, Any]:
    store, _ = _require_store()
    rows = store.list_conversations(user_id)
    return {
        "items": [
            {"id": r.id, "user_id": r.user_id, "title": r.title, "updated_at": r.updated_at}
            for r in rows
        ]
    }


@app.post("/v1/conversations/{conversation_id}/turns")
def append_turn(conversation_id: str, req: TurnWriteRequest) -> dict[str, Any]:
    store, _ = _require_store()
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    store.stm_append(conversation_id, req.query_index, req.role, req.content)
    return {"ok": True, "conversation_id": conversation_id}


@app.get("/v1/conversations/{conversation_id}/turns")
def load_turns(conversation_id: str) -> dict[str, Any]:
    store, _ = _require_store()
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = store.stm_load(conversation_id)
    items = []
    for row in rows:
        items.append({
            "query_index": int(row["query_index"]),
            "role": row["role"],
            "content": row["content"],
            "created_at": float(row["created_at"]),
        })
    return {"conversation_id": conversation_id, "items": items}


@app.post("/v1/cache")
def cache_write(req: CacheWriteRequest) -> dict[str, Any]:
    store, _ = _require_store()
    payload = req.value.encode("utf-8")
    store.cache_set(req.namespace, req.key, payload, ttl_s=req.ttl_s)
    backend = "redis" if getattr(getattr(store, "_kv", None), "kind", "") == "redis" else getattr(store, "kind", "sqlite")
    return {"ok": True, "namespace": req.namespace, "key": req.key, "backend": backend}


@app.get("/v1/cache")
def cache_read(namespace: str = "demo", key: str = "") -> dict[str, Any]:
    store, _ = _require_store()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    raw = store.cache_get(namespace, key)
    if raw is None:
        raise HTTPException(status_code=404, detail="cache miss")
    try:
        value = raw.decode("utf-8")
    except Exception:
        value = raw.hex()
    return {"namespace": namespace, "key": key, "value": value}
