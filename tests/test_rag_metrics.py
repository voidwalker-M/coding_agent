"""
tests/test_rag_metrics.py

Measurement standards for RAG:
1. Chunking + vectorization (avg size, dims, OSS vs in-house)
2. Retrieval perf (latency / QPS) + recall@k / MRR / hit@k
3. Embedding API timeout / retry / resilient failover
4. Incremental cache hit rate + miss regression
5. Production-incident drills (miss / outage / cache storm / context cap)
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from context.rag import (
    EmbeddingBackend,
    HashingEmbeddings,
    OpenAIEmbeddings,
    RagRetriever,
    ResilientEmbeddings,
    _is_non_retryable_embed_error,
    evaluate_recall,
)
from eval.rag_suite import (
    CHUNK_DESIGN,
    EVAL_CASES,
    materialize_corpus,
    measure_chunking_and_embeddings,
    measure_retrieval_perf,
    run_full_report,
    run_incident_drills,
    run_offline_eval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def suite_repo(tmp_path: Path) -> Path:
    return materialize_corpus(tmp_path / "suite")


class FlakyEmbeddings(EmbeddingBackend):
    """Fails ``fail_times`` then delegates to HashingEmbeddings (for retry tests)."""

    def __init__(self, fail_times: int = 2, dim: int = 64, error: str = "timeout") -> None:
        self._fail_times = fail_times
        self._calls = 0
        self._inner = HashingEmbeddings(dim=dim)
        self._error = error
        self.stats = {"calls": 0, "retries": 0, "failures": 0}

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return f"flaky-{self._inner.dim}"

    def embed(self, texts):
        self._calls += 1
        self.stats["calls"] += 1
        if self._calls <= self._fail_times:
            self.stats["failures"] += 1
            raise TimeoutError(self._error)
        return self._inner.embed(texts)


class RetryingEmbeddings(EmbeddingBackend):
    """Mimics OpenAIEmbeddings retry loop around a flaky inner callable."""

    def __init__(self, inner: FlakyEmbeddings, max_retries: int = 3, retry_delay: float = 0.0):
        self._inner = inner
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self.stats = {"calls": 0, "retries": 0, "failures": 0}

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return f"retrying({self._inner.name})"

    def embed(self, texts):
        import time as _time
        delay = self._retry_delay
        last = None
        for attempt in range(1, self._max_retries + 1):
            try:
                # reset flaky counter semantics: FlakyEmbeddings counts every embed()
                # so we call it once per attempt via a thin wrapper
                self.stats["calls"] += 1
                return self._one_shot(texts)
            except Exception as exc:
                last = exc
                self.stats["failures"] += 1
                if _is_non_retryable_embed_error(exc):
                    raise
                if attempt < self._max_retries:
                    self.stats["retries"] += 1
                    if delay:
                        _time.sleep(delay)
                    delay *= 2
        raise last  # type: ignore[misc]

    def _one_shot(self, texts):
        return self._inner.embed(texts)


# ---------------------------------------------------------------------------
# 1. Offline eval metrics (measurement standard)
# ---------------------------------------------------------------------------

def test_offline_suite_reports_mean_min_max(suite_repo):
    metrics = run_offline_eval(suite_repo, k=3)
    assert metrics["n"] == len(EVAL_CASES)
    assert metrics["k"] == 3

    # Aggregate must be in [0, 1]
    for key in ("recall@k", "mrr", "hit@k"):
        assert 0.0 <= metrics[key] <= 1.0

    # Best / worst extremes are reported
    assert metrics["recall@k_max"] == 1.0
    assert metrics["mrr_max"] == 1.0
    assert metrics["recall@k_min"] == 0.0
    assert metrics["mrr_min"] == 0.0

    # Mean is strictly between floor and ceiling on this mixed suite
    assert metrics["recall@k_min"] < metrics["recall@k"] <= metrics["recall@k_max"]
    assert metrics["mrr_min"] < metrics["mrr"] <= metrics["mrr_max"]

    # Tag summaries: best queries should mostly hit; worst should miss
    assert metrics["by_tag"]["best"]["hit_rate"] >= 0.75
    assert metrics["by_tag"]["worst"]["hit_rate"] == 0.0
    assert metrics["by_tag"]["worst"]["recall@k_mean"] == 0.0
    assert metrics["by_tag"]["mid"]["recall@k_mean"] == 0.5


def test_offline_suite_best_cases_perfect(suite_repo):
    metrics = run_offline_eval(suite_repo, k=3)
    best_ids = {c["id"] for c in EVAL_CASES if c["tag"] == "best"}
    best_rows = [row for row in metrics["per_case"] if row.get("id") in best_ids]
    assert best_rows
    assert all(r["recall@k"] == 1.0 and r["mrr"] == 1.0 and r["hit"] for r in best_rows)


def test_evaluate_recall_exposes_per_case_and_hit(suite_repo):
    rag = RagRetriever(suite_repo, embeddings=HashingEmbeddings(), hybrid=True).build()
    metrics = evaluate_recall(rag, EVAL_CASES[:2], k=2)
    assert metrics["hit@k"] == 1.0
    assert len(metrics["per_case"]) == 2
    assert metrics["recall@k_min"] == metrics["recall@k_max"] == 1.0


# ---------------------------------------------------------------------------
# 2. Embedding API fault tolerance
# ---------------------------------------------------------------------------

def test_retrying_embeddings_recovers_after_transient_errors():
    flaky = FlakyEmbeddings(fail_times=2, dim=32)
    emb = RetryingEmbeddings(flaky, max_retries=3, retry_delay=0.0)
    vecs = emb.embed(["verify_jwt_token"])
    assert vecs.shape == (1, 32)
    assert emb.stats["retries"] == 2
    assert emb.stats["failures"] == 2


def test_retrying_embeddings_exhausted_raises():
    flaky = FlakyEmbeddings(fail_times=10, dim=32)
    emb = RetryingEmbeddings(flaky, max_retries=3, retry_delay=0.0)
    with pytest.raises(TimeoutError):
        emb.embed(["x"])
    assert emb.stats["retries"] == 2
    assert emb.stats["failures"] == 3


def test_non_retryable_auth_error_fails_fast():
    flaky = FlakyEmbeddings(fail_times=5, dim=32, error="401 unauthorized invalid api key")
    emb = RetryingEmbeddings(flaky, max_retries=5, retry_delay=0.0)
    with pytest.raises(TimeoutError, match="401"):
        emb.embed(["x"])
    # only one attempt — no retries on auth errors
    assert emb.stats["retries"] == 0
    assert emb.stats["failures"] == 1


def test_resilient_embeddings_fail_over_to_hashing():
    primary = FlakyEmbeddings(fail_times=100, dim=64)  # always fails
    resilient = ResilientEmbeddings(primary, HashingEmbeddings(dim=64))
    vecs = resilient.embed(["connect_socket host"])
    assert vecs.shape == (1, 64)
    assert resilient.using_fallback is True
    assert resilient.stats["failovers"] == 1
    # subsequent calls stay on fallback
    resilient.embed(["another"])
    assert resilient.stats["fallback_calls"] == 2
    assert resilient.stats["primary_calls"] == 1


def test_resilient_embeddings_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="dim"):
        ResilientEmbeddings(HashingEmbeddings(dim=32), HashingEmbeddings(dim=64))


def test_openai_embeddings_accepts_timeout_and_retry_kwargs():
    """Constructor wires timeout/retry without needing a live API (init may still need openai pkg)."""
    openai = pytest.importorskip("openai")
    emb = OpenAIEmbeddings(
        api_key="sk-test-not-used",
        timeout=12.0,
        max_retries=4,
        retry_delay=0.5,
    )
    assert emb._timeout == 12.0
    assert emb._max_retries == 4
    assert emb._retry_delay == 0.5
    assert emb.stats["calls"] == 0


# ---------------------------------------------------------------------------
# 3. Cache hit rate + performance regression markers
# ---------------------------------------------------------------------------

def test_cache_hit_rate_cold_warm_partial_and_invalidated(suite_repo, tmp_path):
    cache = tmp_path / "ragcache"
    emb = HashingEmbeddings(dim=128)

    cold = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    assert cold.stats["cache_hit_rate"] == 0.0
    assert cold.stats["cache_miss_rate"] == 1.0
    assert cold.stats["embedded_chunks"] == cold.stats["chunks"]
    cold_chunks = cold.stats["embedded_chunks"]

    warm = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    assert warm.stats["cache_hit_rate"] == 1.0
    assert warm.stats["cache_miss_rate"] == 0.0
    assert warm.stats["embedded_chunks"] == 0
    # Structural perf regression marker: warm path does zero re-embedding work
    assert warm.stats["embedded_chunks"] < cold_chunks

    # Partial invalidation: touch one file
    (suite_repo / "network.py").write_text("def connect_socket(host, port):\n    return None\n")
    partial = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    assert 0.0 < partial.stats["cache_hit_rate"] < 1.0
    assert partial.stats["reembedded_files"] == 1
    assert partial.stats["embedded_chunks"] >= 1

    # Full invalidation via embedding dim change
    rebuilt = RagRetriever(
        suite_repo, embeddings=HashingEmbeddings(dim=256), cache_dir=cache,
    ).build()
    assert rebuilt.stats["cache_hit_rate"] == 0.0
    assert rebuilt.stats["embedded_chunks"] == rebuilt.stats["chunks"]


def test_cache_miss_performance_regression_is_measurable(suite_repo, tmp_path):
    """
    On cache miss / invalidation, the system falls back to full (or partial) re-embed.
    We assert work markers, not wall-clock, so the test stays deterministic offline.
    """
    cache = tmp_path / "ragcache"
    emb = HashingEmbeddings(dim=64)

    warm = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    warm = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    assert warm.stats["cache_hit_rate"] == 1.0
    warm_work = warm.stats["embedded_chunks"]

    # Invalidate by editing all source files
    for name in ("auth.py", "parser.py", "network.py", "database.py"):
        p = suite_repo / name
        p.write_text(p.read_text() + "\n# touched\n")

    missed = RagRetriever(suite_repo, embeddings=emb, cache_dir=cache).build()
    assert missed.stats["cache_hit_rate"] < 1.0
    assert missed.stats["embedded_chunks"] > warm_work
    assert missed.stats["reembedded_files"] >= 4


# ---------------------------------------------------------------------------
# 4. Chunking + vectorization measurement standard
# ---------------------------------------------------------------------------

def test_chunking_and_embedding_specs(suite_repo):
    m = measure_chunking_and_embeddings(suite_repo)
    assert m["chunk_count"] >= 4
    # Suite files are small functions → avg well under the 40-line window / 120 max
    assert 1 <= m["avg_chunk_lines"] <= CHUNK_DESIGN["max_chunk_lines"]
    assert m["max_chunk_lines_observed"] <= CHUNK_DESIGN["max_chunk_lines"]
    assert m["avg_chunk_chars"] > 0
    assert m["embedding_dim_hashing"] == 256
    assert m["embedding_dim_openai_default"] == 1536
    assert m["vector_shape_sample"][1] == 256
    assert m["design"]["chunk_lines_default"] == 40
    assert m["design"]["overlap_default"] == 10
    assert "SyntaxChunker" in m["design"]["chunker"]
    assert "HashingEmbeddings" in m["libraries"]["in_house"]
    assert any("faiss" in x for x in m["libraries"]["optional_oss"])
    assert any("ast" in x for x in m["libraries"]["stdlib"])


# ---------------------------------------------------------------------------
# 5. Retrieval latency / QPS + recall
# ---------------------------------------------------------------------------

def test_retrieval_perf_and_recall(suite_repo):
    perf = measure_retrieval_perf(suite_repo, k=3, rounds=30, warmup=3)
    # Quality locked to suite
    assert perf["recall@k"] == 0.75
    assert perf["mrr"] == 0.8333
    assert perf["hit@k"] == 0.8333
    assert perf["recall@k_min"] == 0.0
    assert perf["recall@k_max"] == 1.0
    # Offline hashing on tiny corpus should be fast; loose bounds for CI machines
    assert perf["avg_latency_ms"] < 200.0
    assert perf["p95_latency_ms"] < 500.0
    assert perf["avg_qps"] > 20.0
    assert perf["peak_qps"] >= perf["avg_qps"]


# ---------------------------------------------------------------------------
# 6. Production incident drills
# ---------------------------------------------------------------------------

def test_incident_drills(suite_repo, tmp_path):
    drills = run_incident_drills(suite_repo, tmp_path / "cache")
    assert drills["retrieval_miss"]["metric"]["worst_recall@k"] == 0.0
    assert drills["retrieval_miss"]["metric"]["best_recall@k"] == 1.0
    assert drills["embedding_outage"]["metric"]["failovers"] == 1
    assert drills["embedding_outage"]["metric"]["using_fallback"] is True
    assert drills["cache_miss_storm"]["metric"]["warm_hit_rate"] == 1.0
    assert drills["cache_miss_storm"]["metric"]["warm_embedded_chunks"] == 0
    assert drills["cache_miss_storm"]["metric"]["invalidation_hit_rate"] == 0.0
    assert drills["cache_miss_storm"]["metric"]["invalidation_embedded_chunks"] > 0
    assert drills["context_blowup"]["metric"]["within_cap"] is True


def test_full_report_smoke(suite_repo):
    report = run_full_report(suite_repo)
    assert set(report) >= {
        "chunking_and_embeddings",
        "retrieval_perf",
        "offline_quality",
        "incident_drills",
    }
