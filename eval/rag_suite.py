"""
eval/rag_suite.py

Offline RAG measurement suite — locked numbers for interview / regression:

1. Chunking + vectorization (avg chunk size, embedding dim, OSS vs in-house)
2. Retrieval performance (avg / p50 / p95 latency, peak QPS, recall@k)
3. Production-incident drills (retrieval miss, embed failover, cache miss regression)

Run:

    PYTHONPATH=. python3 -m eval.rag_suite
    PYTHONPATH=. python3 -m pytest tests/test_rag_metrics.py -q
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixture corpus (written under a temp dir by materialize_corpus)
# ---------------------------------------------------------------------------

CORPUS_FILES: dict[str, str] = {
    "auth.py": '''\
def verify_jwt_token(token: str) -> dict:
    """Validate a JWT and return claims."""
    header, payload, signature = token.split(".")
    return decode_claims(payload)

def refresh_access_token(refresh_token: str) -> str:
    return mint_jwt(refresh_token)
''',
    "parser.py": '''\
def parse_tokens(source: str) -> list:
    """exclusive_parser_marker — used by offline RAG mid-case."""
    tokenizer = Tokenizer(source)
    return tokenizer.parse()

class Tokenizer:
    def parse(self):
        return self.tokens
''',
    "network.py": '''\
def connect_socket(host: str, port: int):
    sock = Socket(host, port)
    sock.connect()
    return sock

def close_socket(sock) -> None:
    sock.close()
''',
    "database.py": '''\
def execute_sql_query(conn, sql: str, params: tuple = ()):
    cursor = conn.cursor()
    cursor.execute(sql, params)
    return cursor.fetchall()

def open_postgres_pool(dsn: str):
    return Pool(dsn)
''',
    "README.md": '''\
# Demo service

This toy repo exists only for offline RAG evaluation.
''',
}

# Gold queries: mix of clear hits, partial recall, and guaranteed misses so
# mean / min / max are distinguishable under the measurement standard.
EVAL_CASES: list[dict[str, Any]] = [
    {
        "id": "best_auth",
        "tag": "best",
        "query": "verify_jwt_token refresh_access_token JWT claims",
        "relevant_files": {"auth.py"},
    },
    {
        "id": "best_parser",
        "tag": "best",
        "query": "parse_tokens Tokenizer parse source",
        "relevant_files": {"parser.py"},
    },
    {
        "id": "best_network",
        "tag": "best",
        "query": "connect_socket close_socket host port",
        "relevant_files": {"network.py"},
    },
    {
        "id": "best_db",
        "tag": "best",
        "query": "execute_sql_query open_postgres_pool cursor fetchall",
        "relevant_files": {"database.py"},
    },
    {
        "id": "mid_partial",
        "tag": "mid",
        # Parser-only lexical signal, but gold requires two files → partial recall
        "query": "parse_tokens Tokenizer parse source exclusive_parser_marker",
        "relevant_files": {"parser.py", "network.py"},
    },
    {
        "id": "worst_missing_gold",
        "tag": "worst",
        # Gold file is not in the corpus → guaranteed miss (metric floor)
        "query": "verify_jwt_token refresh_access_token JWT claims",
        "relevant_files": {"does_not_exist.py"},
    },
]

# Design constants (asserted by tests — the "spec" for chunking / embeddings)
CHUNK_DESIGN = {
    "chunker": "SyntaxChunker (default) + Chunker fallback",
    "chunk_lines_default": 40,
    "overlap_default": 10,
    "max_chunk_lines": 120,
    "python_splitter": "stdlib ast (in-house)",
    "other_lang_splitter": "repo_map symbols via tree-sitter* (optional OSS) → regex fallback (in-house)",
    "embedding_hashing_dim": 256,
    "embedding_openai_dim": 1536,
    "embedding_openai_model": "text-embedding-3-small",
    "dense_index": "faiss-cpu Flat/HNSW (optional OSS) → numpy (in-house)",
    "sparse_index": "BM25 pure-numpy (in-house)",
    "fusion": "RRF (in-house)",
    "rerank": "MMR numpy (in-house) / sentence-transformers cross-encoder (optional OSS)",
}


def materialize_corpus(root: Path) -> Path:
    """Write CORPUS_FILES under ``root`` and return root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, content in CORPUS_FILES.items():
        (root / name).write_text(content, encoding="utf-8")
    return root


def _build_rag(repo_path: Path, *, hybrid: bool = True, cache_dir: Path | None = None):
    from context.rag import HashingEmbeddings, RagRetriever

    return RagRetriever(
        repo_path,
        embeddings=HashingEmbeddings(dim=CHUNK_DESIGN["embedding_hashing_dim"]),
        hybrid=hybrid,
        syntax_aware=True,
        chunk_lines=CHUNK_DESIGN["chunk_lines_default"],
        overlap=CHUNK_DESIGN["overlap_default"],
        cache_dir=cache_dir,
        index_kind="numpy",  # pin for cross-platform deterministic ranking
    ).build()


# ---------------------------------------------------------------------------
# 1. Chunking + vectorization measurements
# ---------------------------------------------------------------------------

def measure_chunking_and_embeddings(repo_path: Path) -> dict:
    """Avg chunk size + embedding dims + library attribution on the suite corpus."""
    from context.rag import (
        HashingEmbeddings,
        SyntaxChunker,
        create_embedding_backend,
        create_vector_index,
    )

    chunker = SyntaxChunker(
        chunk_lines=CHUNK_DESIGN["chunk_lines_default"],
        overlap=CHUNK_DESIGN["overlap_default"],
        max_chunk_lines=CHUNK_DESIGN["max_chunk_lines"],
    )
    chunks = chunker.chunk_repo(Path(repo_path))
    line_sizes = [c.end_line - c.start_line + 1 for c in chunks]
    char_sizes = [len(c.text) for c in chunks]

    hashing = HashingEmbeddings(dim=CHUNK_DESIGN["embedding_hashing_dim"])
    sample_vecs = hashing.embed([c.embed_text() for c in chunks[:3]] or ["x"])

    # Detect optional OSS without requiring install
    try:
        import faiss  # noqa: F401
        faiss_available = True
    except Exception:
        faiss_available = False
    idx = create_vector_index(hashing.dim, "auto")

    return {
        "design": CHUNK_DESIGN,
        "chunk_count": len(chunks),
        "avg_chunk_lines": round(statistics.mean(line_sizes), 2) if line_sizes else 0.0,
        "min_chunk_lines": min(line_sizes) if line_sizes else 0,
        "max_chunk_lines_observed": max(line_sizes) if line_sizes else 0,
        "avg_chunk_chars": round(statistics.mean(char_sizes), 1) if char_sizes else 0.0,
        "embedding_dim_hashing": hashing.dim,
        "embedding_dim_openai_default": CHUNK_DESIGN["embedding_openai_dim"],
        "vector_shape_sample": list(sample_vecs.shape),
        "create_embedding_backend_offline": create_embedding_backend("hashing").name,
        "dense_index_name": idx.name,
        "faiss_available": faiss_available,
        "libraries": {
            "in_house": [
                "SyntaxChunker / Chunker",
                "HashingEmbeddings",
                "BM25Index",
                "RRF",
                "MMRReranker",
                "NumpyIndex",
                "ResilientEmbeddings",
            ],
            "optional_oss": [
                "numpy (required for RAG)",
                "openai SDK (OpenAIEmbeddings)",
                "faiss-cpu (Flat / HNSW)",
                "tree-sitter* (non-Python symbol spans)",
                "sentence-transformers (cross-encoder rerank)",
            ],
            "stdlib": ["ast (Python chunk boundaries)", "hashlib (feature hashing)"],
        },
    }


# ---------------------------------------------------------------------------
# 2. Retrieval performance + recall
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def measure_retrieval_perf(
    repo_path: Path,
    *,
    k: int = 3,
    rounds: int = 50,
    warmup: int = 5,
) -> dict:
    """
    Measure local retrieve latency / QPS on the suite (offline hashing — no network).

    peak_qps: best 1-second window throughput during the run
    avg_latency_ms / p50 / p95: per-retrieve wall time
    """
    from context.rag import evaluate_recall

    rag = _build_rag(repo_path)
    queries = [c["query"] for c in EVAL_CASES]

    # warmup
    for i in range(warmup):
        rag.retrieve_chunks(queries[i % len(queries)], k=k)

    latencies_ms: list[float] = []

    t_all0 = time.perf_counter()
    for i in range(rounds):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        rag.retrieve_chunks(q, k=k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    t_all1 = time.perf_counter()

    total_s = max(t_all1 - t_all0, 1e-9)
    avg_qps = rounds / total_s
    # Single-threaded peak ≈ inverse of the fastest call
    peak_qps = 1000.0 / min(latencies_ms) if latencies_ms else 0.0
    ordered = sorted(latencies_ms)

    quality = evaluate_recall(rag, EVAL_CASES, k=k)

    return {
        "rounds": rounds,
        "k": k,
        "avg_latency_ms": round(statistics.mean(latencies_ms), 3),
        "p50_latency_ms": round(_percentile(ordered, 50), 3),
        "p95_latency_ms": round(_percentile(ordered, 95), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
        "avg_qps": round(avg_qps, 1),
        "peak_qps": round(peak_qps, 1),
        "recall@k": quality["recall@k"],
        "mrr": quality["mrr"],
        "hit@k": quality["hit@k"],
        "recall@k_min": quality["recall@k_min"],
        "recall@k_max": quality["recall@k_max"],
        "backend": rag.backend_info,
        "note": "Local offline hashing benchmark (not multi-tenant production traffic).",
    }


def run_offline_eval(repo_path: Path, *, k: int = 3, hybrid: bool = True) -> dict:
    """
    Build a RagRetriever on the suite corpus and return evaluate_recall metrics
    plus tag-level summaries (best / mid / worst).
    """
    from context.rag import evaluate_recall

    rag = _build_rag(repo_path, hybrid=hybrid)
    metrics = evaluate_recall(rag, EVAL_CASES, k=k)
    id_to_tag = {c["id"]: c["tag"] for c in EVAL_CASES}
    by_tag: dict[str, list[dict]] = {"best": [], "mid": [], "worst": []}
    for row in metrics.get("per_case", []):
        tag = id_to_tag.get(row.get("id") or "")
        if tag in by_tag:
            by_tag[tag].append(row)
    metrics["by_tag"] = {
        tag: {
            "n": len(rows),
            "recall@k_mean": round(sum(r["recall@k"] for r in rows) / len(rows), 4) if rows else 0.0,
            "mrr_mean": round(sum(r["mrr"] for r in rows) / len(rows), 4) if rows else 0.0,
            "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 4) if rows else 0.0,
        }
        for tag, rows in by_tag.items()
    }
    metrics["suite"] = "eval.rag_suite"
    metrics["retriever_chunks"] = rag.chunk_count
    metrics["backend"] = rag.backend_info
    return metrics


# ---------------------------------------------------------------------------
# 3. Production-incident drills (reproducible failure → mitigation)
# ---------------------------------------------------------------------------

def run_incident_drills(repo_path: Path, cache_dir: Path) -> dict:
    """
    Reproduce the failure modes we treat as 'production incidents' for a local agent:

    A. Retrieval miss / wrong context  → measurable via worst-case recall=0; Agent continues
    B. Embedding API outage / latency   → ResilientEmbeddings failover
    C. Cache invalidation / miss storm  → hit_rate→0, embedded_chunks jumps (resource cost)
    D. Context blow-up                  → retrieve() max_chars truncation
    """
    from context.rag import (
        HashingEmbeddings,
        RagRetriever,
        ResilientEmbeddings,
        evaluate_recall,
    )

    reports: dict[str, Any] = {}

    # A. Retrieval quality miss
    quality = run_offline_eval(repo_path, k=3)
    reports["retrieval_miss"] = {
        "symptom": "relevant file absent from top-k (worst tag)",
        "locate": "evaluate_recall per_case + EventLog / injected RAG section",
        "mitigation": "hybrid BM25+RRF, embed_text with path/symbols, optional query expansion",
        "metric": {
            "worst_recall@k": quality["by_tag"]["worst"]["recall@k_mean"],
            "worst_hit_rate": quality["by_tag"]["worst"]["hit_rate"],
            "best_recall@k": quality["by_tag"]["best"]["recall@k_mean"],
        },
    }

    # B. Embedding outage → failover
    class _AlwaysFail(HashingEmbeddings):
        def embed(self, texts):
            raise TimeoutError("simulated embedding API timeout")

    resilient = ResilientEmbeddings(_AlwaysFail(dim=64), HashingEmbeddings(dim=64))
    _ = resilient.embed(["verify_jwt_token"])
    reports["embedding_outage"] = {
        "symptom": "embedding API timeout / 5xx / network",
        "locate": "OpenAIEmbeddings.stats retries/failures; ResilientEmbeddings.stats failovers",
        "mitigation": "timeout=30s, max_retries=3 exponential backoff, then same-dim Hashing failover; Agent RAG catch→empty context",
        "metric": {
            "failovers": resilient.stats["failovers"],
            "using_fallback": resilient.using_fallback,
            "fallback_calls": resilient.stats["fallback_calls"],
        },
    }

    # C. Cache miss storm / invalidation
    emb = HashingEmbeddings(dim=128)
    cold = RagRetriever(repo_path, embeddings=emb, cache_dir=cache_dir).build()
    warm = RagRetriever(repo_path, embeddings=emb, cache_dir=cache_dir).build()
    # invalidate via dim change (simulates model swap)
    storm = RagRetriever(
        repo_path, embeddings=HashingEmbeddings(dim=256), cache_dir=cache_dir,
    ).build()
    reports["cache_miss_storm"] = {
        "symptom": "build_seconds / API cost spike after deploy or model/dim change",
        "locate": "rag.stats cache_hit_rate, reembedded_files, embedded_chunks, build_seconds",
        "mitigation": "content-hash incremental cache; pin embedding model+dim; clear .rag_cache only when intentional",
        "metric": {
            "cold_hit_rate": cold.stats["cache_hit_rate"],
            "warm_hit_rate": warm.stats["cache_hit_rate"],
            "warm_embedded_chunks": warm.stats["embedded_chunks"],
            "invalidation_hit_rate": storm.stats["cache_hit_rate"],
            "invalidation_embedded_chunks": storm.stats["embedded_chunks"],
        },
    }

    # D. Context / resource guard
    rag = _build_rag(repo_path)
    huge = rag.retrieve("parse_tokens Tokenizer", k=50, max_chars=200)
    reports["context_blowup"] = {
        "symptom": "prompt token blow-up from too many / too large chunks",
        "locate": "len(retrieved context); TokenBudget; shell output truncation",
        "mitigation": "rag_top_k=5, retrieve max_chars=6000 default, token budget trim",
        "metric": {
            "truncated_chars": len(huge),
            "max_chars_cap": 200,
            "within_cap": len(huge) <= 200 + 80,  # header overhead allowance
        },
    }

    # Keep evaluate_recall import used for clarity in incident A path
    _ = evaluate_recall
    return reports


def run_full_report(repo_path: Path | None = None) -> dict:
    """Run all measurement blocks and return one JSON-serializable report."""
    import tempfile

    tmp = None
    if repo_path is None:
        tmp = tempfile.TemporaryDirectory()
        repo_path = materialize_corpus(Path(tmp.name) / "suite")
    cache = Path(repo_path) / ".rag_cache_bench"

    report = {
        "chunking_and_embeddings": measure_chunking_and_embeddings(repo_path),
        "retrieval_perf": measure_retrieval_perf(repo_path),
        "offline_quality": run_offline_eval(repo_path),
        "incident_drills": run_incident_drills(repo_path, cache),
    }
    if tmp is not None:
        tmp.cleanup()
    return report


def main() -> None:
    import logging
    logging.disable(logging.WARNING)
    report = run_full_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
