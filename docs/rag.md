# RAG retrieval pipeline

`context/rag.py` indexes the **target repository's own files** (code + markdown/text/YAML)
and retrieves the chunks most relevant to the task, injecting them into the system prompt.
It is distinct from `repo_map` (static, always-injected symbol summary): RAG is a *dynamic,
relevance-ranked* second source.

Every layer has a pure-numpy / stdlib fallback, so it runs fully offline (no API key, no
faiss, no tree-sitter) and upgrades automatically when those libraries are present.

## Pipeline

```
files ─▶ SyntaxChunker ─▶ EmbeddingBackend ─┬─▶ dense VectorIndex ─┐
                                            └─▶ BM25Index ─────────┤─▶ RRF fusion
                                                                   │
                                            metadata filter ◀──────┘
                                                  │
                                            Reranker (MMR / cross-encoder)
                                                  │
                                              top-k ─▶ system prompt
        ▲
        └── persistent cache + incremental update (re-embed only changed files)
```

## Demo-grade → industrial: what changed

| Dimension | Before | Now |
|---|---|---|
| **Chunking** | fixed 40-line windows | `SyntaxChunker`: Python via stdlib `ast` at def/class boundaries; other langs via repo_map symbols (tree-sitter→regex); line-window fallback |
| **Retrieval** | dense (cosine) only | **hybrid** dense + pure-numpy **BM25**, fused with **Reciprocal Rank Fusion** |
| **Reranking** | none | numpy **MMR** (diversity) by default-available; optional **cross-encoder** (sentence-transformers) |
| **Persistence** | rebuilt in memory every run | on-disk cache (`<repo>/.rag_cache`) with **incremental update** — content-hash manifest, re-embed only changed files, drop deleted |
| **Index** | faiss Flat / numpy | adds faiss **HNSW** (ANN) option via `index_kind="hnsw"` |
| **Metadata** | none | filter by `language` / `path_prefix` |
| **Query** | embed task once | optional **query expansion** (identifier sub-tokens), multi-query fusion |
| **Eval/observability** | none | `retriever.stats` (chunks, reused/re-embedded files, **cache_hit_rate**, timings) + `evaluate_recall()` (recall@k / MRR / hit@k + **min/max/per_case**) + `eval/rag_suite.py` offline measurement standard |
| **Embedding fault tolerance** | bare API call | **timeout** (default 30s) + **exponential retry** (default 3) + **ResilientEmbeddings** same-dim Hashing failover; Agent RAG errors degrade to empty context |

## Usage

```bash
# hybrid retrieval (dense + BM25), persistent incremental cache on by default
agent run --repo . --task "fix the parser bug" --retriever rag

# add reranking
agent run --repo . --task "..." --retriever rag --rerank mmr
agent run --repo . --task "..." --retriever rag --rerank cross-encoder   # needs sentence-transformers
```

```python
from context.rag import RagRetriever, evaluate_recall

rag = RagRetriever(
    "/repo",
    hybrid=True,             # dense + BM25 + RRF
    index_kind="hnsw",       # ANN (falls back to flat/numpy)
    reranker="mmr",          # or "cross-encoder", or None
    cache_dir="/repo/.rag_cache",   # persistence + incremental
).build()

print(rag.stats)            # includes cache_hit_rate, build_seconds, ...
ctx = rag.retrieve("fix the parser bug", k=5, language="python")

from eval.rag_suite import run_offline_eval, materialize_corpus
# Or run the locked measurement suite:
#   pytest tests/test_rag_metrics.py -q
```

## Measurement standard (offline)

Locked suite: `eval/rag_suite.py` (6 labeled queries, hybrid Hashing+BM25, **k=3**).

### How to run

```bash
# deps for RAG metrics
pip install "numpy>=1.21" pytest
# optional: pip install faiss-cpu openai

# full JSON report (chunking + latency/QPS + recall + incident drills)
PYTHONPATH=. python3 -m eval.rag_suite

# automated assertions (CI)
PYTHONPATH=. python3 -m pytest tests/test_rag_metrics.py -q
```

Full pytest inventory (including RAG and eval harness): [TESTING.md](TESTING.md).
RAG tests skip collection when numpy is missing (`pip install -e ".[rag]"`).
Harness loop: [HARNESS.md](HARNESS.md).

### 0. Chunking + vectorization (implementation + measured sizes)

| Item | Value |
|------|-------|
| Chunker | **In-house** `SyntaxChunker` (default) + `Chunker` line-window fallback |
| Python split | **stdlib `ast`** |
| Other languages | optional OSS **tree-sitter*** → in-house regex |
| Design window | **40** lines, overlap **10**, hard max **120** lines |
| Suite measured avg | **~3.0 lines / ~94 chars** per chunk (small fixture functions) |
| Embedding (offline) | In-house **HashingEmbeddings**, dim **256** |
| Embedding (API) | OSS **openai** SDK, `text-embedding-3-small`, dim **1536** |
| Dense index | optional OSS **faiss-cpu** (Flat/HNSW) → in-house **numpy** |
| Sparse / fusion / rerank | In-house **BM25** + **RRF**; MMR in-house / cross-encoder optional OSS |

### 1. recall@k / MRR / hit@k (suite baseline)

| Metric | Mean | Min (worst) | Max (best) |
|--------|------|-------------|------------|
| **recall@k** | **0.75** | **0.0** | **1.0** |
| **MRR** | **0.8333** | **0.0** | **1.0** |
| **hit@k** | **0.8333** | **0.0** (worst tag) | **1.0** (best tag) |

By difficulty tag:

| Tag | n | recall@k | MRR | hit rate | What it represents |
|-----|---|----------|-----|----------|--------------------|
| **best** | 4 | 1.0 | 1.0 | 1.0 | Unique identifier queries |
| **mid** | 1 | 0.5 | 1.0 | 1.0 | Partial multi-file gold |
| **worst** | 1 | 0.0 | 0.0 | 0.0 | Gold file absent from corpus |

### 2. Retrieval performance (local offline microbench)

Measured by `measure_retrieval_perf` (50 retrieves, hashing+numpy, suite corpus).  
**Not** multi-tenant production traffic — numbers vary by machine; CI asserts loose bounds.

| Metric | Typical on this suite (example run) | CI bound |
|--------|--------------------------------------|----------|
| **avg latency** | **~0.03–1 ms** | `< 200 ms` |
| **p50 / p95** | **~0.03 / ~0.04 ms** | p95 `< 500 ms` |
| **avg QPS** | **thousands–tens of thousands** (tiny index) | `> 20` |
| **peak QPS** | `1000 / min_latency_ms` (single-threaded) | `≥ avg_qps` |
| **recall@k / MRR / hit@k** | **0.75 / 0.8333 / 0.8333** | exact |

### 3. Embedding API bottleneck — degradation / fault tolerance

| Measure | Default / behavior |
|---------|-------------------|
| HTTP **timeout** | **30s** (`OpenAIEmbeddings(timeout=...)`) |
| **Retries** | **3** attempts, exponential back-off (`retry_delay` starts at **1.0s**) |
| Non-retryable | 401 / 403 / 400 / invalid API key — fail immediately |
| **Resilient failover** | After primary exhaustion → same-dim `HashingEmbeddings` (`ResilientEmbeddings`) |
| Init fallback | No key / client init error → Hashing only |
| Agent-level | RAG exception → empty context; task continues |
| Vector **cache** | `<repo>/.rag_cache` incremental (content-hash); batch size **128** |

### 4. Incremental cache hit rate + miss regression

Reported in `rag.stats["cache_hit_rate"]` = `reused_files / files`.

| Scenario | cache_hit_rate | embedded_chunks | Performance back-off |
|----------|----------------|-----------------|----------------------|
| Cold build | **0.0** | = all chunks | Full embed + index build |
| Warm, no edits | **1.0** | **0** | Near-zero embed work |
| One file changed | **(0, 1)** e.g. ~0.8 on 5-file suite | only changed file | Partial re-embed |
| Dim / backend change | **0.0** | = all chunks | Full rebuild (cache invalidated) |

### 5. Production-incident drills (`run_incident_drills`)

| Incident | How we locate | Mitigation | Suite metric |
|----------|---------------|------------|--------------|
| Retrieval miss / wrong context | `evaluate_recall` per_case + EventLog | hybrid BM25+RRF, symbol-aware `embed_text` | worst recall/hit = **0**; best = **1.0** |
| Embedding timeout / outage | embed `stats.retries/failures`, resilient `failovers` | timeout+retry → Hashing failover; Agent empty RAG | `failovers == 1` |
| Cache miss storm / model swap | `cache_hit_rate`, `embedded_chunks`, `build_seconds` | pin model/dim; incremental hash cache | warm hit **1.0** → invalidation hit **0.0** |
| Context / resource blow-up | retrieved char length, TokenBudget | `top_k=5`, `max_chars=6000` | truncated context stays within cap |

## Optional dependencies (all degrade gracefully)

- `faiss-cpu` — Flat/HNSW vector index (else numpy dot-product)
- `sentence-transformers` — cross-encoder reranker (else MMR)
- `tree-sitter*` — multi-language symbol boundaries (else regex; Python always uses stdlib `ast`)
- `OPENAI_API_KEY` — real embeddings (else offline hashing embeddings)

## Scope note

The corpus is the **current repository only** — no external/multi-repo code DB, no internet.
Extending to an external corpus (library docs, other repos) means adding a second source feeding
the same chunk → embed → index path; the persistence layer here is the prerequisite for that.
