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
| **Eval/observability** | none | `retriever.stats` (chunks, reused/re-embedded files, timings) + `evaluate_recall()` (recall@k / MRR / hit@k) |

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

print(rag.stats)            # {'chunks':.., 'reused_files':.., 'reembedded_files':.., ...}
ctx = rag.retrieve("fix the parser bug", k=5, language="python")

metrics = evaluate_recall(rag, [
    {"query": "parser tokenizer", "relevant_files": {"parser.py"}},
], k=5)
```

## Optional dependencies (all degrade gracefully)

- `faiss-cpu` — Flat/HNSW vector index (else numpy dot-product)
- `sentence-transformers` — cross-encoder reranker (else MMR)
- `tree-sitter*` — multi-language symbol boundaries (else regex; Python always uses stdlib `ast`)
- `OPENAI_API_KEY` — real embeddings (else offline hashing embeddings)

## Scope note

The corpus is the **current repository only** — no external/multi-repo code DB, no internet.
Extending to an external corpus (library docs, other repos) means adding a second source feeding
the same chunk → embed → index path; the persistence layer here is the prerequisite for that.
```
