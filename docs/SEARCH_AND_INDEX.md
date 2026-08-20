# File Search & Indexing — design and optimizations (#3)

**Goal.** Let the agent (and the retrieval layer) *find the right code fast* without
burning tokens re-reading files. This is a layered system: cheap exact tools for
"I know the name", a structural map for "what's in this repo", a semantic retriever
for "code related to this idea", and — new in this work — a persistent **symbol
index** so definition lookups are indexed instead of re-scanned on every call.

Files: [context/symbol_index.py](../context/symbol_index.py) (new),
[tools/search_tool.py](../tools/search_tool.py), [context/repo_map.py](../context/repo_map.py),
[context/rag.py](../context/rag.py). Tests:
[tests/test_symbol_index.py](../tests/test_symbol_index.py) (11).

---

## The four layers (pick the cheapest that answers the question)

| Layer | Answers | Structure | Cost |
|---|---|---|---|
| **Exact tools** (`search_text`, `find_files`) | "line/​file containing X" | none (walk + regex) | O(files) per call |
| **Symbol index** (`find_symbol`) | "where is `foo` defined?" | inverted `name → [loc]` | **O(1) after one build** |
| **Repo-map** | "what's in this repo?" | ranked symbol summary | built once/task |
| **RAG** (`retriever`) | "code related to this idea" | dense + sparse index | built once, incremental |

The agent is told (system prompt) to prefer targeted tools over dumping whole files —
the layers exist so it rarely needs to read a file just to locate something.

---

## The symbol index (the new optimization)

### Problem

`find_symbol` (and grep-style `search_text`) previously **re-walked the whole repo
and re-ran a regex over every file on every call**. In a loop that looks up dozens of
symbols that is O(files × calls) of repeated disk reads and scanning — the same files
parsed again and again. It was also Python-only (`^def` / `^class` regex).

### Design — `SymbolIndex`

Parse each file **once** into a `name → [SymbolLoc]` map, then answer every later
lookup from memory:

```
_by_name  : exact name       -> [SymbolLoc]   # O(1) exact lookup
_by_lower : lowercased name   -> [SymbolLoc]   # case-insensitive
_names    : sorted distinct names             # prefix / substring scan
```

`lookup(symbol, mode="exact|prefix|substring", kind=…, path_prefix=…, case_sensitive=…)`
returns definition sites ordered **top-level defs first** (usually the intended
target), then by path/line. `FindSymbolTool(index=…)` uses it when present and falls
back to the original regex scan when not (so the tool still works standalone / in tests).

### Optimizations baked in

1. **Build-once, lookup-O(1).** The dominant win: N lookups cost one parse, not N
   scans. No disk access at query time.
2. **Incremental content-hash caching.** Each file's symbols are cached under
   `<repo>/.symbol_cache/symbols.json` keyed by a SHA-256 of its bytes (mirroring the
   RAG cache). A rebuild re-parses **only files whose content changed**; unchanged
   files reuse cached symbols; deleted/renamed files drop out (the index is rebuilt
   from the surviving set each build, so stale entries can't linger). `stats` reports
   `reused_files` vs `reparsed_files`.
3. **Accurate, multi-language extraction.** Reuses `repo_map._extract_symbols`, i.e.
   **tree-sitter** when the language pack is installed (functions/classes/methods
   across Python/JS/TS/Go/Rust/Java/C/C++/Ruby), **regex fallback** otherwise — a
   strict upgrade over the old Python-only regex.
4. **Skip noise.** Reuses `_SKIP_DIRS` (`.git`, `__pycache__`, `node_modules`, …) and
   a per-file byte cap, so vendored/build trees never enter the index.
5. **Bounded results.** `limit` caps output after ordering, protecting the context
   window.

Wired in the CLI (`_build_symbol_index`) so `find_symbol` is indexed on every run;
failures degrade silently to the regex scan.

---

## The existing index layers (recap of what they optimize)

### Repo-map (`context/repo_map.py`)
A compressed, **query-aware** structural summary injected into the prompt. Files are
ranked by `(query relevance, structural importance)` — task-relevant files lead (★),
so scarce token budget isn't spent on globally-"important" but irrelevant files.
tree-sitter symbols with a regex fallback; trimmed to a token budget.

### RAG hybrid retriever (`context/rag.py`)
Semantic code retrieval, every layer with a pure-numpy/stdlib fallback:

- **Chunking** — syntax-aware (`ast` for Python, symbol boundaries elsewhere),
  line-window fallback; chunks annotated with language + symbols.
- **Dense index** — faiss `IndexHNSWFlat` (ANN) or `IndexFlatIP`, degrading to a
  numpy dot-product; normalized vectors so inner product == cosine.
- **Sparse index** — pure-numpy **BM25** for exact identifier/keyword matching (often
  more reliable than dense for code).
- **Fusion** — dense + sparse ranked lists merged by **reciprocal-rank-fusion**.
- **Reranking** — optional MMR (numpy, diversity) or a cross-encoder.
- **Incremental persistence** — chunks/vectors/per-file hashes cached; only changed
  files are re-chunked and re-embedded; unchanged reused; deleted removed.
- **Metadata filtering** — by language / path prefix.

---

## Optimization summary (what and why)

| Optimization | Where | Payoff |
|---|---|---|
| Build-once inverted symbol index | symbol_index | O(1) lookups vs O(files) per call |
| Content-hash incremental cache | symbol_index, rag | re-parse/re-embed only changed files |
| tree-sitter + regex fallback | repo_map (reused) | accurate, multi-language, no hard dep |
| Skip-dirs + byte cap | all layers | never index vendored/build noise |
| HNSW ANN dense index | rag | sub-linear vector search at scale |
| BM25 sparse + RRF fusion | rag | exact-identifier recall + robustness |
| Query-aware repo-map ranking | repo_map | task-relevant files survive the budget |
| MMR / cross-encoder rerank | rag | diverse, precise top-k |
| Result caps + per-task caching | tools, core | bounded, cache-friendly context |

## Scale-out path

- **Shard the symbol cache** — one JSON per file keyed by path hash, or move the
  inverted index into SQLite/LMDB; keep only hot files resident.
- **Persist symbols in a DB** — a `symbols(name, kind, file, line)` table with an
  index on `name` gives prefix queries via `LIKE 'foo%'` without holding all names
  in RAM.
- **Vector store** — swap the in-process faiss index for a server (Qdrant/pgvector),
  memmap + quantize (int8/PQ) vectors at very large corpora.
- **Watch-based incremental** — drive rebuilds from a filesystem watcher / git hook
  instead of a full re-scan per run.
