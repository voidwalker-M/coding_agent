# Agent Memory — short-term + long-term, and how the store scales (#2)

**Goal.** Give the agent memory that survives beyond one raw conversation window:
a **short-term** working memory within a run, and a **long-term** memory that
persists across runs and sessions. The design follows how Claude Code itself does
memory (markdown files + a cache-friendly index + on-demand recall — see
<https://learn.shareai.run/en/s09/>) and adds a stronger retrieval engine on top.

Files: [context/memory.py](../context/memory.py), [context/history.py](../context/history.py)
(the `on_evict` hook), [tools/memory_tool.py](../tools/memory_tool.py),
wired in [agent/core.py](../agent/core.py). Tests:
[tests/test_memory.py](../tests/test_memory.py) (16). Pure-Python — no numpy required.

---

## Why memory (the problem)

The ReAct loop already has `ConversationHistory` — a sliding window. It is **lossy**:
when the window overflows, the oldest turns are dropped outright, and the model
forgets what it already did. Compression/compaction loses detail too. Memory is the
layer that *keeps what compression throws away* — "use tabs not spaces" survives as
an exact fact, not a degraded paraphrase.

## Two tiers

### Short-term (working / session memory) — `ShortTermMemory`

A small, bounded scratchpad for one run, rendered into the system prompt each step:

- **notes** — distilled facts the agent accumulates (bounded working set);
- **files examined** — an ordered, de-duplicated set (so it stops re-opening files
  after the window trims — this also feeds the prompt's efficiency guidance);
- **rolling summary** — the key upgrade: `ConversationHistory(on_evict=…)` now hands
  each *evicted* turn to `ShortTermMemory.fold()`, which folds it into a bounded
  summary. Nothing is silently lost across a window trim. Discarded at end of run.

### Long-term (persistent) memory — `LongTermMemory`

Memory that outlives the process. Two flavors of record:

- **episodic** — a captured past run (`Task X → files changed → outcome=success`),
  written automatically at end of run by `core._capture_episode`;
- **semantic / typed** — a fact written explicitly via the `remember` tool. Typed
  like Claude Code: `user` (preference), `feedback` (constraint), `project`
  (background), `reference` (navigation), plus `semantic`.

---

## Storing many records — the on-disk design

This is the crux of "how do you design when you store many files."

### Default: one markdown file per memory + a `MEMORY.md` index

```
<mem_dir>/
  MEMORY.md                      # the index — one line per memory
  repo-uses-pytest-markers.md    # a record: YAML frontmatter + markdown body
  parser-is-recursive-descent.md
  .memory_meta.json              # consolidation bookkeeping (last run, count)
```

Each record is human-readable and hand-editable:

```markdown
---
name: repo-uses-pytest-markers
description: Slow tests are marked @pytest.mark.slow; CI runs -m "not slow"
kind: reference
tags: [testing, pytest]
files: [tests/conftest.py]
importance: 0.6
created_at: 1700000000.0
last_access: 1700000000.0
access_count: 3
---
The repo marks slow tests with @pytest.mark.slow …
```

**Why this layout:**

- **The index stays in the system prompt, the bodies don't.** `MEMORY.md` is small
  and stable, so it can live in the prompt and stay **prompt-cache friendly**; the
  full body of a record is injected **on demand** only when it's relevant. The
  catalog stays cheap; you borrow a file only when needed.
- **Human-readable & editable.** A user can open, edit, or delete a memory by hand.
- **Crash-safe writes.** Each record is one file; the index rewrite is atomic. A
  half-written record never corrupts the store.
- **De-dup by content hash.** Identical content *reinforces* the existing record
  (bumps `access_count`/`importance`) instead of creating a duplicate.

### Retrieval — two paths, upgraded

1. **Index path** — `index_block()` returns the `MEMORY.md` catalog for the prompt
   (always present, ordered by salience so a truncated catalog keeps the best).
2. **On-demand path** — `select(query, k)` / `recall(query, k)` rank records and
   inject the top few bodies (capped per record). Ranking is a real engine:
   - an **inverted index** (`token → record names`) + **idf** lexical score —
     O(query-terms), not a scan of every record;
   - **recency decay** (`0.5 ** (age / half_life)`) × **importance**, with a small
     boost for frequently-recalled records (recall reinforces, resists decay);
   - optional **dense embeddings** (reusing the RAG embedding backend) fused with
     the lexical score via **reciprocal-rank-fusion** — but only when numpy is
     present, else lexical-only;
   - an optional **LLM selector** hook (`select(…, selector=fn)`) for the
     "ask a cheap model which memories matter" side-query, with the engine as the
     always-available fallback.

### Consolidation (reflection) — bounding growth

`maybe_consolidate()` is **gated** like Claude Code's real implementation — it fires
only when *all* hold: a **file-count threshold**, a **minimum time interval** since
the last pass, and a **minimum number of new records**. `consolidate()` then dedups
by content, applies decay, caps to `max_records` (evicting the lowest
effective-importance records, deleting their files), and rewrites the index. An
optional `reducer(records)` hook lets an LLM merge/resolve contradictions; without
it the pass is deterministic.

### Scale-out (hundreds of thousands of records)

A flat directory eventually hurts the filesystem and a full in-memory index gets
heavy. The same dataclass shards cleanly:

- **Shard files** — `records/<first-2-hex-of-id>/<name>.md` (`shard_path()`), so no
  single directory holds more than ~1/256 of the corpus.
- **Externalize the index** — move the inverted index + metadata into SQLite/LMDB;
  keep only hot shards resident.
- **Memmap vectors** — store dense vectors in a memmap'd `vectors.npy` (as the RAG
  cache does) instead of holding them all in RAM; quantize (int8/PQ) at large scale.
- **Tier by recency** — hot recent memories in the fast index, cold ones on disk,
  paged in on a miss (MemGPT-style).

The shipped implementation is the single-directory + `MEMORY.md` version, which is
the right size for a per-repo / per-user coding agent.

---

## Wiring & usage

Memory is **off by default** and every path is inert when disabled (`long_term_memory
is None`). Enable it per run:

```bash
agent run -t "fix the parser bug" --memory                 # store in <repo>/.agent_memory
agent run -t "…" --memory --memory-dir ~/.agent/mem        # custom location
```

When on, `core.py` injects `recall(task)` + the `MEMORY.md` index into the system
prompt (cached per task), renders short-term working memory each step, registers the
`remember` / `recall` tools, and captures an episodic memory at end of run. Dense
recall is used automatically when numpy is available, else the lexical engine.
