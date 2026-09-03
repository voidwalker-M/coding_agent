# Agent memory — ChatGPT / Claude shaped

Coding agents do **not** dump the whole chat into every prompt. ChatGPT and
Claude keep three layers; this repo follows that split.

| Layer | What it is | Where | In the prompt? |
|---|---|---|---|
| **Thread** | Last *n* user queries (working window) | `ShortTermMemory` | Yes — compact window |
| **Chat history** | Older turns, kept searchable | SQLite `stm_turns` + `kind=conversation` snippets | Only top-k hits |
| **Saved memories** | Short extracted facts (prefs, identity, project notes) | SQLite `ltm_records` | Top-k, always-on |
| **Rules / Skills** | Instructions *you* wrote | `AGENTS.md`, `.cursor/rules`, `.agent/skills` | Rules always-on; skills on demand |
| **Code index** | The repo itself | RAG — not dialogue memory | Retrieved snippets |

Files: [context/memory.py](../context/memory.py),
[context/memory_extract.py](../context/memory_extract.py),
[context/memory_store.py](../context/memory_store.py).

---

## What ChatGPT / Claude actually do

They do **not** embed every utterance and stuff a vector dump into the system
prompt (the interview “encode → store → retrieve → inject” sketch). Product
behavior is closer to:

1. **Thread** — the open chat stays in the context window. When it no longer
   fits, older turns leave the *prompt*, not the product.
2. **Saved Memory** — a background pass extracts short durable facts
   (“prefers concise answers”, “name is …”). Those are the only items injected
   as “known about the user” on later turns.
3. **Chat history search** — older conversations remain stored. A later
   “what did I say about the API key?” retrieves a few snippets; it does not
   replay the whole log.

Overflowing a 10-query window must **not** delete the transcript. This agent
keeps `stm_turns`, extracts facts on each turn (`observe_turn`), and on window
overflow writes searchable `conversation` snippets (`ingest_overflow`). Prompt
injection is still top-k (`recall(..., for_prompt=True)`), skipping noisy
episodic *run logs*.

Retrieval is hybrid lexical (identifier-aware, BM25-like IDF) plus optional
dense embeddings, with recency weighting (14-day half-life). Postgres +
pgvector / Redis are the hosted store; SQLite is the default. No Pinecone.

---

## Short-term = the conversation window

STM is the last *n* user queries **in the prompt**. Bind a store so the full
transcript survives overflow and process restart. Chat / Web own one STM
across rounds (`window_queries`, default 10).

## Memories = small facts, not transcripts

Explicit `remember` is **approved immediately**. `propose()` stages a **pending**
memory when `memory.auto_approve` is false. Episodic run logs
(`record_episode`) stay searchable but are skipped in `for_prompt=True`.

Identity (user / visibility / ACL) still exists for multi-user hosts; the
default CLI path is one user after `agent login`, `auto_approve: true`.

---

## Config

```yaml
memory:
  enabled: false
  window_queries: 10
  auto_approve: true
  top_k: 4
  default_user: default
  default_role: agent
```

```bash
agent login
agent chat --repo .
```
