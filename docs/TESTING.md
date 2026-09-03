# Tests

All agent tests live in `tests/` (not under `eval_runs/` or `.venv/`). Run them
from the repo root with the project venv:

```bash
pip install -e ".[dev]"
.venv/bin/python -m pytest            # full suite
.venv/bin/python -m pytest tests/test_skills.py -q
```

Use `python -m pytest` (or `.venv/bin/python -m pytest`) rather than a bare
`pytest`/`python` on PATH — several tools invoke `sys.executable -m pytest`.

## Optional extras

| Extra | Install | Tests that need it |
| ----- | ------- | ------------------ |
| RAG (numpy; faiss optional) | `pip install -e ".[rag]"` | `test_rag.py`, `test_rag_metrics.py`, `test_rag_external.py` |
| LangGraph | `pip install -e ".[langgraph]"` | `test_langgraph.py` (module skipped otherwise) |
| FastAPI / MCP | `pip install -e ".[server]"` | `test_api.py`, `test_mcp.py` (MCP skipped without `mcp`) |
| Playwright | `pip install -e ".[browser]"` then `playwright install chromium` | not required; `web_fetch engine=http` is stdlib. Playwright path is optional. |

Skipped tests are expected when an extra is missing (LangGraph, MCP SDK, Docker,
Playwright). Failures that mention `python: command not found` usually mean PATH
has no `python` — the venv interpreter is enough for the suite itself.

## Inventory (`tests/test_*.py`)

| File | What it covers |
| ---- | -------------- |
| `test_day1.py` | Task / EventLog dataclasses, replay, loop helpers, `summarize_run` |
| `test_day2.py` | Native ReAct loop with MockBackend: finish, give-up, max-steps, loops, reflection, plan injection |
| `test_day3.py` | File / search / shell / pytest / git tools on a real filesystem |
| `test_day4.py` | LLM router, response parsing, prompt templates, vLLM-without-key |
| `test_day5.py` | Repo-map, token budget, conversation history |
| `test_day6.py` | Config schema + Click CLI (no live GitHub) |
| `test_day7.py` | Retry, git diff, repo-map cache, e2e-ish MockBackend runs |
| `test_plan.py` | Structured `Plan` + `plan` tool create/complete |
| `test_decision.py` | Loop abort, no-op finish reject, reflection, edit-tool counter |
| `test_workflow.py` | Closed loop: ticket → run → verify pass/fail |
| `test_trajectory.py` | Process metrics: tool errors, duplicate actions, time-to-first-edit |
| `test_compare.py` | `eval.compare` summary table |
| `test_eval.py` | Eval harness + independent verifier (ignores self-reported FINISH) |
| `test_orchestrator.py` | Multi-agent topologies: pipeline, pair, autonomous |
| `test_rules.py` | `AGENTS.md` / `.agent/rules` loader (always-apply vs catalog) |
| `test_skills.py` | Skill catalog / always-apply / `skill` tool / prompt injection |
| `test_browser_tool.py` | `web_fetch` HTML extract, local HTTP GET, `file://` blocked |
| `test_memory.py` | Short-term window + long-term facts, remember/recall tools |
| `test_memory_integration.py` | Memory wired through the agent loop |
| `test_checkpoint.py` | Checkpoint save/load and resume |
| `test_compaction.py` | History compaction |
| `test_undo.py` | Snapshot undo + registry wiring |
| `test_timing.py` | LLM vs tool timing on the event log |
| `test_latency_metrics.py` | TTFT / E2E latency fields |
| `test_token_efficiency.py` | Token-budget / efficiency helpers |
| `test_symbol_index.py` | Incremental symbol index |
| `test_model_router.py` | Difficulty / cascade model router |
| `test_rag.py` | Chunking, hashing embeddings, retrieve + prompt injection |
| `test_rag_metrics.py` | recall@k / MRR / hit@k / latency / cache (needs numpy) |
| `test_rag_external.py` | External-source RAG path |
| `test_langgraph.py` | LangGraph engine vs native MockBackend scripts |
| `test_stream.py` | Streaming LLM callback |
| `test_sandbox.py` | Docker runtime (skips if Docker unavailable) |
| `test_confirm.py` | Write-op confirmation callback |
| `test_chat.py` | Multi-round chat session |
| `test_web.py` | Stdlib SSE web chat page + MockBackend round |
| `test_api.py` | FastAPI `agent serve` |
| `test_mcp.py` | MCP server exposes registry tools |

Related (not under `tests/`): `eval/suite.py` (coding eval tasks), `eval/rag_suite.py`
(RAG measurement standard), `benchmarks/` (HumanEval / SWE-bench Lite).

## Last full run

Recorded **2026-08-20** on this machine (`darwin`, Python 3.12.13, `.venv`, extras
`rag` / `langgraph` **not** installed, Docker **not** running):

```
.venv/bin/python -m pytest -q
# 572 passed, 11 skipped
```

| | Count |
| --- | --- |
| Collected | 579 |
| Passed | 572 |
| Skipped at import | 4 modules (`test_rag.py`, `test_rag_metrics.py`, `test_rag_external.py`, `test_langgraph.py`) |
| Skipped at runtime | 7 Docker integration tests in `test_sandbox.py` |
| Failed | 0 |

New in this pass: `tests/test_skills.py` (10) and `tests/test_browser_tool.py` (6).

Install extras to run the skipped modules:

```bash
pip install -e ".[rag]"         # RAG tests (needs numpy)
pip install -e ".[langgraph]"   # LangGraph engine tests
```

Recompute:

```bash
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q
```
