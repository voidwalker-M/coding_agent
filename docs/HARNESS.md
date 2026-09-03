# Harness Engineering

The job of a harness here is: **describe a task → run an agent that may edit
code → independently grade the result → compare runs**. That is the same loop as
“需求描述 / 代码生成 / 审查评估”, implemented for a coding agent rather than a
frontend IDE plugin.

```
task spec (eval/suite.py, HumanEval, SWE-bench, or a Ticket)
        │
        ▼
isolated repo  ──►  Agent.run  (native or LangGraph)
        │                 │
        │                 ├── EventLog (actions, observations, reflections)
        │                 └── git diff
        ▼
independent Verifier  (re-run tests / command — ignores the model's FINISH)
        │
        ▼
EvalReport JSON  ──►  eval.compare  (outcome + trajectory)
```

## Commands

```bash
# Built-in coding suite (pass@1 / steps / tokens / $ / process metrics)
agent eval --engine native --output native.json
agent eval --engine langgraph --output langgraph.json
python -m eval.compare --labels native,langgraph native.json langgraph.json

# RAG retrieval quality (recall@k / MRR / hit@k / latency) — needs numpy
pytest tests/test_rag_metrics.py
# or: python -c "from eval.rag_suite import run_offline_eval; run_offline_eval()"

# Closed loop: ticket → run → verify → optional memory write
# GitHub Issue path uses agent/workflow.py ClosedLoop
python -m entry.github_issue --repo owner/repo --issue 42 --local-path /tmp/myrepo
```

## What is graded

| Kind | Source | Metrics |
| ---- | ------ | ------- |
| **Outcome** | `eval/harness.py` + `eval/verifiers.py` | pass@1, steps, tokens, estimated $, LLM vs tool time |
| **Process** | `eval/trajectory.py` | tool-error rate, consecutive duplicate actions, reflection count, time-to-first-edit |
| **Retrieval** | `context/rag.py` `evaluate_recall` / `eval/rag_suite.py` | recall@k, MRR, hit@k, latency |

The verifier **does not trust** the agent's self-reported `FINISH`. A model that
says “done” without a passing test still fails the harness.

## Production loop (not just eval)

`agent/workflow.py` `ClosedLoop` is the same shape for a real ticket: run the
agent, optionally re-verify with a command, optionally persist a lesson to
long-term memory. GitHub Issue → branch → PR is one instance.

Decision policy (`agent/decision.py`) is part of the harness: loop abort, reject
no-op finishes, reflection nudges after failed tests / long exploration.

## Tests

- `tests/test_eval.py` — harness + independent verifier
- `tests/test_trajectory.py` — process metrics
- `tests/test_compare.py` — report comparison
- `tests/test_workflow.py` — closed loop pass/fail
- `tests/test_decision.py` — loop / finish / reflection policy
- `tests/test_rag_metrics.py` — RAG measurement standard (needs `pip install -e ".[rag]"`)

Do **not** invent pass@k or 风控 accuracy numbers in docs. Generate them with
`agent eval` / `python -m benchmarks.run_swebench` and paste the report.
