# SWE-bench Lite — Evaluation Report

**Model:** `gpt-oss-120b` (UF LiteLLM proxy) · **Agent:** this repo's ReAct loop, `max_steps=40`
**Grading:** official SWE-bench "resolved" criterion — all `FAIL_TO_PASS` pass **and** all
`PASS_TO_PASS` stay passing · **Host:** no Docker; per-instance venvs built with `uv`

## Headline

> **3 / 14 = 21% resolved** (pass@1) on the locally-gradable subset of SWE-bench Lite.

| Metric | Value |
|---|---|
| Resolved | **3 / 14 (21%)** |
| Avg steps | 18.4 |
| Avg tokens / instance | 352,192 |
| Total cost | $49.31 |
| Total model time | 851 s (~14 min) |

### Why the denominator is 14, not 18

Every instance was first run with its **gold patch** applied. Only instances that grade
RESOLVED with the correct patch are trustworthy to score, because a failure there would
mean the *environment* is wrong, not the model. 4 of 18 were excluded on that basis:

| Excluded | Reason |
|---|---|
| `pallets__flask-4992` | gold patch doesn't pass locally (flask 2.3 env) |
| `pallets__flask-5063` | gold patch doesn't pass locally (flask 2.3 env) |
| `pylint-dev__pylint-7080` | gold patch doesn't pass locally |
| `pylint-dev__pylint-7114` | gold patch breaks PASS_TO_PASS (unstable baseline) |

So all 11 failures below are **genuine model misses**, not environment artifacts.

## Per-repo breakdown

| Repo | Resolved |
|---|---|
| mwaskom/seaborn | 1 / 4 |
| pydata/xarray | 1 / 5 |
| pylint-dev/pylint | 1 / 4 |
| pallets/flask | 0 / 1 |

## Failure taxonomy

| Category | Count | Meaning |
|---|---:|---|
| wrong-fix | 9 | Edited plausible code, but target tests still fail |
| resolved | 3 | — |
| max-steps | 1 | Exhausted all 40 steps without converging |
| regression | 1 | Fix broke previously-passing tests |

The dominant mode is **wrong-fix**: the model reliably finds the right *file* and forms a
plausible hypothesis, but mislocates the change. Canonical example — `flask-4045`: it
correctly decided blueprint names containing `.` must raise `ValueError`, but inserted the
check in `BlueprintSetupState` (registration time) instead of `Blueprint.__init__`
(construction time), so the test that constructs `Blueprint("a.b")` never raised.

## Per-instance results

| Instance | Result | Category | Steps | Tokens | Cost |
|---|---|---|---:|---:|---:|
| mwaskom__seaborn-2848 | failed | wrong-fix | 21 | 478,709 | $4.79 |
| mwaskom__seaborn-3010 | **RESOLVED** | — | 14 | 262,915 | $2.63 |
| mwaskom__seaborn-3190 | failed | wrong-fix | 18 | 398,007 | $3.98 |
| mwaskom__seaborn-3407 | failed | wrong-fix | 20 | 430,736 | $4.31 |
| pallets__flask-4045 | failed | wrong-fix | 15 | 237,176 | $2.37 |
| pydata__xarray-3364 | failed | wrong-fix | 15 | 237,330 | $2.37 |
| pydata__xarray-4094 | **RESOLVED** | — | 29 | 530,235 | $5.30 |
| pydata__xarray-4248 | failed | max-steps | 40 | 825,310 | $8.25 |
| pydata__xarray-4493 | failed | wrong-fix | 18 | 297,826 | $2.98 |
| pydata__xarray-5131 | failed | wrong-fix | 19 | 340,934 | $3.41 |
| pylint-dev__pylint-5859 | failed | regression | 11 | 185,134 | $1.85 |
| pylint-dev__pylint-6506 | failed | wrong-fix | 12 | 224,886 | $2.25 |
| pylint-dev__pylint-7228 | failed | wrong-fix | 15 | 299,078 | $2.99 |
| pylint-dev__pylint-7993 | **RESOLVED** | — | 11 | 182,418 | $1.82 |

`xarray-4094` is the strongest solve: it fixed the issue while keeping **862** regression
tests green — a correct, well-scoped change rather than a lucky edit.

## Agent deficiencies this evaluation exposed

Running the benchmark surfaced three defects that had to be fixed before the numbers meant
anything. All three are real agent bugs, not benchmark tuning:

1. **No surgical edit tool** (most severe). The agent could only *replace whole files*, so
   the model overwrote a 500-line `blueprints.py` with 11 characters (`import sys`).
   → Added `file_edit` (exact string replace with a uniqueness guard) in
   `tools/file_tool.py`. Unconditional fix — it was a genuine capability gap.
2. **Hallucinated completion.** The model declared *"File Modified: src/flask/blueprints.py"*
   while `git diff` was empty. → The loop now refuses a `finish` that changed nothing
   (`AgentConfig.require_edit_before_finish`, **off by default**; the eval opts in). It is
   git-aware, so non-git and no-change tasks still finish normally.
3. **Empty reasoning turns.** `gpt-oss-120b` frequently ends a turn after its analysis
   channel with no content or tool call; 2 resamples wasn't enough and it gave up mid-task.
   → `max_empty_retries` is now plumbed through the router; the eval uses 6. Default
   unchanged at 2.

Config-gating keeps `agent chat` / `agent run` behaviour identical to before
(full suite: 462 passed, 9 skipped).

## Caveats

- **Not leaderboard-comparable.** This is 14 instances of SWE-bench Lite's 300, on a
  no-Docker host. Published numbers use the official per-instance Docker images over the
  full set. Treat 21% as a signal for this subset, not an official score.
- **The agent cannot run the hidden tests.** Grading happens in a separate per-repo venv,
  so during the run the model works from the issue text and source alone — it cannot
  iterate against the failing test the way a Docker-based harness allows. This likely
  depresses the score, particularly for the 9 wrong-fix cases.
- **Cost is high** (~$3.5/instance, 352k tokens avg). Most of it is repeated repo-map and
  file-view context re-injection each step — an efficiency target for the agent.

## Reproduce

```bash
python -m benchmarks.download_swebench --split test
bash benchmarks/run_swebench_eval.sh          # gold-validate, then run the model
```

Raw data: `eval_runs/final_mock.json` (validation), `eval_runs/final_real.json` (results).
