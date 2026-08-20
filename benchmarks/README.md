# Benchmarks

Two objective, agent-driven benchmarks, both run through the same
`eval.harness.EvalHarness` (independent verification — the agent's self-reported
success is ignored; grading is an objective re-run).

| Benchmark | What it tests | Grading |
|-----------|---------------|---------|
| **HumanEval** | single-function code completion | re-run pytest on the function |
| **SWE-bench Lite** | fix a real GitHub issue across a real repo (read → plan → edit → test → repair) | official "resolved" criterion: all `FAIL_TO_PASS` pass **and** all `PASS_TO_PASS` stay passing |

---

## HumanEval

```bash
python -m benchmarks.run_humaneval --mock --limit 10          # pipeline check, no API key
python -m benchmarks.run_humaneval --limit 20 -o benchmarks/humaneval_report.json
```

---

## SWE-bench Lite

SWE-bench exercises the *whole* agent loop on real repositories: it is given only
a GitHub issue and the repo checked out at the buggy commit, and must locate and
fix the bug in the source. A hidden test suite (the instance's `test_patch`,
which the agent never sees) then grades the fix.

### 1. Download the dataset

```bash
python -m benchmarks.download_swebench --split test     # SWE-bench Lite, 300 instances
python -m benchmarks.download_swebench --split dev       # 23 instances (fast)
```

Cached under `benchmarks/data/SWE-bench_Lite_{split}.jsonl.gz`. Other variants:
`--dataset princeton-nlp/SWE-bench_Verified` (500) or `princeton-nlp/SWE-bench` (full).

### 2. Validate the pipeline (no API key, no LLM)

`--mock` applies each instance's **gold patch** instead of calling a model, so it
proves the whole clone → edit → apply hidden tests → grade pipeline works on real
data and that the grader scores a correct fix as *resolved*:

```bash
python -m benchmarks.run_swebench --mock \
    --instances pallets__flask-4045,pylint-dev__pylint-5859
```

Confirmed on this host (gold patches, uv-provisioned Python 3.9 + pinned deps):

```
[RESOLVED ] pallets__flask-4045      resolved: F2P 2/2 ok, P2P 50 passed
[RESOLVED ] pylint-dev__pylint-5859  resolved: F2P 1/1 ok, P2P 10 passed
```

(Note: `pytest-dev/pytest` instances are finicky to grade locally — pytest running
its own test suite hits import/conftest quirks — so prefer the other repos for
smoke checks.)

### 3. Evaluate the model

```bash
export GPT_OSS_API_KEY=...                      # or your configured provider's key
python -m benchmarks.run_swebench \
    --repos psf/requests,pallets/flask --limit 4 \
    -o benchmarks/swebench_report.json
```

Useful flags: `--instances <id,id>` (exact instances), `--repos <repo,repo>`
(filter), `--limit/--start` (slice), `-k` (pass@k), `--max-steps`, `-m/-p`
(model/provider override), `-R rag` (retrieval), `--keep` (keep workdirs).

---

## ⚠️ No-Docker caveat (read this before quoting numbers)

The **official** SWE-bench harness runs every instance inside a per-instance
**Docker image** that pins the exact OS, Python interpreter, and dependency
versions. This machine has no Docker, so grading here is **best-effort on the
host**:

- Each instance gets its own virtualenv built with **`uv`**, using the **Python
  version and pinned dependencies** the official harness specifies
  (`benchmarks/data/swebench_specs.json`, extracted from the `swebench` package).
  This is what lets old checkouts (e.g. Flask 2.0 needing `Werkzeug==2.3.7`)
  actually import and run under a modern host.
- Grading uses the official **"resolved"** criterion, run via pytest node ids.

What this means for the numbers:

- **Directly gradable locally:** the pytest-based repos — `requests`, `flask`,
  `pytest`, `pylint`, `seaborn`, and (when their C-extensions build) `astropy`,
  `matplotlib`, `scikit-learn`, `xarray`.
- **Reported as `UNSUPPORTED`, not graded:** `django` (`./tests/runtests.py`),
  `sympy` (`bin/test`), `sphinx` (`tox`) — their official runners aren't pytest
  and aren't reproduced without the Docker harness.
- **Reported as `ENV_ERROR`:** an instance whose host environment couldn't be
  built (a C-extension that won't compile, an interpreter `uv` can't provide).

The runner separates these categories in its summary and reports the resolve
rate **among locally-gradable instances**, so environment gaps are never silently
counted as agent failures. During a real run the agent also does not share the
per-repo grading venv, so it cannot execute the hidden tests itself — it works
from the issue text and the repo source (a known limitation of the no-Docker
setup). For publishable, leaderboard-comparable numbers, generate patches here
and grade them with the official Docker harness or `sb-cli`.

### Caches

Heavy, regenerable artifacts live under the git-ignored `eval_runs/` tree:
`eval_runs/swebench_cache/repos/` (one clone per repo) and
`eval_runs/swebench_cache/venvs/` (one venv per repo+version). Override the root
with `SWEBENCH_CACHE=/path`. Delete `venvs/` to force clean environment rebuilds.
