#!/usr/bin/env bash
# benchmarks/run_swebench_eval.sh
#
# Two-phase SWE-bench Lite evaluation of the coding agent, run sequentially in a
# single process (no subagents needed; same-repo instances reuse one cached venv):
#
#   Phase 1 (no API cost): apply each instance's GOLD patch and grade it. Only
#     instances that grade RESOLVED here are locally gradable, so a later failure
#     is a genuine agent miss rather than an environment artifact.
#   Phase 2: run the configured model on exactly those validated instances.
#
# Usage:  bash benchmarks/run_swebench_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."

set -a; source .env 2>/dev/null; set +a
PY=.venv/bin/python

INSTANCES="mwaskom__seaborn-2848,mwaskom__seaborn-3010,mwaskom__seaborn-3190,mwaskom__seaborn-3407,\
pallets__flask-4045,pallets__flask-4992,pallets__flask-5063,\
pydata__xarray-3364,pydata__xarray-4094,pydata__xarray-4248,pydata__xarray-4493,pydata__xarray-5131,\
pylint-dev__pylint-5859,pylint-dev__pylint-6506,pylint-dev__pylint-7080,pylint-dev__pylint-7114,\
pylint-dev__pylint-7228,pylint-dev__pylint-7993"

echo "############ PHASE 1: gold-patch validation (no model calls) ############"
$PY -u -m benchmarks.run_swebench --mock --instances "$INSTANCES" \
    --test-timeout 240 \
    --results-dir eval_runs/final_mock -o eval_runs/final_mock.json

GRADABLE=$($PY - <<'PY'
import json
try:
    d = json.load(open("eval_runs/final_mock.json"))
    print(",".join(r["task_id"] for r in d["results"] if r["passed"]))
except Exception:
    print("")
PY
)

echo
echo "############ GRADABLE INSTANCES ############"
echo "${GRADABLE:-<none>}"
echo

if [ -z "$GRADABLE" ]; then
    echo "No instance is locally gradable — skipping the model run."
    exit 1
fi

echo "############ PHASE 2: real model run ############"
$PY -u -m benchmarks.run_swebench --instances "$GRADABLE" \
    --max-steps 40 --test-timeout 240 --keep \
    --results-dir eval_runs/final_real -o eval_runs/final_real.json

echo
echo "############ DONE ############"
