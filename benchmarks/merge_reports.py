"""Merge sharded HumanEval worker reports into one combined report."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path

paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/full_w*.json"))
results = []
for p in paths:
    results.extend(json.load(open(p))["results"])
# de-dup by task_id, keep first
seen, uniq = set(), []
for r in results:
    if r["task_id"] in seen:
        continue
    seen.add(r["task_id"]); uniq.append(r)
results = sorted(uniq, key=lambda r: int(r["task_id"].split("_")[1]))

n = len(results)
passed = sum(1 for r in results if r["passed"])
summary = {
    "total": n,
    "passed": passed,
    "success_rate": round(passed / n, 4) if n else 0.0,
    "avg_steps": round(sum(r["steps"] for r in results) / n, 2) if n else 0,
    "avg_tokens": round(sum(r["tokens"] for r in results) / n, 1) if n else 0,
    "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
    "total_time": round(sum(r["elapsed"] for r in results), 2),
}
out = {"summary": summary, "results": results}
Path("benchmarks/humaneval_full_report.json").write_text(json.dumps(out, indent=2))

print(f"HumanEval FULL — {n} problems")
print(f"  pass@1        : {passed}/{n} = {summary['success_rate']:.1%}")
print(f"  avg_steps     : {summary['avg_steps']}")
print(f"  total_cost    : ${summary['total_cost_usd']}")
print(f"  total_time    : {summary['total_time']}s (summed across workers)")

fails = [r for r in results if not r["passed"]]
print(f"\nFailures: {len(fails)}")
for r in fails:
    print(f"  {r['task_id']:<40} status={r['agent_status']:<10} detail={r['detail'][:60]}")
