"""
entry/load_test.py

Simple concurrent load test against the FastAPI Agent service.

Usage:
  # terminal 1
  agent serve --port 8766 --repo .

  # terminal 2 (with MockBackend via test harness, or real API)
  python -m entry.load_test --url http://127.0.0.1:8766 --concurrency 8 --requests 24
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[float, dict]:
    t0 = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (time.perf_counter() - t0) * 1000, body


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_load_test(
    base_url: str,
    *,
    concurrency: int = 4,
    requests: int = 20,
    repo_path: str = ".",
    poll_timeout_s: float = 120.0,
) -> dict:
    submit_url = f"{base_url.rstrip('/')}/v1/tasks"
    latencies_ms: list[float] = []
    ok = 0
    fail = 0
    lock = threading.Lock()

    def one_request(_: int) -> None:
        nonlocal ok, fail
        try:
            submit_ms, sub = _post_json(submit_url, {
                "description": "load test ping",
                "repo_path": repo_path,
                "max_steps": 2,
            })
            job_id = sub["job_id"]
            deadline = time.time() + poll_timeout_s
            while time.time() < deadline:
                st = _get_json(f"{base_url.rstrip('/')}/v1/tasks/{job_id}")
                if st["state"] in ("success", "failed", "interrupted"):
                    break
                time.sleep(0.05)
            total_ms = submit_ms  # submit RTT; poll time excluded for API throughput metric
            with lock:
                latencies_ms.append(total_ms)
                ok += 1
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            with lock:
                fail += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one_request, i) for i in range(requests)]
        for f in as_completed(futs):
            f.result()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    ordered = sorted(latencies_ms)

    return {
        "requests": requests,
        "concurrency": concurrency,
        "ok": ok,
        "fail": fail,
        "elapsed_s": round(elapsed, 3),
        "qps": round(requests / elapsed, 2),
        "submit_p50_ms": round(_percentile(ordered, 50), 2) if ordered else None,
        "submit_p95_ms": round(_percentile(ordered, 95), 2) if ordered else None,
        "submit_avg_ms": round(statistics.mean(ordered), 2) if ordered else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Load test the Agent FastAPI service")
    p.add_argument("--url", default="http://127.0.0.1:8766")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--repo", default=".")
    args = p.parse_args()
    report = run_load_test(
        args.url, concurrency=args.concurrency, requests=args.requests, repo_path=args.repo,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
