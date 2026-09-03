"""
eval/compare.py

Side-by-side comparison of two (or more) EvalReport JSON files.

Typical use: run the same suite with `--engine native` and `--engine langgraph`
(or with/without RAG, or two models) and diff process + outcome metrics.

    python -m eval.compare native.json langgraph.json
    python -m eval.compare --labels native,langgraph a.json b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"{path} is not an EvalReport JSON (missing 'summary')")
    return data


def compare_summaries(reports: dict[str, dict[str, Any]]) -> str:
    """Render a text table of summary metrics across named reports."""
    keys = [
        "success_rate", "pass_at_1_rate", "avg_steps", "avg_tokens",
        "total_cost_usd", "total_time", "total_llm_time", "total_tool_time",
        "total_tool_errors", "total_duplicate_actions", "avg_time_to_first_edit",
    ]
    names = list(reports)
    col_w = max(12, max(len(n) for n in names) + 1)
    header = f"{'metric':<28}" + "".join(f"{n:>{col_w}}" for n in names)
    lines = [header, "-" * len(header)]
    for key in keys:
        row = f"{key:<28}"
        for name in names:
            val = reports[name].get("summary", {}).get(key)
            if val is None:
                cell = "n/a"
            elif isinstance(val, float):
                cell = f"{val:.4g}" if abs(val) < 1 else f"{val:.2f}"
            else:
                cell = str(val)
            row += f"{cell:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare EvalReport JSON files")
    parser.add_argument("reports", nargs="+", help="EvalReport JSON paths")
    parser.add_argument("--labels", default="", help="Comma-separated names matching reports")
    args = parser.parse_args(argv)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if labels and len(labels) != len(args.reports):
        parser.error("--labels count must match the number of report files")
    named = {}
    for i, path in enumerate(args.reports):
        name = labels[i] if labels else Path(path).stem
        named[name] = load_report(path)
    print(compare_summaries(named))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
