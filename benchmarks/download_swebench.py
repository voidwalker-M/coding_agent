"""
benchmarks/download_swebench.py

Download a SWE-bench dataset split from the HuggingFace datasets-server and
cache it locally as gzipped JSONL (mirrors how HumanEval is stored under
benchmarks/data/). Uses the plain JSON `/rows` API so no heavyweight parquet /
`datasets` dependency is required.

Datasets (config is always "default"):
    princeton-nlp/SWE-bench_Lite       300 test + 23 dev   (default here)
    princeton-nlp/SWE-bench_Verified   500 test
    princeton-nlp/SWE-bench            ~2294 test

Examples:
    python -m benchmarks.download_swebench                       # Lite, test split
    python -m benchmarks.download_swebench --split dev           # Lite dev (23, fast)
    python -m benchmarks.download_swebench --dataset princeton-nlp/SWE-bench_Verified
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE = 100  # datasets-server max page size


def _dataset_slug(dataset: str) -> str:
    """A filesystem-safe stem, e.g. princeton-nlp/SWE-bench_Lite -> SWE-bench_Lite."""
    return dataset.split("/")[-1]


def default_cache_path(dataset: str, split: str) -> Path:
    return _DATA_DIR / f"{_dataset_slug(dataset)}_{split}.jsonl.gz"


def _fetch_page(dataset: str, split: str, offset: int, length: int) -> dict:
    qs = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": split,
         "offset": offset, "length": length}
    )
    url = f"{_ROWS_API}?{qs}"
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — retry any transient network error
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def download(dataset: str, split: str, out_path: Path | None = None,
             force: bool = False) -> Path:
    """Download all rows of `dataset`/`split` and write them as JSONL.gz."""
    out_path = out_path or default_cache_path(dataset, split)
    if out_path.exists() and not force:
        n = sum(1 for _ in gzip.open(out_path, "rt", encoding="utf-8"))
        print(f"  already cached: {out_path} ({n} instances). Use --force to refresh.")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = _fetch_page(dataset, split, 0, 1)
    total = first.get("num_rows_total")
    if not total:
        raise RuntimeError(f"dataset {dataset}/{split} reported 0 rows (wrong name?)")
    print(f"  downloading {dataset}/{split}: {total} instances -> {out_path}")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    written = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        offset = 0
        while offset < total:
            page = _fetch_page(dataset, split, offset, _PAGE)
            rows = page.get("rows", [])
            if not rows:
                break
            for entry in rows:
                fh.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
                written += 1
            offset += len(rows)
            print(f"    {written}/{total}", end="\r", flush=True)
    tmp.replace(out_path)
    print(f"\n  done: {written} instances written to {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a SWE-bench split to benchmarks/data/.")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                    help="HF dataset id (default: princeton-nlp/SWE-bench_Lite)")
    ap.add_argument("--split", default="test", choices=["test", "dev"],
                    help="split to download (default: test)")
    ap.add_argument("-o", "--output", default=None, help="output .jsonl.gz path")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    out = Path(args.output) if args.output else None
    print(f"\nSWE-bench downloader")
    download(args.dataset, args.split, out, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
