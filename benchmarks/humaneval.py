"""
benchmarks/humaneval.py

Adapter: turn the official HumanEval dataset (164 Python function-completion
problems) into `eval.harness.TaskSpec` objects so the existing agent eval
harness can grade them with an independent PytestVerifier.

Per problem we lay down two files in the isolated task repo:
    solution.py       -> the HumanEval `prompt` (signature + docstring; a valid,
                         importable stub whose body is only the docstring)
    test_solution.py  -> imports the entry point, pastes the HumanEval `test`
                         (which defines check(candidate)), and calls it under a
                         pytest-collected function.

Grading is objective: re-run pytest. The agent's self-reported FINISH is ignored
(the whole point of the harness). pass@1 here is directly comparable to the
published HumanEval pass@1 numbers.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from eval.harness import TaskSpec
from eval.verifiers import PytestVerifier

_DATA = Path(__file__).parent / "data" / "HumanEval.jsonl.gz"


def load_records(path: str | Path = _DATA) -> list[dict]:
    """Load the raw HumanEval JSONL records."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it first:\n"
            "  curl -sL -o benchmarks/data/HumanEval.jsonl.gz "
            "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
        )
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _test_file(entry_point: str, test_src: str) -> str:
    """Wrap the HumanEval `test` block (defines check()) as a pytest test."""
    return (
        f"from solution import {entry_point}\n\n"
        f"{test_src}\n\n"
        f"def test_humaneval():\n"
        f"    check({entry_point})\n"
    )


def record_to_spec(rec: dict, max_steps: int = 12) -> TaskSpec:
    """Convert one HumanEval record into a gradable TaskSpec."""
    entry = rec["entry_point"]
    task_num = rec["task_id"].split("/")[-1]
    return TaskSpec(
        id=f"humaneval_{task_num}_{entry}",
        description=(
            f"Implement the Python function `{entry}` in the file solution.py.\n\n"
            f"solution.py already contains the function signature and its docstring "
            f"(with examples), but the body is empty. Write a correct implementation "
            f"so that the tests in test_solution.py pass. Do not change the function's "
            f"name or signature. Run pytest to verify your work before finishing."
        ),
        setup_files={
            "solution.py": rec["prompt"],
            "test_solution.py": _test_file(entry, rec["test"]),
        },
        verify=PytestVerifier(),
        max_steps=max_steps,
    )


def canonical_solution_file(rec: dict) -> str:
    """The full correct solution.py content = prompt + canonical_solution.

    Used only by the mock dry-run to validate the harness/adapter without an LLM.
    """
    return rec["prompt"] + rec["canonical_solution"]


def load_humaneval(
    limit: int | None = None,
    start: int = 0,
    max_steps: int = 12,
) -> list[TaskSpec]:
    """Return HumanEval problems [start : start+limit] as TaskSpecs (in order)."""
    recs = load_records()
    sliced = recs[start : (start + limit) if limit is not None else None]
    return [record_to_spec(r, max_steps=max_steps) for r in sliced]
