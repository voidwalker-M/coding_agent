"""
eval/ — agent evaluation framework (benchmark harness).

Runs the agent on a set of verifiable tasks, uses independent verifiers to determine
success or failure, and aggregates metrics (success rate / steps / tokens / time)
into a reproducible evaluation report.

Core principle: success is determined by an independent verifier (re-run tests /
check files / run commands), NOT by the agent's self-reported FINISH —
that is the entire point of a benchmark.
"""

from eval.harness import (
    EvalHarness,
    EvalReport,
    EvalResult,
    TaskSpec,
)
from eval.verifiers import (
    CommandVerifier,
    FileContainsVerifier,
    FileExistsVerifier,
    PytestVerifier,
    Verifier,
)

__all__ = [
    "EvalHarness",
    "EvalReport",
    "EvalResult",
    "TaskSpec",
    "Verifier",
    "PytestVerifier",
    "FileContainsVerifier",
    "FileExistsVerifier",
    "CommandVerifier",
]
