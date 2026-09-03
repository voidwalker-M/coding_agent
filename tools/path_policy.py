"""
tools/path_policy.py

Path guards shared by write/edit tools.

SWE-bench grading applies a *hidden* test_patch after the agent finishes. Edits
under tests/ therefore cannot help FAIL_TO_PASS and often fight the official
patch (agents rewrite or delete fixtures). Blocking those writes forces the
model to change library code instead.
"""

from __future__ import annotations

from pathlib import Path


def is_test_path(path: str | Path) -> bool:
    """True for test packages / test modules the agent must not modify on SWE."""
    if not path:
        return False
    parts = [p.lower() for p in Path(str(path)).parts]
    if any(p in ("tests", "test") for p in parts):
        return True
    name = Path(str(path)).name.lower()
    return name.startswith("test_") or name.endswith("_test.py")


TEST_EDIT_BLOCKED_MSG = (
    "Refusing to modify test files. On this task you must fix the *library* "
    "code that implements the bug — the evaluation suite applies its own hidden "
    "tests after you finish. Use search_text / find_symbol to locate the "
    "production symbol from the issue, then file_edit that file."
)
