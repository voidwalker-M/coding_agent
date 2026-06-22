"""
eval/suite.py

Built-in evaluation suite: a set of small, verifiable programming tasks.

Each task uses an independent verifier for grading:
- bugfix_*      : fix a bug so that pytest passes
- implement_*   : implement a stub function so that pytest passes
- create_*      : create a file from scratch and satisfy behavioral requirements

These tasks are solvable by a real LLM; the harness's own tests drive them via MockBackend scripts.
"""

from __future__ import annotations

from eval.harness import TaskSpec
from eval.verifiers import (
    AllOfVerifier,
    CommandVerifier,
    FileContainsVerifier,
    PytestVerifier,
    UnmodifiedFilesVerifier,
)


def default_suite() -> list[TaskSpec]:
    return [
        # 1. Bug fix: add() is implemented as subtraction
        TaskSpec(
            id="bugfix_add",
            description=(
                "There is a bug in calc.py: the add() function returns the wrong "
                "result and tests are failing. Fix calc.py so that the tests pass. "
                "Run the tests to verify."
            ),
            setup_files={
                "calc.py": "def add(a, b):\n    return a - b\n",
                "test_calc.py": (
                    "from calc import add\n\n"
                    "def test_add():\n"
                    "    assert add(2, 3) == 5\n"
                    "    assert add(0, 0) == 0\n"
                    "    assert add(-1, 1) == 0\n"
                ),
            },
            verify=PytestVerifier(),
            max_steps=15,
        ),

        # 2. Implement stub: factorial
        TaskSpec(
            id="implement_factorial",
            description=(
                "Implement the factorial(n) function in mathutils.py so that the "
                "existing tests in test_mathutils.py pass. Run the tests to verify."
            ),
            setup_files={
                "mathutils.py": (
                    "def factorial(n):\n"
                    "    # TODO: implement\n"
                    "    raise NotImplementedError\n"
                ),
                "test_mathutils.py": (
                    "from mathutils import factorial\n\n"
                    "def test_factorial():\n"
                    "    assert factorial(0) == 1\n"
                    "    assert factorial(1) == 1\n"
                    "    assert factorial(5) == 120\n"
                ),
            },
            verify=PytestVerifier(),
            max_steps=15,
        ),

        # 3. Create a runnable script from scratch
        TaskSpec(
            id="create_hello",
            description=(
                "Create a Python script named hello.py that prints exactly "
                "'hello world' (lowercase) when run with `python hello.py`."
            ),
            setup_files={},
            verify=AllOfVerifier(
                CommandVerifier("python hello.py", expect_substring="hello world"),
            ),
            max_steps=10,
        ),

        # 4. Add a function to an existing module
        TaskSpec(
            id="implement_is_palindrome",
            description=(
                "Add an is_palindrome(s) function to strutils.py that returns True if "
                "the string reads the same forwards and backwards (case-insensitive, "
                "ignoring spaces). Make the tests in test_strutils.py pass."
            ),
            setup_files={
                "strutils.py": "# add is_palindrome here\n",
                "test_strutils.py": (
                    "from strutils import is_palindrome\n\n"
                    "def test_palindrome():\n"
                    "    assert is_palindrome('racecar')\n"
                    "    assert is_palindrome('A man a plan a canal Panama')\n"
                    "    assert not is_palindrome('hello')\n"
                ),
            },
            verify=AllOfVerifier(
                FileContainsVerifier("strutils.py", "def is_palindrome"),
                PytestVerifier(),
            ),
            max_steps=15,
        ),

        # 5. Multi-file navigation: the bug is in ONE of several modules.
        #    Exercises the repo-map (find the right file) AND penalizes
        #    over-editing (the agent must not touch the distractor modules).
        TaskSpec(
            id="bugfix_multi_file",
            description=(
                "A store package has a bug: format_price() in store/format.py is "
                "supposed to turn an integer number of cents into a dollar string, "
                "e.g. 1050 -> '$10.50', but it is wrong and test_format.py fails. "
                "Find and fix the bug in store/format.py only. Do not modify any "
                "other module. Run the tests to verify."
            ),
            setup_files={
                "store/__init__.py": "",
                "store/inventory.py": (
                    "def in_stock(item, qty):\n"
                    "    return qty > 0\n\n"
                    "def restock(item, qty):\n"
                    "    return qty\n"
                ),
                "store/cart.py": (
                    "def subtotal(prices):\n"
                    "    return sum(prices)\n\n"
                    "def item_count(items):\n"
                    "    return len(items)\n"
                ),
                # The bug: returns raw cents, not dollars.
                "store/format.py": (
                    "def format_price(cents):\n"
                    "    return f\"${cents}\"\n"
                ),
                "test_format.py": (
                    "from store.format import format_price\n\n"
                    "def test_format_price():\n"
                    "    assert format_price(1050) == '$10.50'\n"
                    "    assert format_price(99) == '$0.99'\n"
                    "    assert format_price(100) == '$1.00'\n"
                ),
            },
            verify=AllOfVerifier(
                PytestVerifier(),
                UnmodifiedFilesVerifier("store/inventory.py", "store/cart.py"),
            ),
            max_steps=20,
        ),
    ]
