"""Tests for the bug corpus.

Verifies that:
1. The corpus has the right structure and counts.
2. Every buggy_code actually fails its test (we don't want "bugs" that pass).
3. Every fixed_code passes its test.

These tests run Python bugs in-process and skip JS/Go (those need
node/go installed, verified in the executor tests instead).
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.corpus import (
    ALL_BUGS, PYTHON_BUGS, JAVASCRIPT_BUGS, GO_BUGS,
    get_bug, get_bugs_by_language, get_bugs_by_type,
)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_total_bug_count():
    assert len(ALL_BUGS) == 15  # 5 per language


def test_python_bug_count():
    assert len(PYTHON_BUGS) == 5


def test_javascript_bug_count():
    assert len(JAVASCRIPT_BUGS) == 5


def test_go_bug_count():
    assert len(GO_BUGS) == 5


def test_all_bugs_have_required_fields():
    for bug in ALL_BUGS:
        assert bug.id, f"{bug} missing id"
        assert bug.language in ("python", "javascript", "go"), f"{bug} bad language"
        assert bug.description, f"{bug} missing description"
        assert bug.buggy_code, f"{bug} missing buggy_code"
        assert bug.test_code, f"{bug} missing test_code"
        assert bug.fixed_code, f"{bug} missing fixed_code"
        assert bug.bug_type, f"{bug} missing bug_type"


def test_bug_ids_are_unique():
    ids = [b.id for b in ALL_BUGS]
    assert len(ids) == len(set(ids)), "Duplicate bug IDs found"


def test_get_bug_by_id():
    bug = get_bug("py-001")
    assert bug is not None
    assert bug.language == "python"


def test_get_bug_returns_none_for_unknown_id():
    assert get_bug("xx-999") is None


def test_get_bugs_by_language():
    py = get_bugs_by_language("python")
    js = get_bugs_by_language("javascript")
    go = get_bugs_by_language("go")
    assert len(py) == 5
    assert len(js) == 5
    assert len(go) == 5


def test_get_bugs_by_type_covers_all_types():
    types = set(b.bug_type for b in ALL_BUGS)
    expected = {"off-by-one", "wrong-operator", "missing-case",
                "logic-error", "boundary", "wrong-return"}
    assert types == expected


# ---------------------------------------------------------------------------
# Python correctness tests: buggy code fails, fixed code passes
# ---------------------------------------------------------------------------

def _run_python_tests(code: str, test_code: str) -> tuple:
    """Execute code + test_code in a fresh module. Returns (passed, error_msg)."""
    mod = types.ModuleType("solution")
    try:
        exec(compile(code, "solution.py", "exec"), mod.__dict__)
    except Exception as e:
        return False, f"Code failed to compile/run: {e}"

    sys.modules["solution"] = mod
    try:
        test_mod = types.ModuleType("test_solution")
        exec(compile(test_code, "test.py", "exec"), test_mod.__dict__)
        # Run all test functions
        for name, fn in vars(test_mod).items():
            if name.startswith("test_") and callable(fn):
                fn()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        sys.modules.pop("solution", None)


@pytest.mark.parametrize("bug", PYTHON_BUGS, ids=[b.id for b in PYTHON_BUGS])
def test_python_buggy_code_fails(bug):
    passed, error = _run_python_tests(bug.buggy_code, bug.test_code)
    assert not passed, (
        f"{bug.id}: buggy code unexpectedly passed all tests. "
        f"This 'bug' doesn't actually cause test failures."
    )


@pytest.mark.parametrize("bug", PYTHON_BUGS, ids=[b.id for b in PYTHON_BUGS])
def test_python_fixed_code_passes(bug):
    passed, error = _run_python_tests(bug.fixed_code, bug.test_code)
    assert passed, (
        f"{bug.id}: fixed code failed tests.\nError: {error}"
    )
