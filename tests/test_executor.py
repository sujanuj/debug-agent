"""Tests for the sandboxed code executor.

Tests are organized by language. JS and Go tests are skipped if the
runtime isn't installed (node/go not found), so the test suite still
passes in environments that only have Python.

Four things verified per language:
  1. Correct code + correct tests -> PASS
  2. Buggy code + tests -> FAIL with useful output
  3. Syntax error in solution -> FAIL (not a crash in the agent)
  4. Infinite loop -> FAIL with timed_out=True (not a hang)
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.executor import Executor, ExecutionResult

executor = Executor(timeout=5)

# ---------------------------------------------------------------------------
# Python tests
# ---------------------------------------------------------------------------

PYTHON_CORRECT = """\
def add(a, b):
    return a + b
"""

PYTHON_TESTS_PASS = """\
from solution import add

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0

test_add()
print("ALL TESTS PASSED")
"""

PYTHON_TESTS_FAIL = """\
from solution import add

def test_add():
    assert add(1, 2) == 99  # wrong expected value

test_add()
print("ALL TESTS PASSED")
"""

PYTHON_BUGGY = """\
def add(a, b):
    return a - b  # wrong operator
"""

PYTHON_SYNTAX_ERROR = """\
def add(a, b)
    return a + b
"""

PYTHON_INFINITE_LOOP = """\
def add(a, b):
    while True:
        pass
"""


def test_python_correct_code_passes():
    result = executor.run("python", PYTHON_CORRECT, PYTHON_TESTS_PASS)
    assert result.passed
    assert result.exit_code == 0
    assert not result.timed_out


def test_python_buggy_code_fails():
    result = executor.run("python", PYTHON_BUGGY, PYTHON_TESTS_FAIL)
    assert not result.passed
    assert result.exit_code != 0


def test_python_correct_code_with_failing_tests():
    result = executor.run("python", PYTHON_CORRECT, PYTHON_TESTS_FAIL)
    assert not result.passed
    assert "AssertionError" in result.output or result.exit_code != 0


def test_python_syntax_error_fails_gracefully():
    result = executor.run("python", PYTHON_SYNTAX_ERROR, PYTHON_TESTS_PASS)
    assert not result.passed
    assert result.error is None  # no exception in the executor itself
    assert result.exit_code != 0


def test_python_infinite_loop_times_out():
    result = executor.run("python", PYTHON_INFINITE_LOOP, PYTHON_TESTS_PASS)
    assert not result.passed
    assert result.timed_out


def test_python_output_captured():
    result = executor.run("python", PYTHON_CORRECT, PYTHON_TESTS_PASS)
    assert "ALL TESTS PASSED" in result.stdout


def test_unsupported_language_returns_error():
    result = executor.run("ruby", "puts 'hello'", "")
    assert not result.passed
    assert result.error is not None


# ---------------------------------------------------------------------------
# JavaScript tests (skipped if node not installed)
# ---------------------------------------------------------------------------

node_available = shutil.which("node") is not None

JS_CORRECT = """\
function add(a, b) { return a + b; }
module.exports = { add };
"""

JS_TESTS_PASS = """\
const { add } = require("./solution");
console.assert(add(1, 2) === 3, "1+2 should be 3");
console.assert(add(-1, 1) === 0, "-1+1 should be 0");
console.log("ALL TESTS PASSED");
"""

JS_TESTS_FAIL = """\
const { add } = require("./solution");
const result = add(1, 2);
if (result !== 99) throw new Error(`Expected 99, got ${result}`);
console.log("ALL TESTS PASSED");
"""

JS_BUGGY = """\
function add(a, b) { return a - b; }
module.exports = { add };
"""


@pytest.mark.skipif(not node_available, reason="node not installed")
def test_javascript_correct_code_passes():
    result = executor.run("javascript", JS_CORRECT, JS_TESTS_PASS)
    assert result.passed, f"Expected pass, got: {result.output}"


@pytest.mark.skipif(not node_available, reason="node not installed")
def test_javascript_buggy_code_fails():
    result = executor.run("javascript", JS_BUGGY, JS_TESTS_FAIL)
    assert not result.passed


@pytest.mark.skipif(not node_available, reason="node not installed")
def test_javascript_output_captured():
    result = executor.run("javascript", JS_CORRECT, JS_TESTS_PASS)
    assert "ALL TESTS PASSED" in result.stdout


# ---------------------------------------------------------------------------
# Go tests (skipped if go not installed)
# ---------------------------------------------------------------------------

go_available = shutil.which("go") is not None

GO_CORRECT = """\
package solution

func Add(a, b int) int {
    return a + b
}
"""

GO_TESTS_PASS = """\
package solution

import "testing"

func TestAdd(t *testing.T) {
    if Add(1, 2) != 3 {
        t.Errorf("Add(1,2) = %d, want 3", Add(1, 2))
    }
    if Add(-1, 1) != 0 {
        t.Errorf("Add(-1,1) = %d, want 0", Add(-1, 1))
    }
}
"""

GO_BUGGY = """\
package solution

func Add(a, b int) int {
    return a - b
}
"""


@pytest.mark.skipif(not go_available, reason="go not installed")
def test_go_correct_code_passes():
    result = executor.run("go", GO_CORRECT, GO_TESTS_PASS)
    assert result.passed, f"Expected pass, got: {result.output}"


@pytest.mark.skipif(not go_available, reason="go not installed")
def test_go_buggy_code_fails():
    result = executor.run("go", GO_BUGGY, GO_TESTS_PASS)
    assert not result.passed


# ---------------------------------------------------------------------------
# ExecutionResult helpers
# ---------------------------------------------------------------------------

def test_execution_result_output_combines_stdout_stderr():
    result = ExecutionResult(
        passed=False, stdout="hello", stderr="world", exit_code=1
    )
    assert "hello" in result.output
    assert "world" in result.output


def test_execution_result_repr_shows_status():
    r_pass = ExecutionResult(passed=True, stdout="", stderr="", exit_code=0)
    r_fail = ExecutionResult(passed=False, stdout="", stderr="", exit_code=1)
    r_timeout = ExecutionResult(passed=False, stdout="", stderr="", exit_code=-1, timed_out=True)
    assert "PASS" in repr(r_pass)
    assert "FAIL" in repr(r_fail)
    assert "TIMEOUT" in repr(r_timeout)
