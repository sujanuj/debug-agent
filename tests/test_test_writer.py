"""Tests for the test writer module.

Uses mock LLMs so no real model is needed. Verifies:
  1. Prompt construction includes description and code.
  2. Response parsing handles well-formed and malformed responses.
  3. Generated tests are validated against the original code.
  4. Useful flag is set correctly based on validation result.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_writer.test_writer import (
    TestWriter, GeneratedTests,
    build_test_generation_prompt, parse_test_response,
)
from executor.executor import Executor


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_contains_description():
    prompt = build_test_generation_prompt(
        language="python",
        code="def add(a,b): return a-b",
        description="Return the sum of a and b.",
    )
    assert "Return the sum of a and b." in prompt


def test_prompt_contains_code():
    prompt = build_test_generation_prompt(
        language="python",
        code="def add(a,b): return a-b",
        description="Add two numbers.",
    )
    assert "def add(a,b): return a-b" in prompt


def test_prompt_contains_language():
    prompt = build_test_generation_prompt(
        language="javascript",
        code="function f() {}",
        description="Do something.",
    )
    assert "javascript" in prompt.lower()


def test_prompt_mentions_edge_cases():
    prompt = build_test_generation_prompt(
        language="python",
        code="def f(): pass",
        description="A function.",
    )
    assert "edge" in prompt.lower() or "boundary" in prompt.lower()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_parse_well_formed_response():
    response = """\
TEST_CODE:
```python
from solution import add
assert add(1,2) == 3
print("ALL TESTS PASSED")
```"""
    code = parse_test_response(response, "python")
    assert "assert add(1,2) == 3" in code


def test_parse_fenced_block_without_marker():
    response = "Here are tests:\n```python\nassert True\n```"
    code = parse_test_response(response, "python")
    assert "assert True" in code


def test_parse_fallback_to_raw():
    response = "TEST_CODE: assert 1 == 1"
    code = parse_test_response(response, "python")
    assert code  # something returned


# ---------------------------------------------------------------------------
# TestWriter with mock LLM
# ---------------------------------------------------------------------------

BUGGY_ADD = """\
def add(a, b):
    return a - b
"""

GOOD_TESTS = """\
TEST_CODE:
```python
from solution import add
assert add(1, 2) == 3, f"Expected 3, got {add(1,2)}"
assert add(0, 0) == 0
assert add(-1, 1) == 0
print("ALL TESTS PASSED")
```"""

USELESS_TESTS = """\
TEST_CODE:
```python
from solution import add
# These tests match the buggy behavior (a - b)
assert add(3, 1) == 2  # passes on buggy code
assert add(5, 2) == 3  # passes on buggy code
print("ALL TESTS PASSED")
```"""


def _mock_llm_good_tests(prompt: str) -> str:
    return GOOD_TESTS


def _mock_llm_useless_tests(prompt: str) -> str:
    return USELESS_TESTS


def test_test_writer_generates_test_code():
    writer = TestWriter(llm=_mock_llm_good_tests)
    result = writer.generate("python", BUGGY_ADD, "Return sum of a and b.", validate=False)
    assert result.test_code
    assert "assert" in result.test_code


def test_test_writer_useful_when_tests_fail_on_buggy_code():
    writer = TestWriter(llm=_mock_llm_good_tests)
    result = writer.generate("python", BUGGY_ADD, "Return sum of a and b.", validate=True)
    assert result.useful
    assert result.validation_result is not None
    assert not result.validation_result.passed  # buggy code fails good tests


def test_test_writer_not_useful_when_tests_pass_on_buggy_code():
    writer = TestWriter(llm=_mock_llm_useless_tests)
    result = writer.generate("python", BUGGY_ADD, "Return sum of a and b.", validate=True)
    assert not result.useful
    assert result.validation_result is not None
    assert result.validation_result.passed  # useless tests pass on buggy code


def test_test_writer_no_validation():
    writer = TestWriter(llm=_mock_llm_good_tests)
    result = writer.generate("python", BUGGY_ADD, "Return sum of a and b.", validate=False)
    assert result.validation_result is None
    assert result.useful  # assumed useful when not validated


def test_generated_tests_records_prompt_and_response():
    writer = TestWriter(llm=_mock_llm_good_tests)
    result = writer.generate("python", BUGGY_ADD, "Return sum of a and b.", validate=False)
    assert result.generation_prompt
    assert result.raw_response == GOOD_TESTS


def test_test_writer_stores_description_in_prompt():
    captured = []

    def capture_llm(prompt: str) -> str:
        captured.append(prompt)
        return GOOD_TESTS

    writer = TestWriter(llm=capture_llm)
    writer.generate("python", BUGGY_ADD, "Return the sum of two integers.", validate=False)
    assert "Return the sum of two integers." in captured[0]
