"""Tests for the ReAct debug agent.

Uses a mock LLM that returns pre-written responses so tests are
deterministic and fast -- no real LLM needed. Tests verify:

  1. Prompt construction includes the right content.
  2. Response parsing handles well-formed and malformed responses.
  3. Agent loop: solves a bug in 1 iteration when LLM gives the right fix.
  4. Agent loop: retries when the first fix is wrong.
  5. Agent loop: gives up after max_iterations.
  6. DebugSession records attempts correctly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import (
    DebugAgent, DebugSession, build_prompt, parse_response,
    Attempt,
)
from executor.executor import Executor, ExecutionResult


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------

def test_prompt_contains_buggy_code():
    prompt = build_prompt(
        language="python",
        original_code="def add(a, b): return a - b",
        test_code="assert add(1,2)==3",
        current_code="def add(a, b): return a - b",
        attempts=[],
    )
    assert "def add(a, b): return a - b" in prompt


def test_prompt_contains_test_code():
    prompt = build_prompt(
        language="python",
        original_code="def f(): pass",
        test_code="assert f() == 42",
        current_code="def f(): pass",
        attempts=[],
    )
    assert "assert f() == 42" in prompt


def test_prompt_contains_language():
    prompt = build_prompt(
        language="javascript",
        original_code="function f() {}",
        test_code="assert(f()===1)",
        current_code="function f() {}",
        attempts=[],
    )
    assert "javascript" in prompt.lower()


def test_prompt_includes_previous_attempts():
    prev_result = ExecutionResult(
        passed=False, stdout="", stderr="AssertionError", exit_code=1
    )
    prev_attempt = Attempt(
        iteration=1,
        reasoning="I thought the issue was X",
        proposed_code="def add(a,b): return a*b",
        result=prev_result,
    )
    prompt = build_prompt(
        language="python",
        original_code="def add(a,b): return a-b",
        test_code="assert add(1,2)==3",
        current_code="def add(a,b): return a*b",
        attempts=[prev_attempt],
    )
    assert "Attempt 1" in prompt
    assert "I thought the issue was X" in prompt
    assert "AssertionError" in prompt


def test_prompt_no_previous_attempts_omits_section():
    prompt = build_prompt(
        language="python",
        original_code="def f(): pass",
        test_code="assert f()==1",
        current_code="def f(): pass",
        attempts=[],
    )
    assert "Previous fix attempts" not in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

def test_parse_well_formed_response():
    response = """\
REASONING: The bug is that the operator is wrong.
FIXED_CODE:
```python
def add(a, b):
    return a + b
```"""
    reasoning, code = parse_response(response, "python")
    assert "operator" in reasoning
    assert "return a + b" in code


def test_parse_response_extracts_code_block():
    response = "REASONING: fix it\nFIXED_CODE:\n```\ndef f(): return 1\n```"
    _, code = parse_response(response, "python")
    assert "def f(): return 1" in code


def test_parse_response_handles_missing_reasoning():
    response = "```python\ndef add(a,b): return a+b\n```"
    reasoning, code = parse_response(response, "python")
    assert code  # code extracted
    # reasoning may be empty or the first part of response


def test_parse_response_handles_no_code_block():
    response = "REASONING: fix it\nFIXED_CODE: def add(a,b): return a+b"
    reasoning, code = parse_response(response, "python")
    assert code  # something extracted


# ---------------------------------------------------------------------------
# Agent loop tests with mock LLM
# ---------------------------------------------------------------------------

BUGGY_CODE = """\
def add(a, b):
    return a - b
"""

TEST_CODE = """\
from solution import add
assert add(1, 2) == 3, f"Expected 3, got {add(1,2)}"
assert add(0, 0) == 0
print("ALL TESTS PASSED")
"""

CORRECT_CODE = """\
def add(a, b):
    return a + b
"""


def _mock_llm_correct(prompt: str) -> str:
    """Always returns the correct fix on first try."""
    return f"""\
REASONING: The bug is that - should be +.
FIXED_CODE:
```python
def add(a, b):
    return a + b
```"""


def _mock_llm_wrong_then_correct(call_count=[0]):
    """Returns wrong fix first, correct fix second."""
    def generate(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return """\
REASONING: I think the issue is multiplication.
FIXED_CODE:
```python
def add(a, b):
    return a * b
```"""
        else:
            return """\
REASONING: Actually it should be addition.
FIXED_CODE:
```python
def add(a, b):
    return a + b
```"""
    return generate


def _mock_llm_always_wrong(prompt: str) -> str:
    return """\
REASONING: I think the issue is multiplication.
FIXED_CODE:
```python
def add(a, b):
    return a * b
```"""


def test_agent_solves_in_one_iteration():
    agent = DebugAgent(llm=_mock_llm_correct, max_iterations=5)
    session = agent.debug("test-001", "python", BUGGY_CODE, TEST_CODE)
    assert session.solved
    assert session.num_iterations == 1
    assert session.first_passing_iteration == 1


def test_agent_retries_on_wrong_first_fix():
    agent = DebugAgent(llm=_mock_llm_wrong_then_correct(), max_iterations=5)
    session = agent.debug("test-002", "python", BUGGY_CODE, TEST_CODE)
    assert session.solved
    assert session.num_iterations == 2
    assert session.first_passing_iteration == 2


def test_agent_gives_up_after_max_iterations():
    agent = DebugAgent(llm=_mock_llm_always_wrong, max_iterations=3)
    session = agent.debug("test-003", "python", BUGGY_CODE, TEST_CODE)
    assert not session.solved
    assert session.num_iterations == 3


def test_agent_records_all_attempts():
    agent = DebugAgent(llm=_mock_llm_always_wrong, max_iterations=3)
    session = agent.debug("test-004", "python", BUGGY_CODE, TEST_CODE)
    assert len(session.attempts) == 3
    for i, attempt in enumerate(session.attempts, 1):
        assert attempt.iteration == i
        assert not attempt.result.passed


def test_agent_final_code_is_set_on_success():
    agent = DebugAgent(llm=_mock_llm_correct, max_iterations=5)
    session = agent.debug("test-005", "python", BUGGY_CODE, TEST_CODE)
    assert session.final_code is not None
    assert "return a + b" in session.final_code


def test_agent_final_code_is_set_on_failure():
    agent = DebugAgent(llm=_mock_llm_always_wrong, max_iterations=2)
    session = agent.debug("test-006", "python", BUGGY_CODE, TEST_CODE)
    assert session.final_code is not None


def test_agent_records_timing():
    agent = DebugAgent(llm=_mock_llm_correct, max_iterations=5)
    session = agent.debug("test-007", "python", BUGGY_CODE, TEST_CODE)
    assert session.total_time_s >= 0


def test_debug_session_bug_id_preserved():
    agent = DebugAgent(llm=_mock_llm_correct, max_iterations=5)
    session = agent.debug("my-bug-42", "python", BUGGY_CODE, TEST_CODE)
    assert session.bug_id == "my-bug-42"
