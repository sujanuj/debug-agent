"""Test writer: generates a test suite for a function when none is provided.

The debug agent's primary mode is: given buggy code + failing tests, fix
the bug. But in practice, many real bugs arrive without any tests at all
-- a function that "doesn't work" with no reproduction case. This module
handles that case by generating tests first, then handing them to the
main agent loop.

Two-step process:
  1. TEST GENERATION: given a function's code and description, ask the
     LLM to write a test suite that covers normal cases, edge cases, and
     boundary conditions.
  2. TEST VALIDATION: run the generated tests against the original code.
     If they ALL pass, the tests are useless (they don't catch the bug).
     If at least one fails, the tests are useful and we proceed to the
     ReAct fix loop.

Why validate the tests?
  LLMs sometimes write tests that match the buggy behavior rather than
  the correct behavior. For example, given a function that returns -1
  instead of None for missing elements, the LLM might write:
    assert find(lst, x) == -1  # matches the bug, not the spec
  Running the tests against the buggy code and checking that at least
  one fails is a cheap sanity check that the tests are actually testing
  something.

This module is intentionally separate from agent.py so it can be used
standalone (generate tests for any function, not just for debugging).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from executor.executor import Executor, ExecutionResult


@dataclass
class GeneratedTests:
    """Result of the test generation step.

    test_code: the generated test suite as a string.
    validation_result: result of running the tests against the original
        (presumably buggy) code. If passed=True, the tests didn't catch
        the bug -- they may be too permissive.
    useful: True if at least one test failed against the original code.
        Only useful tests are worth passing to the fix loop.
    generation_prompt: the prompt sent to the LLM (for debugging).
    raw_response: the LLM's raw response (for debugging).
    """
    test_code: str
    validation_result: Optional[ExecutionResult]
    useful: bool
    generation_prompt: str
    raw_response: str


def build_test_generation_prompt(
    language: str,
    code: str,
    description: str,
) -> str:
    """Build a prompt asking the LLM to write tests for a function.

    The prompt asks for:
    - At least 4 test cases
    - Normal cases, edge cases, and boundary conditions
    - Tests that would catch common bugs (off-by-one, wrong operator, etc.)
    - A specific format so the response is easy to parse
    """
    return f"""\
You are an expert {language} test writer.

Write a comprehensive test suite for the following {language} function.
The function description says what it SHOULD do (treat this as the spec).

## Function description
{description}

## Function code
```{language}
{code.strip()}
```

## Requirements for your tests
- Write at least 4 test cases
- Cover: normal inputs, edge cases (empty, single element, negatives), boundary conditions
- Tests should catch common bugs: off-by-one errors, wrong operators, missing cases
- Do NOT assume the current implementation is correct -- test the SPEC, not the code

## Response format (follow exactly)
TEST_CODE:
```{language}
<your test code here>
```

The test code must:
- For Python: import from 'solution' module, use assert statements, print "ALL TESTS PASSED" at end
- For JavaScript: require('./solution'), use throw new Error() on failure, console.log("ALL TESTS PASSED")
- For Go: use package solution, standard testing.T
"""


def parse_test_response(response: str, language: str) -> str:
    """Extract test code from the LLM's response.

    Looks for a fenced code block after TEST_CODE:, falls back to
    any fenced code block, then falls back to everything after TEST_CODE:.
    """
    # Fenced block after TEST_CODE:
    after_marker = re.search(r"TEST_CODE:\s*```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if after_marker:
        return after_marker.group(1).strip()

    # Any fenced block
    fenced = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Everything after TEST_CODE:
    after = re.search(r"TEST_CODE:\s*(.+)$", response, re.DOTALL)
    if after:
        return after.group(1).strip()

    return response.strip()


class TestWriter:
    """Generates test suites for functions using an LLM.

    Args:
        llm: callable (prompt: str) -> str.
        executor: Executor for validating the generated tests.
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        executor: Optional[Executor] = None,
    ):
        self.llm = llm
        self.executor = executor or Executor(timeout=10)

    def generate(
        self,
        language: str,
        code: str,
        description: str,
        validate: bool = True,
    ) -> GeneratedTests:
        """Generate tests for a function.

        Args:
            language: "python" | "javascript" | "go"
            code: the function implementation to write tests for.
            description: what the function is supposed to do (the spec).
            validate: if True, run the tests against code and check that
                at least one fails (confirming the tests catch bugs).

        Returns:
            GeneratedTests with test_code, validation_result, and useful flag.
        """
        prompt = build_test_generation_prompt(language, code, description)
        response = self.llm(prompt)
        test_code = parse_test_response(response, language)

        validation_result = None
        useful = True  # assume useful if not validating

        if validate and test_code:
            validation_result = self.executor.run(language, code, test_code)
            # Tests are useful if they catch at least one failure
            # (i.e., the buggy code fails the tests)
            useful = not validation_result.passed

        return GeneratedTests(
            test_code=test_code,
            validation_result=validation_result,
            useful=useful,
            generation_prompt=prompt,
            raw_response=response,
        )
