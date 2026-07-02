"""ReAct (Reason + Act) agent loop for autonomous debugging.

The agent follows a strict Reason -> Act -> Observe loop:

  1. REASON: given the buggy code + test output, think about what's wrong
  2. ACT: produce a fixed version of the code
  3. OBSERVE: run the tests against the fix, observe pass/fail + output
  4. REPEAT until tests pass or max_iterations reached

Each iteration the agent sees:
  - The original buggy code
  - The current (possibly partially fixed) code
  - All previous attempts and their test outputs
  - The full test output from the latest attempt

The LLM backend is configurable: by default uses the llama-inference
HTTP server (from github.com/sujanuj/llama-inference), but any callable
(prompt: str) -> str works, including a mock for testing.

Why ReAct rather than a single-shot fix?
  Single-shot: send the bug once, get a fix, done.
  ReAct: send the bug, get a fix attempt, run it, if it fails send the
  error back and try again. The iterative loop is what makes this an
  *agent* rather than a smart autocomplete call -- it can recover from
  wrong first guesses by observing real test output.

  For simple bugs (off-by-one, wrong operator) the agent usually fixes
  it in 1-2 iterations. For harder bugs (logic errors, missing cases)
  it may need 3-4 iterations of seeing different failure modes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from executor.executor import Executor, ExecutionResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    """One iteration of the ReAct loop."""
    iteration: int
    reasoning: str          # LLM's explanation of what it thinks is wrong
    proposed_code: str      # LLM's proposed fix
    result: ExecutionResult # test run result for this attempt


@dataclass
class DebugSession:
    """Full record of a debugging session on one bug."""
    bug_id: str
    language: str
    original_code: str
    test_code: str
    attempts: List[Attempt] = field(default_factory=list)
    final_code: Optional[str] = None
    solved: bool = False
    total_time_s: float = 0.0

    @property
    def num_iterations(self) -> int:
        return len(self.attempts)

    @property
    def first_passing_iteration(self) -> Optional[int]:
        for attempt in self.attempts:
            if attempt.result.passed:
                return attempt.iteration
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    language: str,
    original_code: str,
    test_code: str,
    current_code: str,
    attempts: List[Attempt],
) -> str:
    """Build the ReAct prompt for the current iteration.

    The prompt is structured so the LLM must respond with:
      REASONING: <explanation of what's wrong>
      FIXED_CODE:
      ```
      <corrected code>
      ```

    This structured format makes it easy to parse the response without
    relying on the LLM to follow a JSON schema.
    """
    lines = [
        f"You are an expert {language} debugger.",
        f"Your task: fix the bug in the following {language} code so all tests pass.",
        "",
        "## Original buggy code",
        f"```{language}",
        original_code.strip(),
        "```",
        "",
        "## Test suite (must pass after your fix)",
        f"```{language}",
        test_code.strip(),
        "```",
    ]

    if attempts:
        lines += ["", "## Previous fix attempts (all failed)"]
        for attempt in attempts:
            lines += [
                f"### Attempt {attempt.iteration}",
                f"Reasoning: {attempt.reasoning}",
                f"```{language}",
                attempt.proposed_code.strip(),
                "```",
                f"Test output:",
                "```",
                attempt.result.output[:500] if attempt.result.output else "(no output)",
                "```",
                f"Exit code: {attempt.result.exit_code}",
                "",
            ]

    if attempts:
        lines += [
            "## Current code (your last attempt)",
            f"```{language}",
            current_code.strip(),
            "```",
            "",
            "The tests still fail. Study the error output carefully and try a different fix.",
        ]

    lines += [
        "",
        "## Your response format (follow exactly)",
        "REASONING: <one paragraph explaining what the bug is and how you will fix it>",
        "FIXED_CODE:",
        f"```{language}",
        "<your corrected code here>",
        "```",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response: str, language: str) -> tuple[str, str]:
    """Extract (reasoning, fixed_code) from the LLM's response.

    The response is expected to follow:
      REASONING: <text>
      FIXED_CODE:
      ```<lang>
      <code>
      ```

    Falls back gracefully if the format isn't followed exactly:
    - If no REASONING found, uses the full response as reasoning.
    - If no code block found, uses everything after FIXED_CODE: as code.
    """
    # Extract reasoning
    reasoning = ""
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?=FIXED_CODE:|```|$)", response, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = response[:200].strip()

    # Extract code block -- try fenced first, then plain
    code = ""
    # Fenced: ```language or ``` followed by code
    fenced = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if fenced:
        code = fenced.group(1).strip()
    else:
        # Plain: everything after FIXED_CODE:
        fixed_match = re.search(r"FIXED_CODE:\s*(.+)$", response, re.DOTALL)
        if fixed_match:
            code = fixed_match.group(1).strip()
        else:
            # Last resort: use the whole response as code
            code = response.strip()

    return reasoning, code


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def make_llama_backend(
    host: str = "127.0.0.1",
    port: int = 8080,
    max_new_tokens: int = 512,
) -> Callable[[str], str]:
    """Call the llama-inference HTTP server (github.com/sujanuj/llama-inference).

    The server accepts POST /generate with {"token_ids": [...], "max_new_tokens": N}
    and returns {"token_ids": [...]}.

    Since the server works with token IDs (not text), this backend uses a
    simple character-level tokenization for the prompt and decodes the
    output token IDs back to text. For production use, wire in a real
    BPE tokenizer; for this portfolio project, the architecture is what
    matters -- swapping the tokenizer is a one-line change.
    """
    import urllib.request
    import json as _json

    def generate(prompt: str) -> str:
        # Simple ASCII tokenization: each character becomes its ord value.
        # This works for the structured prompts this agent produces.
        token_ids = [ord(c) for c in prompt if ord(c) < 128][:2048]

        payload = _json.dumps({
            "token_ids": token_ids,
            "max_new_tokens": max_new_tokens,
        }).encode()

        req = urllib.request.Request(
            f"http://{host}:{port}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = _json.loads(resp.read())

        # Decode only the newly generated tokens (after the prompt)
        generated_ids = result["token_ids"][len(token_ids):]
        return "".join(chr(t) for t in generated_ids if 32 <= t < 128)

    return generate


def make_hf_backend(
    model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    api_key: str = "",
    max_new_tokens: int = 512,
) -> Callable[[str], str]:
    """Call the HuggingFace Inference API as an alternative backend."""
    import urllib.request
    import json as _json

    def generate(prompt: str) -> str:
        payload = _json.dumps({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            }
        }).encode()

        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{model}",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = _json.loads(resp.read())

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        return str(result)

    return generate


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class DebugAgent:
    """Autonomous debugging agent using the ReAct loop.

    Args:
        llm: callable (prompt: str) -> str. The language model backend.
             Can be make_llama_backend(), make_hf_backend(), or any mock.
        executor: Executor instance for running tests.
        max_iterations: maximum fix attempts before giving up.
        verbose: if True, print each iteration's reasoning and test output.
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        executor: Optional[Executor] = None,
        max_iterations: int = 5,
        verbose: bool = False,
    ):
        self.llm = llm
        self.executor = executor or Executor(timeout=10)
        self.max_iterations = max_iterations
        self.verbose = verbose

    def debug(
        self,
        bug_id: str,
        language: str,
        buggy_code: str,
        test_code: str,
    ) -> DebugSession:
        """Run the ReAct loop on one bug. Returns a DebugSession."""
        session = DebugSession(
            bug_id=bug_id,
            language=language,
            original_code=buggy_code,
            test_code=test_code,
        )
        current_code = buggy_code
        t0 = time.time()

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n[{bug_id}] Iteration {iteration}/{self.max_iterations}")

            # REASON + ACT: ask the LLM for a fix
            prompt = build_prompt(
                language=language,
                original_code=buggy_code,
                test_code=test_code,
                current_code=current_code,
                attempts=session.attempts,
            )
            response = self.llm(prompt)
            reasoning, proposed_code = parse_response(response, language)

            if not proposed_code.strip():
                proposed_code = current_code  # fallback: keep current

            if self.verbose:
                print(f"  Reasoning: {reasoning[:100]}...")

            # OBSERVE: run the tests
            result = self.executor.run(language, proposed_code, test_code)

            attempt = Attempt(
                iteration=iteration,
                reasoning=reasoning,
                proposed_code=proposed_code,
                result=result,
            )
            session.attempts.append(attempt)
            current_code = proposed_code

            if self.verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"  Tests: {status} (exit={result.exit_code})")
                if not result.passed and result.output:
                    print(f"  Output: {result.output[:200]}")

            if result.passed:
                session.solved = True
                session.final_code = proposed_code
                break

        if not session.solved:
            session.final_code = current_code

        session.total_time_s = time.time() - t0
        return session
