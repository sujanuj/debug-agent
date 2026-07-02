"""Sandboxed code executor for Python, JavaScript, and Go.

Writes code to a temp directory, runs the test file in a subprocess,
and captures stdout/stderr/exit code. Each execution gets its own
isolated temp directory that is cleaned up afterward.

Why subprocess rather than exec()?
  - exec() shares the same process -- a buggy solution that calls
    sys.exit() or corrupts global state would kill the test runner.
  - subprocess gives a clean process boundary: the solution runs in
    isolation, we get its exit code and output, and crashes don't
    affect the agent loop.
  - Real CI systems (GitHub Actions, pytest, etc.) do the same thing.

Timeout: every execution has a hard timeout (default 10s) so an
infinite loop in a buggy solution doesn't hang the agent forever.

Language support:
  - Python: writes solution.py + test.py, runs `python test.py`
  - JavaScript: writes solution.js + test.js, runs `node test.js`
  - Go: writes solution.go + solution_test.go in a module, runs `go test`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of running a test suite against a solution.

    passed: True if the test process exited with code 0.
    stdout: combined stdout from the test run.
    stderr: combined stderr from the test run.
    exit_code: raw process exit code.
    timed_out: True if the process was killed for exceeding the timeout.
    error: any exception raised while trying to run (e.g. language not installed).
    """
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def output(self) -> str:
        """Combined stdout + stderr for display in the agent loop."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(self.stderr.strip())
        return "\n".join(parts)

    def __repr__(self):
        status = "PASS" if self.passed else ("TIMEOUT" if self.timed_out else "FAIL")
        return f"ExecutionResult({status}, exit={self.exit_code})"


class Executor:
    """Runs solution code + test code in a sandboxed subprocess.

    Args:
        timeout: seconds before killing the subprocess. Default 10s.
            Long enough for real test suites, short enough to catch
            infinite loops quickly.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run(
        self,
        language: str,
        solution_code: str,
        test_code: str,
    ) -> ExecutionResult:
        """Run test_code against solution_code for the given language.

        Args:
            language: "python" | "javascript" | "go"
            solution_code: the implementation to test.
            test_code: the test suite to run against it.

        Returns:
            ExecutionResult with pass/fail, stdout, stderr, exit_code.
        """
        runners = {
            "python": self._run_python,
            "javascript": self._run_javascript,
            "go": self._run_go,
        }
        if language not in runners:
            return ExecutionResult(
                passed=False, stdout="", stderr="",
                exit_code=-1,
                error=f"Unsupported language: {language!r}. "
                      f"Supported: {list(runners)}"
            )
        return runners[language](solution_code, test_code)

    def _run_python(self, solution_code: str, test_code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            sol_path = Path(tmpdir) / "solution.py"
            test_path = Path(tmpdir) / "test_solution.py"
            sol_path.write_text(solution_code)
            test_path.write_text(test_code)
            return self._subprocess(
                ["python3", str(test_path)],
                cwd=tmpdir,
            )

    def _run_javascript(self, solution_code: str, test_code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            sol_path = Path(tmpdir) / "solution.js"
            test_path = Path(tmpdir) / "test_solution.js"
            sol_path.write_text(solution_code)
            test_path.write_text(test_code)
            return self._subprocess(
                ["node", str(test_path)],
                cwd=tmpdir,
            )

    def _run_go(self, solution_code: str, test_code: str) -> ExecutionResult:
        """Go requires a module with a package name. We use 'solution'
        throughout and create a minimal go.mod so the test runner can
        find the package without network access.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # go.mod -- minimal module definition, no external deps
            go_mod = textwrap.dedent("""\
                module debugagent/solution

                go 1.21
            """)
            (Path(tmpdir) / "go.mod").write_text(go_mod)
            (Path(tmpdir) / "solution.go").write_text(solution_code)
            (Path(tmpdir) / "solution_test.go").write_text(test_code)
            return self._subprocess(
                ["go", "test", "-v", "./..."],
                cwd=tmpdir,
            )

    def _subprocess(self, cmd: list, cwd: str) -> ExecutionResult:
        """Run cmd in cwd with timeout. Captures stdout+stderr separately."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ExecutionResult(
                passed=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False,
                stdout="",
                stderr=f"Process timed out after {self.timeout}s",
                exit_code=-1,
                timed_out=True,
            )
        except FileNotFoundError as e:
            # Language runtime not installed
            cmd_name = cmd[0]
            return ExecutionResult(
                passed=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                error=f"Runtime not found: {cmd_name!r}. "
                      f"Is {cmd_name} installed and on PATH?",
            )
        except Exception as e:
            return ExecutionResult(
                passed=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                error=str(e),
            )
