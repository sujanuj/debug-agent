"""Benchmark harness: runs the debug agent on the full bug corpus and
reports fix rate, average iterations, and failure analysis.

This is the "does it actually work?" measurement for the portfolio.
Unlike the unit tests (which use mock LLMs), the benchmark runs against
a real LLM backend and reports honest numbers:

  - Fix rate by language (Python/JS/Go)
  - Fix rate by bug type (off-by-one, wrong-operator, etc.)
  - Average iterations to fix across solved bugs
  - Which bugs the agent couldn't fix and why

The benchmark uses the same DebugAgent and corpus as the rest of the
project, so the numbers directly reflect the full stack.

Usage (with HuggingFace Inference API):
  export HF_API_KEY=your_key
  python benchmark/run_benchmark.py

Usage (with mock LLM, for CI/testing):
  python benchmark/run_benchmark.py --mock

Usage (single bug, for debugging the agent):
  python benchmark/run_benchmark.py --bug py-001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import DebugAgent, DebugSession, make_hf_backend
from corpus.corpus import ALL_BUGS, BugCase, get_bug, get_bugs_by_language
from executor.executor import Executor


# ---------------------------------------------------------------------------
# Mock LLM for CI runs (no API key needed)
# ---------------------------------------------------------------------------

def make_mock_llm(corpus: List[BugCase]):
    """A mock LLM that returns the known fixed code for each bug.

    Used in CI and testing to verify the benchmark harness works
    end-to-end without a real LLM. Simulates a perfect agent -- fix
    rate should be 100% on all Python bugs (JS/Go depend on runtimes).

    The mock finds the fixed code by matching the bug ID embedded in
    the prompt (each prompt includes the bug's original code, which we
    can match against the corpus).
    """
    code_to_fix = {b.buggy_code.strip(): b for b in corpus}

    def generate(prompt: str) -> str:
        # Find which bug this prompt is for by matching buggy code
        for buggy_code, bug in code_to_fix.items():
            if buggy_code[:50] in prompt:
                return f"""\
REASONING: Found the bug in {bug.id}. Applying the known fix.
FIXED_CODE:
```{bug.language}
{bug.fixed_code}
```"""
        # Fallback: return the prompt's current code unchanged
        return "REASONING: Could not identify bug.\nFIXED_CODE:\n```\npass\n```"

    return generate


# ---------------------------------------------------------------------------
# Benchmark result structures
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class BugResult:
    bug_id: str
    language: str
    bug_type: str
    solved: bool
    iterations: int
    time_s: float
    failure_reason: Optional[str] = None


@dataclass
class BenchmarkReport:
    results: List[BugResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.results if r.solved)

    @property
    def fix_rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    def by_language(self) -> Dict[str, dict]:
        langs = {}
        for r in self.results:
            if r.language not in langs:
                langs[r.language] = {"total": 0, "solved": 0}
            langs[r.language]["total"] += 1
            if r.solved:
                langs[r.language]["solved"] += 1
        for lang in langs:
            t = langs[lang]["total"]
            s = langs[lang]["solved"]
            langs[lang]["rate"] = s / t if t else 0.0
        return langs

    def by_bug_type(self) -> Dict[str, dict]:
        types = {}
        for r in self.results:
            if r.bug_type not in types:
                types[r.bug_type] = {"total": 0, "solved": 0}
            types[r.bug_type]["total"] += 1
            if r.solved:
                types[r.bug_type]["solved"] += 1
        for t in types:
            total = types[t]["total"]
            solved = types[t]["solved"]
            types[t]["rate"] = solved / total if total else 0.0
        return types

    def avg_iterations(self) -> float:
        solved = [r for r in self.results if r.solved]
        if not solved:
            return 0.0
        return sum(r.iterations for r in solved) / len(solved)

    def print_report(self):
        print("\n" + "=" * 70)
        print("DEBUG AGENT BENCHMARK RESULTS")
        print("=" * 70)
        print(f"\nOverall: {self.solved}/{self.total} fixed "
              f"({self.fix_rate*100:.0f}%) in {self.total_time_s:.1f}s")
        print(f"Avg iterations to fix: {self.avg_iterations():.1f}")

        print("\nBy language:")
        for lang, stats in sorted(self.by_language().items()):
            print(f"  {lang:<12} {stats['solved']}/{stats['total']} "
                  f"({stats['rate']*100:.0f}%)")

        print("\nBy bug type:")
        for btype, stats in sorted(self.by_bug_type().items()):
            print(f"  {btype:<20} {stats['solved']}/{stats['total']} "
                  f"({stats['rate']*100:.0f}%)")

        failures = [r for r in self.results if not r.solved]
        if failures:
            print("\nUnsolved bugs:")
            for r in failures:
                reason = r.failure_reason or "max iterations reached"
                print(f"  {r.bug_id} ({r.language}, {r.bug_type}): {reason}")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    bugs: List[BugCase],
    llm,
    max_iterations: int = 5,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run the debug agent on each bug and collect results."""
    executor = Executor(timeout=10)
    agent = DebugAgent(
        llm=llm,
        executor=executor,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    report = BenchmarkReport()
    t0 = time.time()

    for bug in bugs:
        print(f"  [{bug.id}] {bug.description[:50]}...", end=" ", flush=True)
        session = agent.debug(
            bug_id=bug.id,
            language=bug.language,
            buggy_code=bug.buggy_code,
            test_code=bug.test_code,
        )
        status = "FIXED" if session.solved else f"FAIL({session.num_iterations} iters)"
        print(status)

        failure_reason = None
        if not session.solved and session.attempts:
            last = session.attempts[-1]
            if last.result.timed_out:
                failure_reason = "timed out"
            elif last.result.error:
                failure_reason = last.result.error[:80]
            else:
                output = last.result.output
                failure_reason = output[:80] if output else "unknown"

        report.results.append(BugResult(
            bug_id=bug.id,
            language=bug.language,
            bug_type=bug.bug_type,
            solved=session.solved,
            iterations=session.num_iterations,
            time_s=session.total_time_s,
            failure_reason=failure_reason,
        ))

    report.total_time_s = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug agent benchmark")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no API key needed, for CI)")
    parser.add_argument("--bug", type=str, default=None,
                        help="Run on a single bug ID (e.g. py-001)")
    parser.add_argument("--language", type=str, default=None,
                        help="Run only bugs for this language (python/javascript/go)")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    # Select bugs to run
    if args.bug:
        bug = get_bug(args.bug)
        if not bug:
            print(f"Bug {args.bug!r} not found in corpus.")
            sys.exit(1)
        bugs = [bug]
    elif args.language:
        bugs = get_bugs_by_language(args.language)
        if not bugs:
            print(f"No bugs found for language {args.language!r}")
            sys.exit(1)
    else:
        bugs = ALL_BUGS

    # Select LLM backend
    if args.mock:
        print("Using mock LLM (perfect oracle -- tests the harness, not the agent)")
        llm = make_mock_llm(bugs)
    else:
        api_key = os.environ.get("HF_API_KEY", "")
        if not api_key:
            print("Error: set HF_API_KEY or use --mock")
            sys.exit(1)
        llm = make_hf_backend(api_key=api_key)

    print(f"\nRunning benchmark on {len(bugs)} bugs "
          f"(max_iterations={args.max_iterations}):\n")
    report = run_benchmark(bugs, llm, args.max_iterations, args.verbose)
    report.print_report()

    if args.output:
        data = {
            "total": report.total,
            "solved": report.solved,
            "fix_rate": report.fix_rate,
            "avg_iterations": report.avg_iterations(),
            "total_time_s": report.total_time_s,
            "results": [
                {
                    "bug_id": r.bug_id,
                    "language": r.language,
                    "bug_type": r.bug_type,
                    "solved": r.solved,
                    "iterations": r.iterations,
                    "time_s": r.time_s,
                    "failure_reason": r.failure_reason,
                }
                for r in report.results
            ],
        }
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
