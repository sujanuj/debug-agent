"""Command-line interface for the autonomous debug agent.

Usage:
  # Fix a bug given buggy code and tests
  python -m debug_agent fix buggy.py --tests test.py --language python

  # Fix a bug, generate tests automatically if none provided
  python -m debug_agent fix buggy.py --language python --description "Return sum of a and b"

  # Run the benchmark on the full corpus
  python -m debug_agent benchmark --mock
  python -m debug_agent benchmark --language python

  # Run on a single bug from the corpus
  python -m debug_agent benchmark --bug py-001 --verbose

  # List all bugs in the corpus
  python -m debug_agent list
  python -m debug_agent list --language python
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import DebugAgent, make_hf_backend
from benchmark.run_benchmark import make_mock_llm, run_benchmark
from corpus.corpus import ALL_BUGS, get_bug, get_bugs_by_language
from executor.executor import Executor
from test_writer.test_writer import TestWriter


# ---------------------------------------------------------------------------
# LLM selection
# ---------------------------------------------------------------------------

def get_llm(mock: bool = False, bugs=None):
    if mock:
        print("[mock] Using oracle LLM (knows fixed code from corpus)")
        return make_mock_llm(bugs or ALL_BUGS)

    api_key = os.environ.get("HF_API_KEY", "")
    if not api_key:
        print("Error: set HF_API_KEY environment variable or use --mock")
        sys.exit(1)

    print(f"Using HuggingFace Inference API")
    return make_hf_backend(api_key=api_key)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fix(args):
    """Fix a bug in a file."""
    buggy_path = Path(args.file)
    if not buggy_path.exists():
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    buggy_code = buggy_path.read_text()
    language = args.language or _detect_language(buggy_path)
    if not language:
        print("Error: could not detect language. Use --language python|javascript|go")
        sys.exit(1)

    llm = get_llm(mock=args.mock)
    executor = Executor(timeout=args.timeout)

    # Get or generate tests
    if args.tests:
        test_path = Path(args.tests)
        if not test_path.exists():
            print(f"Error: test file not found: {args.tests}")
            sys.exit(1)
        test_code = test_path.read_text()
        print(f"Using tests from {args.tests}")
    elif args.description:
        print(f"No tests provided. Generating tests for: {args.description}")
        writer = TestWriter(llm=llm, executor=executor)
        generated = writer.generate(language, buggy_code, args.description, validate=True)
        if not generated.test_code:
            print("Error: failed to generate tests")
            sys.exit(1)
        if not generated.useful:
            print("Warning: generated tests pass on buggy code -- they may not catch the bug")
        test_code = generated.test_code
        print(f"Generated {test_code.count('assert')} assertions")
    else:
        print("Error: provide --tests <file> or --description '<what the function should do>'")
        sys.exit(1)

    # Run the agent
    print(f"\nRunning debug agent on {args.file} ({language})")
    print(f"Max iterations: {args.max_iterations}\n")

    agent = DebugAgent(
        llm=llm,
        executor=executor,
        max_iterations=args.max_iterations,
        verbose=True,
    )
    session = agent.debug(
        bug_id=buggy_path.stem,
        language=language,
        buggy_code=buggy_code,
        test_code=test_code,
    )

    print(f"\n{'='*60}")
    if session.solved:
        print(f"FIXED in {session.num_iterations} iteration(s) "
              f"({session.total_time_s:.1f}s)")
        if args.output:
            Path(args.output).write_text(session.final_code)
            print(f"Fixed code written to {args.output}")
        else:
            print(f"\nFixed code:\n{'-'*40}")
            print(session.final_code)
    else:
        print(f"COULD NOT FIX after {session.num_iterations} iteration(s) "
              f"({session.total_time_s:.1f}s)")
        print("The agent's best attempt:")
        print(session.final_code)
        sys.exit(1)


def cmd_benchmark(args):
    """Run the benchmark on the corpus."""
    if args.bug:
        bug = get_bug(args.bug)
        if not bug:
            print(f"Bug {args.bug!r} not in corpus. Use 'list' to see available bugs.")
            sys.exit(1)
        bugs = [bug]
    elif args.language:
        bugs = get_bugs_by_language(args.language)
        if not bugs:
            print(f"No bugs for language {args.language!r}")
            sys.exit(1)
    else:
        bugs = ALL_BUGS

    llm = get_llm(mock=args.mock, bugs=bugs)
    print(f"\nBenchmark: {len(bugs)} bugs, max_iterations={args.max_iterations}\n")
    report = run_benchmark(bugs, llm, args.max_iterations, args.verbose)
    report.print_report()

    if args.output:
        import json
        data = {
            "total": report.total,
            "solved": report.solved,
            "fix_rate": report.fix_rate,
            "avg_iterations": report.avg_iterations(),
            "results": [
                {
                    "bug_id": r.bug_id, "language": r.language,
                    "bug_type": r.bug_type, "solved": r.solved,
                    "iterations": r.iterations, "time_s": r.time_s,
                }
                for r in report.results
            ],
        }
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.output}")


def cmd_list(args):
    """List bugs in the corpus."""
    bugs = get_bugs_by_language(args.language) if args.language else ALL_BUGS
    print(f"\n{'ID':<12} {'Language':<14} {'Type':<22} {'Description'}")
    print("-" * 80)
    for bug in bugs:
        print(f"{bug.id:<12} {bug.language:<14} {bug.bug_type:<22} "
              f"{bug.description[:35]}")
    print(f"\nTotal: {len(bugs)} bugs")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {".py": "python", ".js": "javascript", ".go": "go"}.get(ext, "")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="debug_agent",
        description="Autonomous debugging agent for Python, JavaScript, and Go",
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no API key needed)")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=10,
                        help="Test execution timeout in seconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Write output to file")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # fix
    fix_p = sub.add_parser("fix", help="Fix a bug in a file")
    fix_p.add_argument("file", help="Path to the buggy source file")
    fix_p.add_argument("--tests", type=str, default=None,
                       help="Path to test file")
    fix_p.add_argument("--language", type=str, default=None,
                       help="Language (auto-detected from extension if omitted)")
    fix_p.add_argument("--description", type=str, default=None,
                       help="What the function should do (used to generate tests)")

    # benchmark
    bench_p = sub.add_parser("benchmark", help="Run on the bug corpus")
    bench_p.add_argument("--mock", action="store_true")
    bench_p.add_argument("--bug", type=str, default=None)
    bench_p.add_argument("--language", type=str, default=None)

    # list
    list_p = sub.add_parser("list", help="List bugs in the corpus")
    list_p.add_argument("--language", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fix":
        cmd_fix(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
