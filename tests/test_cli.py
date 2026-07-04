"""Tests for the CLI.

Tests the argument parser and command dispatch without running real LLM
calls. Uses mock LLM throughout.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from debug_agent.cli import build_parser, cmd_list, _detect_language


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_detect_python():
    assert _detect_language(Path("solution.py")) == "python"


def test_detect_javascript():
    assert _detect_language(Path("solution.js")) == "javascript"


def test_detect_go():
    assert _detect_language(Path("solution.go")) == "go"


def test_detect_unknown():
    assert _detect_language(Path("solution.rb")) == ""


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def test_parser_fix_command():
    parser = build_parser()
    args = parser.parse_args(["fix", "buggy.py", "--language", "python",
                               "--tests", "test.py"])
    assert args.command == "fix"
    assert args.file == "buggy.py"
    assert args.language == "python"
    assert args.tests == "test.py"


def test_parser_fix_with_mock():
    parser = build_parser()
    args = parser.parse_args(["--mock", "fix", "buggy.py", "--language", "python",
                               "--tests", "test.py"])
    assert args.mock is True


def test_parser_benchmark_command():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--mock", "--language", "python"])
    assert args.command == "benchmark"
    assert args.language == "python"


def test_parser_benchmark_single_bug():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--mock", "--bug", "py-001"])
    assert args.bug == "py-001"


def test_parser_list_command():
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"


def test_parser_list_with_language():
    parser = build_parser()
    args = parser.parse_args(["list", "--language", "python"])
    assert args.language == "python"


def test_parser_max_iterations_default():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--mock"])
    assert args.max_iterations == 5


def test_parser_max_iterations_custom():
    parser = build_parser()
    args = parser.parse_args(["--max-iterations", "3", "benchmark", "--mock"])
    assert args.max_iterations == 3


def test_parser_output_flag():
    parser = build_parser()
    args = parser.parse_args(["--output", "results.json", "benchmark", "--mock"])
    assert args.output == "results.json"


# ---------------------------------------------------------------------------
# cmd_list smoke test
# ---------------------------------------------------------------------------

def test_cmd_list_all(capsys):
    parser = build_parser()
    args = parser.parse_args(["list"])
    cmd_list(args)
    captured = capsys.readouterr()
    assert "py-001" in captured.out
    assert "js-001" in captured.out
    assert "go-001" in captured.out
    assert "Total: 15" in captured.out


def test_cmd_list_python_only(capsys):
    parser = build_parser()
    args = parser.parse_args(["list", "--language", "python"])
    cmd_list(args)
    captured = capsys.readouterr()
    assert "py-001" in captured.out
    assert "js-001" not in captured.out
    assert "Total: 5" in captured.out


# ---------------------------------------------------------------------------
# fix command with mock LLM and temp files
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

FIXED_RESPONSE = """\
REASONING: The operator should be + not -.
FIXED_CODE:
```python
def add(a, b):
    return a + b
```"""


def test_fix_command_with_mock_llm(tmp_path, monkeypatch):
    """End-to-end: fix command reads files, runs agent, prints result."""
    buggy = tmp_path / "buggy.py"
    tests = tmp_path / "test_buggy.py"
    buggy.write_text(BUGGY_CODE)
    tests.write_text(TEST_CODE)

    # Patch get_llm to return our mock
    import debug_agent.cli as cli_module
    monkeypatch.setattr(cli_module, "get_llm", lambda **kwargs: (lambda p: FIXED_RESPONSE))

    parser = build_parser()
    args = parser.parse_args([
        "--mock", "fix", str(buggy),
        "--tests", str(tests),
        "--language", "python",
    ])
    # Should not raise
    cli_module.cmd_fix(args)


def test_fix_command_writes_output_file(tmp_path, monkeypatch, capsys):
    buggy = tmp_path / "buggy.py"
    tests = tmp_path / "test_buggy.py"
    output = tmp_path / "fixed.py"
    buggy.write_text(BUGGY_CODE)
    tests.write_text(TEST_CODE)

    import debug_agent.cli as cli_module
    monkeypatch.setattr(cli_module, "get_llm", lambda **kwargs: (lambda p: FIXED_RESPONSE))

    parser = build_parser()
    args = parser.parse_args([
        "--mock", "--output", str(output),
        "fix", str(buggy),
        "--tests", str(tests),
        "--language", "python",
    ])
    cli_module.cmd_fix(args)
    assert output.exists()
    assert "return a + b" in output.read_text()
