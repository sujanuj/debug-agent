"""Tests for the benchmark harness.

Uses mock LLM so no real API key needed. Verifies:
  1. BenchmarkReport computes fix_rate, by_language, by_bug_type correctly.
  2. run_benchmark runs on a small corpus and returns correct structure.
  3. Mock LLM produces 100% fix rate on Python bugs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.run_benchmark import (
    BenchmarkReport, BugResult, run_benchmark, make_mock_llm,
)
from corpus.corpus import PYTHON_BUGS, ALL_BUGS


# ---------------------------------------------------------------------------
# BenchmarkReport unit tests
# ---------------------------------------------------------------------------

def _make_report(results):
    r = BenchmarkReport()
    r.results = results
    return r


def test_fix_rate_all_solved():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=True, iterations=1, time_s=0.1),
        BugResult("py-002", "python", "logic-error", solved=True, iterations=2, time_s=0.2),
    ])
    assert report.fix_rate == 1.0


def test_fix_rate_none_solved():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=False, iterations=5, time_s=0.5),
    ])
    assert report.fix_rate == 0.0


def test_fix_rate_partial():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=True, iterations=1, time_s=0.1),
        BugResult("py-002", "python", "logic-error", solved=False, iterations=5, time_s=0.5),
        BugResult("js-001", "javascript", "wrong-operator", solved=True, iterations=2, time_s=0.2),
        BugResult("js-002", "javascript", "missing-case", solved=False, iterations=5, time_s=0.5),
    ])
    assert report.fix_rate == 0.5
    assert report.solved == 2
    assert report.total == 4


def test_by_language_groups_correctly():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=True, iterations=1, time_s=0.1),
        BugResult("py-002", "python", "logic-error", solved=False, iterations=5, time_s=0.5),
        BugResult("js-001", "javascript", "wrong-operator", solved=True, iterations=1, time_s=0.1),
    ])
    by_lang = report.by_language()
    assert by_lang["python"]["total"] == 2
    assert by_lang["python"]["solved"] == 1
    assert by_lang["javascript"]["total"] == 1
    assert by_lang["javascript"]["solved"] == 1


def test_by_bug_type_groups_correctly():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=True, iterations=1, time_s=0.1),
        BugResult("py-005", "python", "off-by-one", solved=False, iterations=5, time_s=0.5),
        BugResult("py-002", "python", "wrong-operator", solved=True, iterations=2, time_s=0.2),
    ])
    by_type = report.by_bug_type()
    assert by_type["off-by-one"]["total"] == 2
    assert by_type["off-by-one"]["solved"] == 1
    assert by_type["wrong-operator"]["solved"] == 1


def test_avg_iterations_only_counts_solved():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=True, iterations=1, time_s=0.1),
        BugResult("py-002", "python", "logic-error", solved=True, iterations=3, time_s=0.3),
        BugResult("py-003", "python", "missing-case", solved=False, iterations=5, time_s=0.5),
    ])
    assert report.avg_iterations() == 2.0  # (1+3)/2, not counting the failed one


def test_avg_iterations_no_solved():
    report = _make_report([
        BugResult("py-001", "python", "off-by-one", solved=False, iterations=5, time_s=0.5),
    ])
    assert report.avg_iterations() == 0.0


def test_empty_report():
    report = BenchmarkReport()
    assert report.total == 0
    assert report.solved == 0
    assert report.fix_rate == 0.0
    assert report.avg_iterations() == 0.0


# ---------------------------------------------------------------------------
# run_benchmark integration test with mock LLM
# ---------------------------------------------------------------------------

def test_run_benchmark_mock_llm_solves_python_bugs():
    """Mock LLM is a perfect oracle -- should fix all Python bugs."""
    llm = make_mock_llm(PYTHON_BUGS)
    report = run_benchmark(PYTHON_BUGS, llm, max_iterations=3)
    assert report.total == 5
    # Mock LLM knows the fixed code, so all should be solved
    assert report.solved == 5
    assert report.fix_rate == 1.0


def test_run_benchmark_returns_correct_bug_ids():
    llm = make_mock_llm(PYTHON_BUGS)
    report = run_benchmark(PYTHON_BUGS[:2], llm, max_iterations=3)
    ids = {r.bug_id for r in report.results}
    assert "py-001" in ids
    assert "py-002" in ids


def test_run_benchmark_records_iterations():
    llm = make_mock_llm(PYTHON_BUGS)
    report = run_benchmark(PYTHON_BUGS[:1], llm, max_iterations=3)
    assert report.results[0].iterations >= 1


def test_run_benchmark_records_timing():
    llm = make_mock_llm(PYTHON_BUGS)
    report = run_benchmark(PYTHON_BUGS[:1], llm, max_iterations=3)
    assert report.results[0].time_s >= 0
    assert report.total_time_s >= 0
