# debug-agent

An autonomous debugging agent that fixes bugs in Python, JavaScript, and Go using a ReAct (Reason + Act) loop. Given buggy code and failing tests, the agent iteratively proposes fixes, runs the tests, observes the output, and retries until the tests pass or it gives up.

Built from scratch — no LangChain, no AutoGen. Every component (bug corpus, sandboxed executor, ReAct loop, test writer, benchmark harness) is implemented directly.

---

## What it does

1. **Takes buggy code + failing tests** as input
2. **Reasons** about what the bug is based on the test output
3. **Proposes a fix** and runs the tests in a sandboxed subprocess
4. **Observes** the result — if tests still fail, feeds the error back and retries
5. **Repeats** until tests pass or max iterations reached

When no tests are provided, the **test writer** module generates them first, validates they actually catch the bug, then passes them to the fix loop.

---

## Architecture

```
debug-agent/
├── corpus/
│   └── corpus.py          <- 15 bugs across Python/JS/Go (Phase 1)
├── executor/
│   └── executor.py        <- sandboxed subprocess runner, timeout (Phase 2)
├── agent/
│   └── agent.py           <- ReAct loop, prompt builder, response parser (Phase 3)
├── test_writer/
│   └── test_writer.py     <- LLM-generated tests with validation (Phase 4)
├── benchmark/
│   └── run_benchmark.py   <- fix rate, by-language/type breakdown (Phase 5)
└── tests/                 <- 76 tests, all passing
```

---

## Phases

**Phase 1: Bug corpus — done**

- [x] `corpus/corpus.py` — 15 hand-crafted bugs across Python (5), JavaScript (5), and Go (5). Each bug has: buggy code, failing tests, fixed code, bug type, and description.
- [x] Bug types covered: `off-by-one`, `wrong-operator`, `missing-case`, `logic-error`, `boundary`, `wrong-return`.
- [x] Every Python bug verified: buggy code fails its tests, fixed code passes. 20 tests.

**Phase 2: Sandboxed executor — done**

- [x] `executor/executor.py` — runs solution code + test code in a subprocess per language. Each run gets its own temp directory, cleaned up afterward.
- [x] Hard timeout (default 10s) kills infinite loops without hanging the agent.
- [x] Supports Python (`python3`), JavaScript (`node`), Go (`go test`). JS/Go tests skipped if runtime not installed.
- [x] 14 tests: correct code passes, buggy code fails, syntax errors handled gracefully, infinite loops time out.

**Phase 3: ReAct agent loop — done**

- [x] `agent/agent.py` — the core Reason+Act loop. Each iteration: build prompt with full history → call LLM → parse response (reasoning + fixed code) → run tests → observe → repeat.
- [x] Prompt includes original buggy code, test suite, and all previous attempts with their test output so the agent can learn from failed fixes.
- [x] Response parser handles well-formed and malformed LLM responses gracefully (missing REASONING, no code block, etc.).
- [x] LLM backend is configurable: `make_llama_backend()` calls the llama-inference HTTP server; `make_hf_backend()` calls the HuggingFace Inference API; any callable works.
- [x] 17 tests using mock LLM: solves in 1 iteration, retries on wrong fix, gives up after max_iterations, records all attempts.

**Phase 4: Test writer — done**

- [x] `test_writer/test_writer.py` — when no tests are provided, generates a test suite using an LLM, then validates that the tests actually catch the bug (at least one test must fail against the buggy code). Tests that pass on buggy code are flagged as `useful=False`.
- [x] 13 tests: prompt construction, response parsing, useful/not-useful validation.

**Phase 5: Benchmark harness — done**

- [x] `benchmark/run_benchmark.py` — runs the agent on the full corpus and reports: overall fix rate, fix rate by language, fix rate by bug type, average iterations to fix, and detailed failure analysis.
- [x] `--mock` flag uses a perfect oracle LLM (knows the fixed code) for CI runs without an API key.
- [x] `--bug py-001` runs on a single bug for debugging the agent.
- [x] `--output results.json` saves full results as JSON.
- [x] 12 tests: BenchmarkReport metrics, mock LLM achieves 100% fix rate on Python bugs.

**Phase 6: CLI + README — planned**

---

## Running tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install pytest
python -m pytest tests/ -v   # 76 tests
```

---

## Running the benchmark

With mock LLM (no API key, tests the harness):

```bash
python benchmark/run_benchmark.py --mock
```

With HuggingFace Inference API:

```bash
export HF_API_KEY=your_key
python benchmark/run_benchmark.py
```

Single bug:

```bash
python benchmark/run_benchmark.py --mock --bug py-001 --verbose
```

---

## LLM backends

The agent supports two backends out of the box:

**llama-inference** (local, from [sujanuj/llama-inference](https://github.com/sujanuj/llama-inference)):
```python
from agent.agent import make_llama_backend, DebugAgent
llm = make_llama_backend(host="127.0.0.1", port=8080)
agent = DebugAgent(llm=llm)
```

**HuggingFace Inference API** (cloud, free tier):
```python
from agent.agent import make_hf_backend, DebugAgent
llm = make_hf_backend(api_key="your_key")
agent = DebugAgent(llm=llm)
```

Any callable `(prompt: str) -> str` works as a backend.

---

## Example

```python
from agent.agent import DebugAgent
from corpus.corpus import get_bug

bug = get_bug("py-001")
agent = DebugAgent(llm=your_llm, max_iterations=5, verbose=True)
session = agent.debug(
    bug_id=bug.id,
    language=bug.language,
    buggy_code=bug.buggy_code,
    test_code=bug.test_code,
)
print(f"Solved: {session.solved}")
print(f"Iterations: {session.num_iterations}")
print(f"Fixed code:\n{session.final_code}")
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)
