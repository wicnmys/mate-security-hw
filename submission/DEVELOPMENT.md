# Development Log - SQL Query Agent

**Project:** Mate Security Take-Home Assignment
**Goal:** Build an intelligent SQL query generation agent using Agno framework

---

## Progress Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Setup & Foundation | ~2 hrs | ✅ Complete |
| Phase 2: Testing | ~1 hr | ✅ Complete |
| Phase 3: Experimental Framework | ~2 hrs | ✅ Complete |
| Phase 4: Run Experiments | ~1 hr | ✅ Complete |
| Phase 5: Code Review Implementation | ~1 hr | ✅ Complete |
| Phase 6: Integrity Testing | ~1.5 hrs | ✅ Complete |
| Phase 7: ReAct Agent | ~2 hrs | ✅ Complete |
| Phase 8: Report Generation Refactor | ~1 hr | ✅ Complete |
| **Total** | **~11.5 hrs** | **100%** |

---

## Phase 1: Setup & Foundation ✅

**Built:**
- Project structure with `src/agents`, `src/retrieval`, `src/utils`, `tests/`, `experiments/`
- **Schema Loader** - Load/validate JSON schemas, format for LLM context
- **SQL Validator** - Syntax validation, table/field checking, dangerous operation detection
- **Keyword Retrieval** - Keyword-based table matching with scoring
- **Semantic Retrieval** - Embedding-based search using `sentence-transformers` (local, no API)
- **Semantic Agent** - Main agent using Agno framework with Claude, structured output via Pydantic
- **CLI** (`main.py`) - Command-line interface with `--explain`, `--json`, `--model` flags

**Key Decisions:**
- Semantic retrieval as primary (better accuracy than keyword matching)
- Local embeddings with sentence-transformers (no OpenAI API needed)
- Pre-computed embeddings cached to disk for fast iteration

---

## Phase 2: Testing ✅

**Built:** Comprehensive test suite with **138 tests**

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_schema_loader.py` | 23 | Schema loading, validation, formatting |
| `test_keyword_retrieval.py` | 27 | Index building, scoring, fallback |
| `test_semantic_retrieval.py` | 18 | Embeddings, caching, similarity |
| `test_validator.py` | 55 | Syntax, delimiters, dangerous ops |
| `test_semantic_agent.py` | 15 | Agent init, query generation, validation |

**Strategy:** All external dependencies mocked for fast (~4s), reliable tests with no API keys required.

---

## Phase 3: Experimental Framework ✅

**Built:**
- **Keyword Agent** - Alternative retrieval strategy for comparison
- **Metrics Module** - Retrieval precision, latency, token usage
- **LLM Judge** - Semantic SQL correctness evaluation (0.0-1.0 scoring)
- **Test Case Generator** - LLM-generated test cases at 3 complexity levels
- **Experiment Runner** - Generalized runner accepting any agent via dependency injection
- **Report Generator** - Markdown reports with methodology, results, failure analysis

**Key Decision:** LLM-as-judge for correctness evaluation (no database to execute against, handles semantic equivalence).

---

## Phase 4: Run Experiments ✅

**Generated:** 21 test cases (10 simple, 9 medium, 2 complex)

**Results - Keyword vs Semantic:**

| Metric | Keyword | Semantic | Winner |
|--------|---------|----------|--------|
| Correctness | 58.8% | **62.9%** | Semantic |
| Latency | 10.3s | 10.7s | Similar |
| Retrieval Precision | 79.4% | **87.6%** | Semantic |

**Finding:** Semantic retrieval provides 4.1% better correctness and 8.2% better precision.

---

## Phase 5: Code Review Implementation ✅

Implemented all recommendations from code review:

| Issue | Fix |
|-------|-----|
| Agent code duplication (95% identical) | Created unified `SQLQueryAgent` with dependency injection |
| Pickle security risk | Changed to NumPy `.npz` format with `allow_pickle=False` |
| Print statements | Converted to `logging` module |
| Late imports | Moved to module level |
| Bare except blocks | Added specific exception handling |
| Magic numbers | Created `src/constants.py` |

**Result:** ~150 lines of duplicate code eliminated, security improved.

---

## Phase 6: Integrity Testing ✅

**Built:** Adversarial test framework with 6 categories:

| Category | Tests | Expected Behavior |
|----------|-------|-------------------|
| Prompt Injection | 10 | Reject injection attempts |
| Off-Topic | 10 | Refuse non-SQL questions |
| Dangerous SQL | 10 | Warn about destructive ops |
| Unanswerable | 10 | Acknowledge uncertainty |
| Malformed Input | 10 | Handle gracefully |
| PII Sensitive | 10 | Warn about sensitive data |

**Generated:** 60 integrity test cases for adversarial evaluation.

---

## Phase 7: ReAct Agent ✅

**Built:** ReAct (Reasoning + Acting) agent with iterative tool use:

| Tool | Purpose |
|------|---------|
| `retrieve_tables` | Search for relevant tables |
| `validate_sql` | Check syntax and schema |
| `submit_answer` | Submit final query |

**Results - All Agents Comparison:**

| Metric | Keyword | Semantic | ReAct |
|--------|---------|----------|-------|
| **Correctness** | 60.0% | 63.3% | **67.6%** |
| **Latency** | **9.9s** | 10.6s | 31.6s |
| Retrieval Precision | 80.2% | **93.7%** | 91.3% |

**Findings:**
- ReAct achieves highest correctness (+4.3% over semantic)
- Biggest improvement on medium complexity queries (68.3% vs 55.0%)
- Latency tradeoff: ~3x slower due to iterative tool calls

---

## Phase 8: Report Generation & Bug Fixes ✅

**Built:**
- Polymorphic judge system supporting correctness, integrity, and categorical evaluation
- Multi-judge report generation
- Fixed 0% integrity pass rate bug (field extraction issue)

**Final Integrity Results:**

| Agent | Pass Rate |
|-------|-----------|
| Semantic | **56.7%** |
| Keyword | 50.0% |
| ReAct | 36.7% |
| ReAct-v2 | 30.0% |

**Finding:** Simple single-pass agents outperform ReAct on safety tests.

---

## Final Test Count

- **269 unit tests** passing
- **13 integration tests** passing
- **Total: 282 tests**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Selection                           │
│         (keyword | semantic | react | react-v2)             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Retrieval Layer                            │
│     ┌─────────────────┐    ┌─────────────────┐              │
│     │ Keyword Search  │    │ Semantic Search │              │
│     │ (TF-IDF style)  │    │  (Embeddings)   │              │
│     └─────────────────┘    └─────────────────┘              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Generation                            │
│              (Claude via Agno Framework)                     │
│         Structured output with Pydantic models              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQL Validation                            │
│        (Syntax, schema, dangerous operation checks)          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   SQLQueryResponse                           │
│    {query, explanation, tables_used, confidence, warnings}  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Learnings

1. **Semantic > Keyword** for retrieval accuracy (+8% precision)
2. **ReAct improves correctness** but at significant latency cost (3x slower)
3. **Simple agents better for safety** - iterative reasoning can be exploited
4. **Embedding caching critical** for fast iteration during development
5. **LLM-as-judge** enables evaluation without database execution
6. **Dependency injection** enables clean agent comparison experiments

---

*Completed: 2025-12-02*
