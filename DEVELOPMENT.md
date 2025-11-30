# Development Log - SQL Query Agent

**Project:** Mate Security Take-Home Assignment
**Goal:** Build an intelligent SQL query generation agent using Agno framework
**Implementation Plan:** See [context/implementation-context.md](context/implementation-context.md)

---

## Progress Tracker

### Phase 1: Setup & Foundation ✅ COMPLETE
**Started:** 2025-11-30
**Completed:** 2025-11-30
**Duration:** ~2 hours

#### What We Built

**1. Project Setup**
- ✅ Created `.gitignore` for Python/venv/IDE files
- ✅ Defined dependencies in `requirements.txt` (production) and `requirements-dev.txt` (testing)
- ✅ Created directory structure (`src/agents`, `src/retrieval`, `src/utils`, `tests/`, `experiments/`)
- ✅ Added `setup.sh` for one-command installation

**2. Core Utilities**
- ✅ **Schema Loader** (`src/utils/schema_loader.py`)
  - Load and validate JSON schemas
  - Format schemas for LLM context
  - Build table descriptions for embeddings
  - Supports single file or directory of schema files

- ✅ **SQL Validator** (`src/utils/validator.py`)
  - Validate SQL syntax (SELECT/FROM structure)
  - Check balanced delimiters (quotes, parentheses)
  - Verify table names exist in schemas
  - Detect dangerous operations (DROP, DELETE, etc.)
  - Validate field names against schemas
  - Return structured validation results with errors/warnings

**3. Retrieval Systems**
- ✅ **Keyword Retrieval** (`src/retrieval/keyword_retrieval.py`)
  - Build keyword index from table/field names and descriptions
  - Score tables based on keyword overlap
  - Fallback strategy when no matches found
  - Method to find matching fields within tables

- ✅ **Semantic Retrieval** (`src/retrieval/semantic_retrieval.py`)
  - Use OpenAI `text-embedding-3-small` for embeddings
  - Pre-compute and cache table embeddings (saves API calls)
  - Calculate cosine similarity between question and tables
  - Return top-K most relevant tables
  - Bonus: Find similar tables (useful for JOIN suggestions)

**4. Agent Implementation**
- ✅ **Base Agent** (`src/agents/base.py`)
  - Abstract base class for all agent variants
  - Pydantic `SQLQueryResponse` model with structured output
  - Error handling utilities

- ✅ **Semantic Agent** (`src/agents/semantic_agent.py`)
  - Main implementation using Agno framework
  - Integrates semantic retrieval + schema formatting + validation
  - Uses Claude 3.5 Sonnet by default (configurable)
  - Confidence scoring based on complexity/ambiguity
  - Reasoning steps for transparency
  - Validation feedback integrated into responses

**5. User Interface**
- ✅ **Main CLI** (`main.py`)
  - Command-line interface with argparse
  - `--explain` flag to show retrieval process
  - `--json` flag for structured output
  - `--model` to select LLM (Claude/GPT-4)
  - `--top-k` to configure table retrieval count
  - Human-readable formatted output by default

#### Architecture Decisions

**Decision 1: Semantic Retrieval as Primary**
- **Choice:** Using embedding-based semantic search (Variant B from plan)
- **Rationale:** Better accuracy than keyword matching, acceptable latency
- **Tradeoff:** ~~Requires OpenAI API for embeddings~~ Uses local sentence-transformers (no API needed)

**Decision 2: Single Agent with Structured Output**
- **Choice:** One agent with Pydantic response model vs. multi-agent pipeline
- **Rationale:** Simpler, faster, easier to debug. Multi-agent can be added as Variant C.
- **Agno Pattern:** Using `response_model=SQLQueryResponse` for structured JSON output

**Decision 3: Pre-compute & Cache Embeddings**
- **Choice:** Compute all table embeddings once, cache to pickle file
- **Rationale:** 20 tables × $0.00002 = negligible cost, but caching makes dev iteration free
- **Implementation:** `embeddings_cache/table_embeddings.pkl` (git-ignored)
- **UPDATE (2025-11-30):** Revised to use local sentence-transformers instead of OpenAI
  - Model: `multi-qa-mpnet-base-dot-v1` (~400MB, downloaded once)
  - Reason: Only have Anthropic API key, local is faster/free/offline-capable
  - Cache now includes model name to detect model changes

**Decision 4: Validation Integrated, Not Blocking**
- **Choice:** Validate SQL but don't block response; include warnings in output
- **Rationale:** LLM can make small syntax errors that don't invalidate the query semantics
- **User Experience:** User sees query + validation feedback, can decide

#### Git Commit History
```
d0f3fae Add setup script for easy installation
79268c2 Add main CLI entrypoint
0eb0b26 Implement base agent and semantic agent
d4bae0f Implement SQL query validator
3b61e90 Implement semantic retrieval with embeddings
2367060 Implement keyword-based table retrieval
1ddb365 Add schema loader and project structure
7740348 Add project setup files
```

#### Files Created (9 files)
1. `.gitignore`
2. `requirements.txt`
3. `requirements-dev.txt`
4. `src/utils/schema_loader.py` (153 lines)
5. `src/utils/validator.py` (323 lines)
6. `src/retrieval/keyword_retrieval.py` (176 lines)
7. `src/retrieval/semantic_retrieval.py` (208 lines)
8. `src/agents/base.py` (66 lines)
9. `src/agents/semantic_agent.py` (211 lines)
10. `main.py` (165 lines)
11. `setup.sh` (126 lines)

**Total:** ~1,428 lines of code (excluding schemas)

---

## Next Steps

### Phase 2: Testing (Next)
**Priority:** HIGH - Critical for demonstrating production readiness

**Unit Tests to Write:**
1. ✅ `tests/test_schema_loader.py` - Schema loading and validation
2. ✅ `tests/test_retrieval.py` - Keyword & semantic retrieval (most important!)
3. ✅ `tests/test_validation.py` - SQL validation logic
4. ✅ `tests/test_agents.py` - End-to-end agent tests

**Why Testing Matters for Mate Security:**
- Demonstrates production-ready thinking
- Can isolate whether errors come from retrieval (wrong tables) vs generation (bad SQL)
- Shows we can measure retrieval precision independently
- Required for experimental evaluation framework

### Phase 3: Example Queries
**Required:** 5 example queries from assignment
1. "Show me all high-severity security events from the last 24 hours"
2. "Which users had the most failed login attempts this month?"
3. "Find all suspicious file access events related to sensitive documents"
4. "What are the top 10 most common security event types?"
5. "Show me events where the same IP address triggered multiple alerts"

### Phase 4: Experimental Framework (Differentiator)
**Goal:** Compare retrieval strategies with data

**Components:**
1. Test case generator (synthetic questions + expected SQL)
2. Evaluator (correctness, latency, cost, retrieval precision)
3. Run experiments comparing keyword vs. semantic retrieval
4. Generate comparison.md with findings

**Key Metrics:**
- Correctness: % semantically correct queries
- Latency: End-to-end time per query
- Cost: Total tokens per query
- Retrieval Precision: % retrieved tables actually used

### Phase 5: Documentation & Video
1. Complete README.md with setup, architecture, examples
2. Record 5-minute video demo
3. Final polish

---

## Technical Notes

### Environment Setup
**Python Version:** 3.10+ required (for Pydantic 2.x)
**Package Manager:** pip + venv (standard library)

**Required Environment Variables:**
```bash
ANTHROPIC_API_KEY=sk-ant-...   # For Claude agent (only API key needed)
# Note: Embeddings use sentence-transformers (local, no API key required)
```

**Installation:**
```bash
./setup.sh
# or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Dataset
- **Location:** `schemas/dataset.json`
- **Tables:** 20 security event tables
- **Categories:** endpoint, network, authentication, cloud, email, threat_intel, vulnerability, dlp, security_ops, application, asset
- **Complexity:** 12-25 fields per table (not the 300 mentioned in assignment, but realistic for demo)

### Known Limitations
1. **No actual database:** Agent generates SQL but doesn't execute
2. **Validation is heuristic:** Regex-based, not a full SQL parser
3. **Single retrieval strategy:** Only semantic (keyword available but not wired to CLI yet)

### Questions/Decisions Pending
- [ ] Should we add baseline agent (keyword retrieval) as CLI option?
- [ ] How many test cases to generate for experiments? (20-25 per plan)
- [ ] Video format: Screen recording + voiceover or live demo?

---

## Time Tracking

| Phase | Estimated | Actual | Status |
|-------|-----------|--------|--------|
| Phase 1: Setup & Foundation | 1.5 hrs | ~2 hrs | ✅ Complete |
| Phase 2: Testing | 1.5 hrs | - | 🔄 Next |
| Phase 3: Examples | 0.5 hrs | - | ⏳ Pending |
| Phase 4: Experiments | 2 hrs | - | ⏳ Pending |
| Phase 5: Documentation | 1 hr | - | ⏳ Pending |
| **Total** | **6.5 hrs** | **~2 hrs** | **31% complete** |

---

## Lessons Learned

### What Went Well
1. **Structured planning:** Having implementation-context.md made coding straightforward
2. **Incremental commits:** Each feature committed separately = clear history
3. **Agno framework:** Structured outputs with Pydantic made response handling clean
4. **Caching embeddings:** Will save time/money during testing and iteration

### Challenges
1. **None so far** - Phase 1 went smoothly

### For Video Demo
**Key Messages to Emphasize:**
1. **Core Solution:** Semantic search + single agent balances accuracy and speed
2. **Production Thinking:** Validation, error handling, caching, comprehensive testing
3. **Data-Driven:** Will show experimental comparison of retrieval strategies
4. **Mate Security Alignment:** Addresses "slow/inaccurate agents" pain point with measurable improvements

---

*Last Updated: 2025-11-30 - Phase 1 Complete*
