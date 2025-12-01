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

### Phase 2: Testing ✅ COMPLETE
**Started:** 2025-11-30
**Completed:** 2025-11-30
**Duration:** ~1 hour

#### What We Built

**Comprehensive Unit Test Suite - 138 Tests Passing**

**1. Schema Loader Tests** (`tests/test_schema_loader.py`) - 23 tests
- ✅ Loading schemas from single files and directories
- ✅ Schema validation (missing fields, invalid types, malformed JSON)
- ✅ Table description generation with configurable parameters
- ✅ Context limit truncation with embedding model tokenization
- ✅ LLM formatting with field limits
- **Key Test:** Validates that `get_table_description` includes all fields by default (no arbitrary caps that lose semantic signal)

**2. Keyword Retrieval Tests** (`tests/test_keyword_retrieval.py`) - 27 tests
- ✅ Keyword index building from table/field names and descriptions
- ✅ Tokenization (case handling, short token filtering, delimiter splitting)
- ✅ Top-k retrieval with scoring, normalization, and fallback strategies
- ✅ Field matching by keywords in names and descriptions
- **Key Test:** Validates scoring mechanism prioritizes tables with more keyword matches

**3. Semantic Retrieval Tests** (`tests/test_semantic_retrieval.py`) - 18 tests
- ✅ Initialization and SentenceTransformer model loading
- ✅ Embedding precomputation for all tables
- ✅ Cache saving/loading with model name verification
- ✅ Cache invalidation on model mismatch
- ✅ Similarity-based retrieval and ranking
- ✅ Cosine similarity calculations (identical, orthogonal, opposite vectors)
- **Key Test:** Validates cache is properly invalidated when embedding model changes

**4. SQL Validator Tests** (`tests/test_validator.py`) - 55 tests
- ✅ Basic SQL syntax validation (SELECT/FROM structure)
- ✅ Balanced delimiter checking (quotes, parentheses, escape sequences)
- ✅ Table existence validation against schemas
- ✅ Dangerous operation detection (DROP, DELETE, UPDATE, INSERT, etc.)
- ✅ Field validation with strict and non-strict modes
- ✅ SELECT field extraction with aliases and table qualifiers
- ✅ Comprehensive validation with errors and warnings
- **Key Test:** Validates that unclosed quotes/parentheses are correctly detected

**5. Semantic Agent Tests** (`tests/test_semantic_agent.py`) - 15 tests
- ✅ Agent initialization with schema loading and dependencies
- ✅ Schema context building for LLM with relevance scores
- ✅ Query generation with validation integration
- ✅ Confidence adjustment based on validation results
- ✅ Error handling with graceful fallback
- ✅ Retrieval explanation for debugging
- **Key Test:** Validates that validation errors reduce confidence score to ≤0.5

#### Testing Strategy

**Mocking for Speed and Reliability:**
- All external dependencies mocked (SentenceTransformer, Agno Agent, Claude API)
- Tests run in ~4 seconds total, no API keys required
- Deterministic behavior for CI/CD pipelines

**Test Organization:**
- Each component in separate file for clarity
- Fixtures for common test data (schemas, mock responses)
- Descriptive test names explaining what is being validated
- Comprehensive edge case coverage (empty inputs, errors, invalid data)

#### Git Commit History (Phase 2)
```
748c8ee Add comprehensive unit tests for semantic agent
5399ef0 Add comprehensive unit tests for SQL validator
3492753 Add comprehensive unit tests for semantic retrieval
3c8aa24 Add comprehensive unit tests for keyword retrieval
868f285 Add comprehensive unit tests for schema loader
8cdec58 Refactor get_table_description with configurable parameters
```

#### Files Created (5 test files)
1. `tests/test_schema_loader.py` (306 lines)
2. `tests/test_keyword_retrieval.py` (267 lines)
3. `tests/test_semantic_retrieval.py` (353 lines)
4. `tests/test_validator.py` (424 lines)
5. `tests/test_semantic_agent.py` (404 lines)

**Total:** ~1,754 lines of test code

#### Why This Testing Matters for Mate Security

**1. Production Readiness:**
- Comprehensive test coverage demonstrates professional engineering practices
- Fast test suite enables rapid iteration and refactoring
- Clear test names serve as documentation of expected behavior

**2. Debugging Capability:**
- Can isolate whether errors come from:
  - Retrieval (wrong tables selected)
  - Generation (invalid SQL syntax)
  - Validation (false positives/negatives)
- Each component tested independently with mocked dependencies

**3. Quality Assurance:**
- Edge cases covered (empty inputs, nonexistent tables, malformed queries)
- Validation of key business logic (scoring, confidence adjustment, caching)
- Regression prevention as we add new features

**4. Experimental Foundation:**
- Test infrastructure ready for Phase 4 evaluation framework
- Can measure retrieval precision with controlled test cases
- Automated testing enables reproducible experiments

---

### Phase 3: Experimental Framework ✅ COMPLETE
**Started:** 2025-12-01
**Completed:** 2025-12-01
**Duration:** ~2 hours

#### What We Built

**1. Keyword Agent Implementation**
- ✅ **Keyword Agent** (`src/agents/keyword_agent.py`)
  - Implements keyword-based retrieval variant for comparison
  - Nearly identical to SemanticAgent except uses `KeywordRetrieval` instead of `SemanticRetrieval`
  - Same LLM (Claude), same validation, same pipeline
  - Enables direct comparison of retrieval strategies

- ✅ **Keyword Agent Tests** (`tests/test_keyword_agent.py`) - 15 tests
  - Created by copying semantic agent tests and substituting class names
  - All tests passing
  - Validates initialization, retrieval, query generation, validation integration, error handling

**2. Experimental Utilities**
- ✅ **Metrics Module** (`experiments/utils/metrics.py`)
  - `calculate_retrieval_precision()`: % of retrieved tables actually used
  - `extract_tables_from_sql()`: Parse table names from SQL using regex
  - `calculate_aggregate_metrics()`: Average correctness, latency, tokens, precision
  - Supports breakdown by complexity and category

- ✅ **LLM Judge** (`experiments/utils/llm_judge.py`)
  - LLM-as-judge for semantic SQL correctness evaluation
  - Uses Agno Agent with structured output (Pydantic `CorrectnessEvaluation`)
  - Returns score (0.0-1.0), reasoning, and list of issues
  - Scoring rubric: 1.0=perfect, 0.8-0.9=minor issues, 0.5-0.7=partial, 0.0-0.4=wrong
  - Handles edge cases (empty SQL, LLM errors)

**3. Test Case Generator**
- ✅ **Test Case Generator** (`experiments/generate_test_cases.py`)
  - Uses Agno Agent to generate synthetic test cases
  - Structured output with Pydantic models (`TestCase`, `TestCaseBatch`)
  - Three complexity levels: simple (single table), medium (aggregations), complex (JOINs)
  - Validates generated SQL before accepting
  - Saves to JSON for reuse (avoid regeneration costs)
  - Configurable counts per complexity level

**4. Experiment Runner**
- ✅ **Generalized Experiment Runner** (`experiments/run_experiments.py`)
  - **Accepts agents dynamically** via dependency injection (not hardcoded!)
  - Supports any combination of agents through CLI
  - Measures all metrics: correctness (LLM judge), latency, tokens, retrieval precision
  - Runs all test cases through all agents
  - Saves detailed results + aggregate metrics by agent/complexity/category
  - Real-time progress feedback with emojis (✅/⚠️/❌)

**5. Report Generator**
- ✅ **Report Generator** (`experiments/generate_report.py`)
  - Loads experiment results from JSON
  - Generates comprehensive markdown report with:
    - Executive summary with winner
    - Methodology section
    - Overall results table
    - Breakdown by complexity and category
    - Failure analysis with common issues and examples
    - Key insights comparing agents
    - Production recommendations and future improvements

**6. Documentation**
- ✅ **Experiments README** (`experiments/README.md`)
  - Complete usage guide with quick start
  - Architecture overview and design principles
  - Instructions for adding new agents
  - Cost estimates (~$0.70 per complete run)
  - Output structure examples
  - Troubleshooting guide

#### Architecture Decisions

**Decision 1: Generalized Agent Support**
- **Choice:** Experiment runner accepts agents as parameters (dependency injection) vs. hardcoding
- **Rationale:** Enables comparison of arbitrary agent architectures, not just keyword vs semantic
- **Implementation:**
  ```python
  def __init__(self, test_cases_path: str, agents: Dict[str, BaseAgent]):
      self.agents = agents  # Not hardcoded!
  ```
- **Benefit:** Easy to add new agent variants for future experiments

**Decision 2: LLM-as-Judge for Correctness**
- **Choice:** Use LLM to evaluate SQL correctness vs. executing queries against database
- **Rationale:** No database to execute against; semantic evaluation handles equivalent queries
- **Tradeoff:** Judge costs ~$0.08 per 25 test cases, but more flexible than exact string matching

**Decision 3: Test Cases Generated Once, Reused**
- **Choice:** Generate test cases with LLM, save to JSON, reuse across experiments
- **Rationale:** Avoids regeneration costs (~$0.15 each time), ensures consistency
- **Benefit:** Can cheaply iterate on agent improvements with same test suite

**Decision 4: Comprehensive Metrics**
- **Choice:** Measure correctness, latency, tokens, retrieval precision (not just accuracy)
- **Rationale:** Production readiness requires understanding speed/cost tradeoffs
- **Benefit:** Can optimize for different goals (accuracy vs. speed vs. cost)

#### Git Commit History (Phase 3)
```
2eaaf44 Add comprehensive documentation for experimental framework
613bece Add generalized experiment runner and report generator
fb95b58 Add test case generator with LLM-based generation
8f3e2c4 Add metrics and LLM judge utilities
a72c0b5 Add comprehensive unit tests for keyword agent
d3f1c8a Implement keyword agent for comparison experiments
```

#### Files Created (8 files)
1. `src/agents/keyword_agent.py` (200 lines)
2. `tests/test_keyword_agent.py` (404 lines)
3. `experiments/utils/metrics.py` (~150 lines)
4. `experiments/utils/llm_judge.py` (~150 lines)
5. `experiments/generate_test_cases.py` (~300 lines)
6. `experiments/run_experiments.py` (~330 lines)
7. `experiments/generate_report.py` (~410 lines)
8. `experiments/README.md` (~260 lines)

**Total:** ~2,204 lines of code

**Updated Test Count:** 153 tests passing (138 from Phase 2 + 15 keyword agent tests)

#### Why This Framework Matters for Mate Security

**1. Data-Driven Decision Making:**
- Quantitative comparison of retrieval strategies (not just intuition)
- Clear metrics show which approach is better and by how much
- Can measure impact of architectural changes objectively

**2. Production Confidence:**
- LLM judge validates correctness before deployment
- Failure analysis identifies weak spots to address
- Latency/cost metrics inform deployment decisions

**3. Extensibility:**
- Easy to add new agent architectures for comparison
- Test suite can grow with more edge cases
- Framework supports A/B testing in production

**4. Addressing Mate Security Pain Points:**
- "Slow agents" → Latency metrics quantify speed
- "Inaccurate agents" → Correctness scores measure accuracy
- "No data to back claims" → Full experimental reports with statistics

---

## Next Steps

### Phase 4: Run Experiments & Documentation (Next)
**Priority:** HIGH - Final deliverables
**Goal:** Generate experimental results and complete documentation

**Immediate Tasks:**
1. Run complete experimental pipeline
   - Generate test cases (simple=10, medium=10, complex=5)
   - Run experiments comparing keyword vs semantic agents
   - Generate comparison.md report with findings
2. Complete README.md with setup, architecture, examples
3. Test example queries from assignment:
   - "Show me all high-severity security events from the last 24 hours"
   - "Which users had the most failed login attempts this month?"
   - "Find all suspicious file access events related to sensitive documents"
   - "What are the top 10 most common security event types?"
   - "Show me events where the same IP address triggered multiple alerts"
4. Record 5-minute video demo
5. Final polish and submission

**Quality Assurance & Improvements:**
- [x] ~~Consider caching mechanisms for better performance~~ ✅ **DONE**: Enabled Anthropic prompt caching across all agents
  - Caches system instructions and tool definitions (90% cost reduction on cached tokens)
  - Estimated savings: $0.30-0.50 per experiment run
  - 1-hour TTL suitable for experiment sessions
- [ ] QA homework one by one with help of Claude - check all optional requirements
- [ ] Semantic agent and keyword agent are nearly identical - change them to be configurable?
- [ ] Look at token usage in more detail: Maybe make token usage part of base agent?
- [ ] Go over metrics.py, some of the metrics are too simple
- [ ] Improve generate_report.py - final report is over the top and model usage is hardcoded
- [ ] Check if really need dev requirements and if yes, if it is updated correctly
- [ ] Check about using Tools as part of architecture
- [ ] QA test database schema and start experimenting with real queries
- [ ] Write an "additional features" report with ideas of what else can be done

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
4. **Hardcoded Anthropic provider:** Agent imports and uses `Claude` class directly from `agno.models.anthropic`, making it inflexible for switching providers (e.g., OpenAI, local models). Should use dynamic provider selection based on model string.

### Questions/Decisions Pending
- [ ] Should we add baseline agent (keyword retrieval) as CLI option for experimental comparison?
- [ ] How many test cases to generate for experiments? (20-25 per plan)

---

## Time Tracking

| Phase | Estimated | Actual | Status |
|-------|-----------|--------|--------|
| Phase 1: Setup & Foundation | 1.5 hrs | ~2 hrs | ✅ Complete |
| Phase 2: Testing | 1.5 hrs | ~1 hr | ✅ Complete |
| Phase 3: Experimental Framework | 2 hrs | ~2 hrs | ✅ Complete |
| Phase 4: Run Experiments & Documentation | 1.5 hrs | - | 🔄 Next |
| **Total** | **6.5 hrs** | **~5 hrs** | **77% complete** |

---

## Lessons Learned

### What Went Well
1. **Structured planning:** Having implementation-context.md made coding straightforward
2. **Incremental commits:** Each feature committed separately = clear history
3. **Agno framework:** Structured outputs with Pydantic made response handling clean
4. **Caching embeddings:** Will save time/money during testing and iteration
5. **Comprehensive testing:** 138 tests completed in ~1 hour, faster than estimated
6. **Mocking strategy:** All external dependencies mocked = fast, reliable, no API keys needed
7. **Test-driven bug finding:** Writing tests revealed edge cases we hadn't considered (e.g., arbitrary field caps in schema loader)
8. **Prompt caching:** Agno's built-in Anthropic prompt caching (via `cache_system_prompt=True`) is much simpler than custom caching - reduces costs by ~90% on cached tokens with minimal code changes

### Challenges
1. **Test fixture design:** Initial test for truncation failed because sample schema was too small - had to create larger test data
2. **Tokenizer behavior:** Underscores treated as word characters by `\w` regex, not as delimiters - tests needed adjustment
3. **Generalization requirement:** Initial experiment runner hardcoded agents; refactored to use dependency injection after feedback
4. **Balancing comprehensiveness:** Report generator creates very detailed output - may need simplification for readability

### For Video Demo
**Key Messages to Emphasize:**
1. **Core Solution:** Semantic search + single agent balances accuracy and speed
2. **Production Thinking:** Validation, error handling, caching, comprehensive testing
3. **Data-Driven:** Will show experimental comparison of retrieval strategies
4. **Mate Security Alignment:** Addresses "slow/inaccurate agents" pain point with measurable improvements

---

*Last Updated: 2025-12-01 - Phase 3 Complete (153 tests passing, experimental framework ready)*
