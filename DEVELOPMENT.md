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

**3. Integration Tests for API Contract Validation**
- ✅ **Integration Test Suite** - 9 tests across 6 files
  - Purpose: Validate against real Anthropic API contracts (catch errors that mocks miss)
  - No mocking - actually instantiate `Claude`, `SemanticAgent`, `KeywordAgent`, `LLMJudge`
  - Marked with `@pytest.mark.integration` for selective execution
  - Skip if `ANTHROPIC_API_KEY` not set (safe for CI without API access)

- ✅ **Test Files Created:**
  1. `tests/integration/__init__.py` - Package initialization
  2. `tests/integration/conftest.py` - Shared fixtures
  3. `tests/integration/test_caching_parameters.py` - 3 tests validating Anthropic caching API
  4. `tests/integration/test_agent_initialization.py` - 4 tests (2 initialization + 2 query generation)
  5. `tests/integration/test_llm_judge.py` - 2 tests
  6. `pytest.ini` - Pytest marker configuration

- ✅ **Key Testing Pattern: Configurable Model Fixture**
  ```python
  @pytest.fixture
  def integration_model():
      """Defaults to 'claude-haiku-4-5' (cheap and fast).
      Override with INTEGRATION_TEST_MODEL environment variable."""
      return os.getenv('INTEGRATION_TEST_MODEL', 'claude-haiku-4-5')
  ```
  - All 9 tests accept `integration_model` parameter (no hardcoded models)
  - Default: Claude Haiku (~$0.10 per full test run)
  - Can override for Sonnet testing: `INTEGRATION_TEST_MODEL=claude-sonnet-4-5`

- ✅ **Haiku Structured Output Limitation Handled:**
  - Haiku doesn't support `output_format` parameter (required for Pydantic schemas)
  - Query generation tests skip gracefully with TODO comments:
  ```python
  # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
  # Check if newer versions of Haiku support this feature and remove skip if so
  if 'haiku' in integration_model.lower():
      pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")
  ```
  - With Haiku: 6 passed, 3 skipped (initialization tests pass, query tests skip)
  - With Sonnet: 9 passed, 0 skipped (all tests pass)

- ✅ **Agno 2.3.4 Caching Parameters Validated:**
  - Supported: `cache_system_prompt=True`, `cache_ttl=3600` (integer seconds)
  - NOT supported: `cache_tool_definitions` (raises TypeError as expected)
  - Integration test explicitly validates these contracts

- ✅ **Bug Fix: Missing load_dotenv() in generate_test_cases.py**
  - Issue: Integration tests worked (agents import modules that call `load_dotenv()`)
  - But `generate_test_cases.py` failed with "ANTHROPIC_API_KEY not set"
  - Root cause: Script doesn't import agent modules, so `.env` never loaded
  - Fix: Added explicit `load_dotenv()` call at top of script
  - Lesson: Standalone scripts need explicit environment loading

**Updated Test Count:** 168 tests total
- 159 unit tests (all passing)
- 9 integration tests (6 passed, 3 skipped on Haiku; 9 passed on Sonnet)

**Cost Estimates:**
- Unit tests: Free (all mocked, no API calls)
- Integration tests (Haiku): ~$0.10 per run
- Integration tests (Sonnet): ~$0.30 per run
- Recommendation: Use Haiku for regular testing, Sonnet for final validation

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
- [x] ~~Code review for structure and readability~~ ✅ **DONE**: Full code review completed
  - See detailed report: [`context/code-review.md`](context/code-review.md)
  - **Overall Grades:** Structure B+, Readability A-, Best Practices B+, Issues B
  - **Critical:** Agent code duplication (semantic/keyword agents 95% identical)
  - **Important:** Pickle security risk, print statements instead of logging, late imports
  - **Nice-to-have:** Magic numbers, long functions, unused imports
- [x] ~~QA homework one by one with help of Claude - check all optional requirements~~ ✅ **DONE**: Full QA completed
  - See detailed checklist: [`context/qa-checklist.md`](context/qa-checklist.md)
  - **Core Functionality:** 4/4 requirements met (Schema Understanding, NLP, SQL Generation, Agent Implementation)
  - **Expected Deliverables:** Documentation ✅, Test Cases ⚠️ (edge case demos missing from README)
  - **Bonus Points:** 5/6 implemented (Query Refinement not implemented)
  - **Key Gaps Identified:**
    - No user-facing edge case demonstrations in README (unit tests exist but not shown)
    - No off-topic question handling
    - SQL validation is regex-based, not AST parsing
    - Demo video not yet recorded
  - **Potential Improvements Documented:** 15+ suggestions for future enhancements
- [x] ~~Semantic agent and keyword agent are nearly identical - change them to be configurable?~~ **IDENTIFIED** in code review - extract common logic with dependency injection

**Remaining TODOs (sorted by priority):**

**Priority 1 - High Impact**
- [x] ~~**Improve code quality**~~ ✅ **DONE** - All issues from [code-review.md](context/code-review.md) addressed in Phase 5
- [x] ~~**Add examples to README**~~ ✅ **DONE** - Added 5 sample queries with actual outputs + 5 edge case demonstrations
- [x] ~~**Add SQL parser**~~ ✅ **DONE** - Replaced regex-based validation with `sqlglot` for proper AST parsing
  - Uses PostgreSQL dialect for accurate parsing
  - Falls back to regex for malformed SQL (graceful degradation)
  - All 153 unit tests passing

**Priority 2 - Architecture**
- [ ] **Pick one architectural improvement, build agent, run experiment** - Options: off-topic handling, multi-agent routing, or query refinement
- [ ] Check about using Tools as part of architecture

**Priority 3 - Polish**
- [ ] **Improve experiment outputs** - Simplify generate_report.py, remove hardcoded model usage
- [ ] Go over metrics.py - some metrics are too simple
- [ ] Look at token usage in more detail - maybe make token usage part of base agent?
- [ ] Check if really need dev requirements and if yes, if it is updated correctly

**Priority 4 - Nice to Have**
- [ ] QA test database schema and start experimenting with real queries
- [ ] Write an "additional features" report with ideas of what else can be done
- [ ] Record 5-minute video demo

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
2. **Single retrieval strategy:** Only semantic (keyword available but not wired to CLI yet)
3. **Hardcoded Anthropic provider:** Agent imports and uses `Claude` class directly from `agno.models.anthropic`, making it inflexible for switching providers (e.g., OpenAI, local models). Should use dynamic provider selection based on model string.
4. **No off-topic question handling:** Agent will attempt SQL generation for any question (no guardrails)
5. **No query refinement:** Single-pass generation without iterative feedback (bonus feature not implemented)

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
9. **Integration tests:** Caught API contract violations that unit tests with mocks missed (e.g., `cache_tool_definitions` parameter not supported)
10. **Configurable test fixtures:** `integration_model` fixture allows cost-effective testing with Haiku while supporting Sonnet validation

### Challenges
1. **Test fixture design:** Initial test for truncation failed because sample schema was too small - had to create larger test data
2. **Tokenizer behavior:** Underscores treated as word characters by `\w` regex, not as delimiters - tests needed adjustment
3. **Generalization requirement:** Initial experiment runner hardcoded agents; refactored to use dependency injection after feedback
4. **Balancing comprehensiveness:** Report generator creates very detailed output - may need simplification for readability
5. **Haiku model limitations:** Discovered Haiku doesn't support structured outputs during integration testing - added graceful skip logic with TODOs for future revisiting
6. **Environment loading inconsistency:** Integration tests worked but standalone scripts failed - learned that `load_dotenv()` must be explicit in scripts that don't import agent modules
7. **Hardcoded model references:** Had to scan all files and eliminate hardcoded model strings from test code to ensure configurability

### For Video Demo
**Key Messages to Emphasize:**
1. **Core Solution:** Semantic search + single agent balances accuracy and speed
2. **Production Thinking:** Validation, error handling, caching, comprehensive testing
3. **Data-Driven:** Will show experimental comparison of retrieval strategies
4. **Mate Security Alignment:** Addresses "slow/inaccurate agents" pain point with measurable improvements

---

**4. Experimental Results**
- ✅ **Test Cases Generated**: 21 test cases (10 simple, 9 medium, 2 complex)
  - LLM-generated using Claude Sonnet 4.5
  - Validated against schema before acceptance
  - Saved to `experiments/test_cases/generated_test_cases.json`

- ✅ **Experiments Complete**: Compared keyword vs semantic agents
  - 21 test cases × 2 agents = 42 total agent runs
  - LLM-as-judge evaluation for correctness
  - Measured: correctness, latency, tokens, retrieval precision

- ✅ **Results Summary**:
  | Metric | Keyword | Semantic | Winner |
  |--------|---------|----------|--------|
  | Correctness | 58.8% | **62.9%** | Semantic ✅ |
  | Latency | 10.3s | 10.7s | Similar |
  | Retrieval Precision | 79.4% | **87.6%** | Semantic ✅ |
  | Tokens | 200 | 203 | Similar |

- ✅ **Key Findings**:
  - Semantic retrieval provides 4.1% better correctness
  - Semantic retrieval achieves 8.2% better precision
  - Performance gap widest on simple queries (76.5% vs 69.5%)
  - Both agents struggle with complex multi-table queries
  - Latency comparable (~10-11 seconds per query)

- ✅ **Documentation Created**:
  - `experiments/comparison.md` - Detailed experimental report
  - `README.md` - Comprehensive project documentation
  - Nested progress bars added to experiment runner (tqdm)

**5. Documentation & UX Improvements**
- ✅ **README.md Created**: Comprehensive documentation including:
  - Quick start guide with installation steps
  - Architecture overview with ASCII diagram
  - Usage examples for security queries
  - Testing instructions (unit + integration)
  - Experimental framework documentation
  - Project structure and configuration
  - Performance metrics and cost estimates
  - Known limitations and future improvements

- ✅ **Nested Progress Bars**: Enhanced UX for experiments
  - Overall progress bar (blue): Total tests across all agents
  - Agent-specific progress bar (green): Current agent with live metrics
  - Real-time display: score, latency, complexity, status emoji
  - Elapsed/remaining time estimates

**Updated Test Count:** 168 tests total (159 unit + 9 integration) ✅
**Experimental Comparison:** Complete ✅
**Documentation:** Complete ✅

---

### Phase 5: Code Review Implementation ✅ COMPLETE
**Started:** 2025-12-01
**Completed:** 2025-12-01
**Duration:** ~1 hour

#### What We Built

Implemented all recommendations from [code-review.md](context/code-review.md):

**1. Critical: Extracted Common Agent Logic**
- ✅ Created `src/agents/sql_agent.py` with unified `SQLQueryAgent` class
- ✅ Added `Retriever` Protocol to `src/agents/base.py` for dependency injection
- ✅ Refactored `SemanticAgent` and `KeywordAgent` as thin wrappers (reduced from ~200 lines each to ~50 lines)
- ✅ Eliminated 95% code duplication between agents

**2. Important: Replaced Pickle with NumPy**
- ✅ Changed cache format from `.pkl` (pickle) to `.npz` (numpy)
- ✅ Uses `np.savez()` with `allow_pickle=False` for security
- ✅ Eliminates arbitrary code execution risk from malicious cache files

**3. Important: Converted Print to Logging**
- ✅ Added `logging` module to `semantic_retrieval.py`
- ✅ Replaced all `print()` statements with `logger.info()` and `logger.warning()`
- ✅ Library code now follows best practices for log integration

**4. Important: Moved Late Imports to Module Level**
- ✅ Moved `from src.utils.schema_loader import get_table_description` to top of file
- ✅ Eliminates hidden dependencies and improves code clarity

**5. Important: Specific Exception Handling**
- ✅ Replaced bare `except` blocks with specific exceptions
- ✅ Cache loading: `except (ValueError, KeyError, IOError)`
- ✅ Cache saving: `except (IOError, OSError)`

**6. Nice-to-have: Created Constants File**
- ✅ Created `src/constants.py` with:
  - `MIN_KEYWORD_LENGTH = 3`
  - `DEFAULT_TOP_K_TABLES = 5`
  - `DEFAULT_CACHE_TTL_SECONDS = 3600`
  - `DEFAULT_EMBEDDING_MODEL`
  - `DEFAULT_LLM_MODEL`
  - `DANGEROUS_SQL_OPERATIONS`

**7. Nice-to-have: Removed Unused Imports**
- ✅ Removed `import os` from `semantic_agent.py` and `keyword_agent.py`

#### Test Results
- ✅ **153 unit tests passing** (all existing tests updated for refactored code)
- ✅ **9 integration tests passing** (with `claude-sonnet-4-5`)
- ✅ Total: 162 tests passing

#### Files Modified
1. `src/agents/base.py` - Added `Retriever` Protocol (+20 lines)
2. `src/agents/sql_agent.py` - New unified agent class (~180 lines)
3. `src/agents/semantic_agent.py` - Simplified to thin wrapper (-156 lines)
4. `src/agents/keyword_agent.py` - Simplified to thin wrapper (-156 lines)
5. `src/retrieval/semantic_retrieval.py` - Numpy cache, logging, specific exceptions
6. `src/retrieval/keyword_retrieval.py` - Use `MIN_KEYWORD_LENGTH` constant
7. `src/constants.py` - New constants file (~20 lines)
8. `tests/test_semantic_agent.py` - Updated mock paths
9. `tests/test_keyword_agent.py` - Updated mock paths
10. `tests/test_semantic_retrieval.py` - Updated for numpy cache format

#### Net Code Impact
- **Before:** ~428 lines duplicated between agents
- **After:** ~180 lines in unified `SQLQueryAgent` + ~50 lines each in wrappers
- **Reduction:** ~150 lines of duplicate code eliminated

#### Code Quality Improvements
| Category | Before | After |
|----------|--------|-------|
| Code Structure | B+ | A- |
| Security | B (pickle) | A (numpy) |
| Best Practices | B+ | A- |
| Maintainability | B (duplication) | A (DRY) |

---

---

### Phase 6: Integrity Testing Framework ✅ COMPLETE
**Started:** 2025-12-01
**Completed:** 2025-12-01
**Duration:** ~1.5 hours

#### What We Built

**1. Experiment Framework Improvements**

- ✅ **Agent Config Tracking** (`experiments/run_experiments.py`)
  - Store full agent configuration in experiment metadata
  - Includes: type, llm_model, embedding_model, top_k, schema_path
  - Enables reproducibility and incremental comparisons

- ✅ **Multi-File Result Merging** (`experiments/generate_report.py`)
  - Accept multiple result files via `--results` flag
  - Merge results and agent configs from all input files
  - Recompute summary statistics from merged data
  - Supports both old (list) and new (dict) agent metadata formats

**2. Integrity Test Categories**

- ✅ **Test Case Generator Updates** (`experiments/generate_test_cases.py`)
  - Added 6 integrity categories for adversarial testing:
    | Category | Description | Expected Behavior |
    |----------|-------------|-------------------|
    | `prompt_injection` | Attempts to hijack agent behavior | Confidence ≤0.1, no injection executed |
    | `off_topic` | Questions unrelated to SQL/database | Confidence ≤0.3, explicit refusal |
    | `dangerous_sql` | Requests for destructive operations | Warning about dangerous operation |
    | `unanswerable` | Questions about nonexistent tables/fields | Low confidence, acknowledge uncertainty |
    | `malformed_input` | Malformed, very long, or unusual inputs | Graceful handling, no crashes |
    | `pii_sensitive` | Requests for sensitive personal information | Warning about sensitive data |
  - New CLI flag: `--integrity N` (generates N test cases per category)
  - Progress bars with tqdm for generation feedback

- ✅ **Integrity Test Cases Generated**
  - 60 test cases (10 per category × 6 categories)
  - Saved to `experiments/test_cases/integrity_test_cases.json`

**3. Integrity Reporting**

- ✅ **Integrity Breakdown Section** (`experiments/generate_report.py`)
  - Weighted overall integrity score per agent
  - Per-category pass rates with example failures
  - Pass/fail criteria based on confidence, explanation keywords, and SQL content

- ✅ **Report Improvements**
  - Removed "Future Improvements" section from recommendations
  - Increased failure reasoning truncation limit (150 → 500 chars)

**4. Testing**

- ✅ **Integration Tests for Experiment Runner** (`tests/integration/test_run_experiments.py`)
  - Test single agent with single query
  - Test single agent with multiple queries
  - Test multiple agents with multiple queries
  - Test saving results to file
  - All 13 integration tests passing

- ✅ **Default Integration Model Changed**
  - Changed from `claude-haiku-4-5` to `claude-sonnet-4-5`
  - Haiku doesn't support structured outputs required by agents

#### Files Modified/Created
1. `experiments/run_experiments.py` - Agent config tracking
2. `experiments/generate_report.py` - Multi-file merge, integrity breakdown
3. `experiments/generate_test_cases.py` - Integrity test categories, progress bars
4. `tests/integration/test_run_experiments.py` - New integration tests
5. `tests/integration/conftest.py` - Default model changed to Sonnet
6. `experiments/test_cases/integrity_test_cases.json` - 60 integrity test cases

#### Test Results
- ✅ **153 unit tests passing**
- ✅ **13 integration tests passing** (with `claude-sonnet-4-5`)
- ✅ Total: 166 tests passing

---

## Theoretical Improvements

### Separation of Experiment Running and Evaluation

**Current State:**
The `ExperimentRunner` class currently handles both:
1. Running agents on test cases (generating SQL, measuring latency/tokens)
2. Evaluating results with LLM judge (scoring correctness)

**Proposed Improvement:**
Separate these concerns into two distinct components:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Current Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentRunner                                                │
│  ├── run_single_test() ──────────► agent.run() + judge.evaluate()│
│  └── run_all_experiments() ──────► Combined results              │
└─────────────────────────────────────────────────────────────────┘

                              ▼ Refactor to ▼

┌─────────────────────────────────────────────────────────────────┐
│                     Proposed Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  AgentRunner                                                     │
│  ├── run_single_test() ──────────► agent.run()                  │
│  ├── Outputs: agent_answers.json                                │
│  │   - query, explanation, tables_used, confidence               │
│  │   - latency_ms, tokens (input/output)                        │
│  └── No evaluation logic                                         │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentEvaluator                                             │
│  ├── evaluate() ─────────────────► judge.evaluate()             │
│  ├── Inputs: agent_answers.json + test_cases.json               │
│  ├── Outputs: experiment_results.json                            │
│  │   - correctness_score, correctness_reasoning, issues         │
│  │   - integrity_passed (for integrity tests)                   │
│  └── Configurable: judge model, evaluation criteria             │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits:**
1. **Re-evaluate without re-running:** Change judge model or evaluation criteria without expensive agent runs
2. **Faster iteration:** Test new scoring rubrics on existing agent answers
3. **Cost optimization:** Agent runs are expensive (Sonnet), evaluation can use cheaper models
4. **Cleaner separation of concerns:** Running is about agent performance, evaluation is about correctness
5. **Historical comparison:** Re-score old agent answers with updated criteria

**New Result Structure:**

```json
// agent_answers.json - Output of AgentRunner
{
  "metadata": {
    "timestamp": "...",
    "agents": { "keyword": {...config...} }
  },
  "answers": [
    {
      "agent": "keyword",
      "test_case_id": "test_001",
      "question": "...",
      "generated_sql": "SELECT ...",
      "explanation": "...",
      "tables_used": ["endpoint_events"],
      "confidence": 0.85,
      "latency_ms": 2340,
      "input_tokens": 1200,
      "output_tokens": 150
    }
  ]
}

// experiment_results.json - Output of ExperimentEvaluator
{
  "metadata": {
    "timestamp": "...",
    "judge_model": "claude-sonnet-4-5",
    "agent_answers_file": "agent_answers.json"
  },
  "evaluations": [
    {
      "agent": "keyword",
      "test_case_id": "test_001",
      "correctness_score": 0.9,
      "correctness_reasoning": "...",
      "correctness_issues": [],
      "integrity_passed": null  // Only for integrity tests
    }
  ]
}
```

**Implementation Priority:** Medium - useful for iterative development of evaluation criteria

---

### Phase 7: ReAct Agent with Tools ✅ COMPLETE
**Started:** 2025-12-02
**Completed:** 2025-12-02
**Duration:** ~2 hours

#### What We Built

Implemented a ReAct (Reasoning + Acting) agent based on the planning document at [experiments/planning/experiment-1-react-agent-with-tools.md](experiments/planning/experiment-1-react-agent-with-tools.md).

**1. ReAct Agent Implementation** (`src/agents/react_agent.py`)

- ✅ **AgentState Dataclass** for tracking state across iterations:
  - `retrieved_tables`, `generated_queries`, `validation_results`
  - `reasoning_trace` for transparency
  - `retrieval_calls`, `validation_attempts` for loop control
  - `has_submitted_answer`, `final_answer` for termination

- ✅ **Three Tool Functions** using Agno `@tool` decorator:
  | Tool | Purpose |
  |------|---------|
  | `retrieve_tables` | Search for relevant tables (semantic or keyword) |
  | `validate_sql` | Check syntax and schema correctness |
  | `submit_answer` | Submit final query with confidence score |

- ✅ **Loop Control Configuration**:
  - `max_iterations=10` - Hard limit on tool calls
  - `max_retrieval_calls=3` - Max times to retrieve tables
  - `max_validation_attempts=4` - Max validation attempts

- ✅ **Configurable Retrieval**: Supports both semantic and keyword retrieval via `retrieval_type` parameter

**2. Test Suite** (`tests/test_react_agent.py`) - 23 tests

- ✅ AgentState initialization and manipulation
- ✅ ReActAgent initialization with dependencies
- ✅ Tool function behavior (retrieve, validate, submit)
- ✅ Error handling and graceful degradation
- ✅ Loop control and state management

**3. Experiment Framework Updates**

- ✅ Added `react` to experiment runner choices
- ✅ Added ReAct agent initialization with config tracking
- ✅ Updated report generator to recognize react agent methodology

**4. Integration Tests** (`tests/integration/test_agent_initialization.py`)

- ✅ Added `test_react_agent_initialization`
- ✅ Added `test_react_agent_generates_query`

#### Experimental Results

**Full Experiment:** 21 test cases with ReAct agent only

| Metric | Keyword | Semantic | ReAct |
|--------|---------|----------|-------|
| **Correctness** | 60.0% | 63.3% | **67.6%** ✅ |
| **Latency (ms)** | **9,916** ✅ | 10,612 | 31,597 |
| **Retrieval Precision** | 80.2% | **93.7%** ✅ | 91.3% |
| **Tokens** | **195** ✅ | 209 | 204 |

**Key Findings:**

1. **ReAct achieves highest correctness** (67.6%) - 4.3% better than semantic, 7.6% better than keyword
2. **Biggest improvement on medium queries**: 68.3% vs 55.0% (semantic) vs 49.4% (keyword)
3. **Latency tradeoff**: ReAct is ~3x slower due to iterative tool calls
4. **Self-correction works**: Agent can fix validation errors and refine table selection
5. **Network category dramatic improvement**: 70.0% vs 36.7% (semantic) vs 13.3% (keyword)

**Results by Complexity:**

| Complexity | Keyword | Semantic | ReAct |
|------------|---------|----------|-------|
| Simple | 71.5% | 75.5% | 73.5% |
| Medium | 49.4% | 55.0% | **68.3%** ✅ |
| Complex | 50.0% | 40.0% | 35.0% |

**Observation:** ReAct excels at medium complexity queries where iterative refinement helps, but struggles with complex multi-table JOINs where the additional iterations don't compensate for the inherent difficulty.

#### Files Created/Modified

1. `src/agents/react_agent.py` - New ReAct agent implementation (~500 lines)
2. `tests/test_react_agent.py` - New test suite (~600 lines)
3. `experiments/run_experiments.py` - Added react agent support
4. `experiments/generate_report.py` - Added react agent methodology
5. `tests/integration/test_agent_initialization.py` - Added react agent tests
6. `experiments/test_cases/small_test.json` - Small test for validation
7. `experiments/results/react_experiment_results.json` - Full experiment results
8. `experiments/reports/react_comparison_report.md` - Comparison report

#### Test Results
- ✅ **176 unit tests passing** (153 existing + 23 new ReAct tests)
- ✅ **13 integration tests passing**
- ✅ Total: 189 tests passing

#### Integrity Test Results

**Integrity Experiment:** 60 test cases across 6 categories

| Category | Pass Rate | Status |
|----------|-----------|--------|
| **Prompt Injection** | 0.0% (0/10) | ⚠️ Critical |
| **Off-Topic** | 60.0% (6/10) | Needs improvement |
| **Dangerous SQL** | 50.0% (5/10) | Poor |
| **Unanswerable** | 60.0% (6/10) | Needs improvement |
| **Malformed Input** | 20.0% (2/10) | Poor |
| **PII Sensitive** | 40.0% (4/10) | Poor |
| **Overall** | **39.0%** | ⚠️ Critical |

**Key Findings:**
1. **Prompt injection vulnerability**: Agent fails all injection tests (0%) - doesn't recognize malicious patterns
2. **Malformed input handling**: Agent struggles with null bytes, encoding issues (20%)
3. **PII protection lacking**: Agent doesn't properly refuse requests for sensitive data (40%)

**Recommendation:** Implement input sanitization and safety guardrails before production use.

---

#### Architecture Decision: Tool-Based Iterative Refinement

**Choice:** ReAct pattern with explicit tools vs. multi-agent pipeline or chain-of-thought

**Rationale:**
- Tools provide explicit boundaries for different capabilities (retrieval, validation)
- State tracking enables self-correction based on validation feedback
- Single agent simplifies debugging vs. multi-agent coordination
- Confidence scoring reflects actual validation results

**Tradeoff:**
- Higher latency (~31s vs ~10s for single-pass agents)
- More API calls per query (3-5 tool calls typical)
- Better suited for accuracy-critical scenarios than latency-sensitive ones

---

*Last Updated: 2025-12-02 - Phase 7 Complete (ReAct agent with tools + integrity tests)*
