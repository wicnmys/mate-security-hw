# SQL Query Agent for Security Events

**Mate Security Take-Home Assignment**

An intelligent SQL query generation system built with the Agno framework that converts natural language questions into PostgreSQL queries for security event databases.

## Overview

This project implements a production-ready SQL agent that:
- Converts natural language security questions into SQL queries
- Uses semantic retrieval to find relevant tables from a 20-table security schema
- Validates generated SQL for correctness and safety
- Provides explanations and confidence scores
- Includes comprehensive testing (168 tests: 159 unit + 9 integration)
- Features experimental comparison of agent architectures

**Key Features:**
- **Semantic Retrieval**: Uses sentence-transformers for embedding-based table selection
- **Structured Output**: Pydantic models ensure consistent, parseable responses
- **SQL Validation**: Detects syntax errors, dangerous operations, and schema violations
- **Prompt Caching**: Anthropic caching reduces costs by ~90% on repeated queries
- **Comprehensive Testing**: Unit tests (fast, mocked) + integration tests (real API)
- **Experimental Framework**: LLM-as-judge evaluation with quantitative metrics

## Quick Start

### Prerequisites
- Python 3.10+
- Anthropic API key

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd mate-security-hw
./setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Configuration

Create a `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
```

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run a query
python main.py "Show me all high severity endpoint events from the last 24 hours"

# See detailed explanation
python main.py "Which users had the most failed login attempts?" --explain

# Get JSON output
python main.py "Find suspicious file access events" --json
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input (Natural Language)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Retrieval                         │
│  • Embed question with sentence-transformers                │
│  • Compare to pre-cached table embeddings                   │
│  • Return top-K most relevant tables                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agno Agent                              │
│  • Model: Claude Sonnet 4.5 (configurable)                  │
│  • Input: Question + relevant table schemas                 │
│  • Output: Structured SQLQueryResponse (Pydantic)           │
│  • Features: Prompt caching, reasoning steps                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    SQL Validator                             │
│  • Check syntax (SELECT/FROM structure)                     │
│  • Verify table/field existence                             │
│  • Detect dangerous operations (DROP, DELETE)               │
│  • Return errors/warnings                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Formatted Response                           │
│  • SQL query                                                 │
│  • Natural language explanation                             │
│  • Confidence score                                          │
│  • Validation results                                        │
│  • Tables used                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Semantic Retrieval (Variant B)**
- Uses `multi-qa-mpnet-base-dot-v1` for embeddings
- Local model (no API calls) with caching
- Better accuracy than keyword matching for security queries
- Precomputed embeddings saved in `embeddings_cache/`

**2. Single Agent Architecture**
- One agent with structured output (Pydantic `SQLQueryResponse`)
- Simpler than multi-agent pipeline, easier to debug
- Uses Agno framework for reliability

**3. Prompt Caching Strategy**
- Cache system instructions and table schemas
- 90% cost reduction on cached tokens ($0.30 → $0.03 per query)
- 1-hour TTL suitable for interactive sessions

**4. Non-Blocking Validation**
- Validate SQL but include warnings, not errors
- Allows minor syntax variations
- User sees query + validation feedback

## Usage Examples

### Example 1: High Severity Events

```bash
python main.py "Show me all high-severity security events from the last 24 hours"
```

**Output:**
```sql
SELECT *
FROM endpoint_events
WHERE severity IN ('high', 'critical')
  AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
```

### Example 2: Failed Login Analysis

```bash
python main.py "Which users had the most failed login attempts this month?"
```

**Output:**
```sql
SELECT user_id, username, COUNT(*) as failed_attempts
FROM authentication_events
WHERE event_type = 'login_failed'
  AND timestamp >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY user_id, username
ORDER BY failed_attempts DESC
LIMIT 10;
```

### Example 3: File Access Investigation

```bash
python main.py "Find all suspicious file access events related to sensitive documents" --explain
```

**Output includes:**
- Generated SQL query
- Explanation of query logic
- Retrieved tables and relevance scores
- Confidence score
- Validation results

## Testing

### Run All Tests (Fast)

```bash
# Unit tests only (no API key required)
pytest tests/ -v -m "not integration"

# Output: 159 passed in ~4s
```

### Run Integration Tests (Requires API Key)

```bash
# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Run integration tests (validates real API contracts)
pytest tests/integration/ -v -m integration

# Output: 9 tests (6 passed, 3 skipped on Haiku)
```

### Run Specific Tests

```bash
# Test schema loader
pytest tests/test_schema_loader.py -v

# Test semantic retrieval
pytest tests/test_semantic_retrieval.py -v

# Test SQL validator
pytest tests/test_validator.py -v

# Test agents
pytest tests/test_semantic_agent.py tests/test_keyword_agent.py -v
```

## Experiments

The project includes an experimental framework for comparing different agent architectures and approaches.

### Generate Test Cases

```bash
python experiments/generate_test_cases.py \
  --schema-path schemas/dataset.json \
  --output experiments/test_cases/test_cases.json \
  --simple 10 --medium 10 --complex 5
```

### Run Experiments

```bash
python experiments/run_experiments.py \
  --test-cases experiments/test_cases/generated_test_cases.json \
  --schema-path schemas/dataset.json \
  --agents keyword semantic \
  --output experiments/results/results.json
```

### Generate Report

```bash
python experiments/generate_report.py \
  --results experiments/results/results.json \
  --output experiments/comparison.md
```

See [experiments/README.md](experiments/README.md) for detailed documentation.

## Project Structure

```
mate-security-hw/
├── src/
│   ├── agents/
│   │   ├── base.py              # Abstract base agent
│   │   ├── semantic_agent.py    # Main agent (semantic retrieval)
│   │   └── keyword_agent.py     # Baseline agent (keyword retrieval)
│   ├── retrieval/
│   │   ├── semantic_retrieval.py  # Embedding-based retrieval
│   │   └── keyword_retrieval.py   # Keyword-based retrieval
│   └── utils/
│       ├── schema_loader.py     # Load and format schemas
│       └── validator.py         # SQL validation
├── tests/
│   ├── test_*.py               # Unit tests (159 tests)
│   └── integration/            # Integration tests (9 tests)
│       ├── test_agent_initialization.py
│       ├── test_llm_judge.py
│       └── test_caching_parameters.py
├── experiments/
│   ├── utils/
│   │   ├── metrics.py          # Correctness, precision, latency
│   │   └── llm_judge.py        # LLM-as-judge evaluation
│   ├── generate_test_cases.py
│   ├── run_experiments.py
│   ├── generate_report.py
│   └── README.md
├── schemas/
│   └── dataset.json            # 20-table security schema
├── main.py                     # CLI entrypoint
├── setup.sh                    # One-command setup
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev + test dependencies
└── DEVELOPMENT.md              # Detailed development log
```

## Key Features

### 1. Semantic Retrieval
- Uses `sentence-transformers` for table embeddings
- Cosine similarity for relevance scoring
- Precomputed embeddings cached to disk
- No API calls needed (local model)

### 2. SQL Validation
- Syntax checking (SELECT/FROM structure)
- Balanced delimiter validation (quotes, parentheses)
- Table and field existence verification
- Dangerous operation detection (DROP, DELETE, etc.)
- Strict and non-strict modes

### 3. Prompt Caching
- Caches system instructions and table schemas
- 90% cost reduction on cached tokens
- Automatic cache management (1-hour TTL)
- Enabled via `cache_system_prompt=True`

### 4. Structured Output
- Pydantic models for type safety
- Consistent JSON schema
- Easy to parse and validate
- Includes reasoning steps and confidence

### 5. Comprehensive Testing
- **159 unit tests**: Fast, mocked, no API keys
- **9 integration tests**: Real API validation
- **Test coverage**: Schema loading, retrieval, validation, agents
- **CI-ready**: Separate test markers for unit vs integration

## Configuration Options

### Command-Line Arguments

```bash
python main.py [question] [options]

Options:
  --explain         Show retrieval process and reasoning
  --json            Output results as JSON
  --model MODEL     LLM model (default: claude-sonnet-4-5)
  --top-k K         Number of tables to retrieve (default: 5)
```

### Environment Variables

```bash
ANTHROPIC_API_KEY    # Required: Your Anthropic API key
INTEGRATION_TEST_MODEL  # Optional: Model for integration tests (default: claude-haiku-4-5)
```

## Performance & Costs

### Query Latency
- First query: ~3-5s (schema loading + embedding computation)
- Subsequent queries: ~1-2s (cached embeddings + prompt caching)

### API Costs (with caching)
- First query: ~$0.30 (full prompt)
- Cached queries: ~$0.03 (90% reduction)
- Integration tests: ~$0.10 per run (Haiku)

### Test Execution
- Unit tests: ~4s for 159 tests
- Integration tests: ~30s for 9 tests

## Development

### Development Log
See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Phase-by-phase implementation details
- Architecture decisions and rationale
- Lessons learned
- Time tracking
- Git commit history

### Adding New Features

**Add a new agent:**
1. Subclass `BaseAgent` in `src/agents/`
2. Implement `__init__` and `run` methods
3. Add tests in `tests/test_your_agent.py`
4. Update experiments to include new agent

**Add a new retrieval strategy:**
1. Create new class in `src/retrieval/`
2. Implement `get_top_k(question, k)` method
3. Add tests in `tests/test_your_retrieval.py`
4. Wire up to agent initialization

## Known Limitations

1. **No database execution**: Agent generates SQL but doesn't execute it
2. **Validation is heuristic**: Regex-based, not a full SQL parser
3. **Haiku limitations**: Doesn't support structured outputs (skips some integration tests)
4. **Single retrieval pass**: No iterative refinement based on LLM feedback

## Future Improvements

- [ ] Add execution layer with actual database
- [ ] Implement multi-turn conversations for query refinement
- [ ] Add support for more complex queries (CTEs, window functions)
- [ ] Create web UI for easier interaction
- [ ] Add caching layer for repeated questions
- [ ] Support for multiple database schemas simultaneously

## Assignment Requirements

### Deliverables Checklist

- [x] Working SQL agent with semantic retrieval
- [x] Comprehensive testing (168 tests total)
- [x] Experimental comparison framework
- [x] Documentation (README, DEVELOPMENT.md, inline docs)
- [x] Clean code with proper structure
- [x] Git history with meaningful commits
- [ ] 5-minute video demo (in progress)
- [x] Bonus: Integration tests, prompt caching, LLM-as-judge

## Contact

For questions about this implementation, please refer to the code documentation and development log.

---

**Built with:** Agno Framework, Claude Sonnet 4.5, sentence-transformers, pytest
