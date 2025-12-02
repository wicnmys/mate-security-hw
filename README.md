# SQL Query Agent for Security Events

**Mate Security Take-Home Assignment**

An intelligent SQL query generation system built with the Agno framework that converts natural language questions into PostgreSQL queries for security event databases.

## Overview

This project implements a production-ready SQL agent that:
- Converts natural language security questions into SQL queries
- Uses semantic retrieval to find relevant tables from a 20-table security schema
- Validates generated SQL for correctness and safety
- Provides explanations and confidence scores
- Includes comprehensive testing (288 tests: unit + integration)
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

# Run a query (default: semantic agent)
python main.py "Show me all high severity endpoint events from the last 24 hours"

# Use a specific agent
python main.py "Which users had the most failed login attempts?" --agent react

# See detailed explanation
python main.py "Which users had the most failed login attempts?" --explain

# Get JSON output
python main.py "Find suspicious file access events" --json
```

## Architecture

### Agent Types

This project implements multiple agent architectures for SQL generation. Each agent has a detailed documentation card:

| Agent | Description | Documentation |
|-------|-------------|---------------|
| **KeywordAgent** | Lightweight agent using keyword-based table retrieval | [Agent Card](src/agents/agent_cards/keyword_agent.md) |
| **SemanticAgent** | Uses embedding-based semantic retrieval for better NL understanding | [Agent Card](src/agents/agent_cards/semantic_agent.md) |
| **ReActAgent** | Iterative tool-use agent with self-correction capabilities | [Agent Card](src/agents/agent_cards/react_agent.md) |
| **ReActAgentV2** | Enhanced ReAct with dual validation (structural + LLM judge) | [Agent Card](src/agents/agent_cards/react_agent_v2.md) |

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

**2. Multiple Agent Architectures**
- **Simple agents**: KeywordAgent and SemanticAgent for single-pass generation
- **ReAct agents**: Iterative tool-use with self-correction (ReActAgent, ReActAgentV2)
- All agents use structured output (Pydantic `SQLQueryResponse`)
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

*All examples below show actual outputs from the agent (run with `--json` flag).*

### Example 1: High Severity Events

```bash
python main.py "Show me all high-severity security events from the last 24 hours"
```

**Output:**
```json
{
  "query": "SELECT alert_id, timestamp, alert_name, alert_type, severity, priority,
            status, source_system, affected_assets, affected_users, description
            FROM security_alerts
            WHERE severity IN ('high', 'critical')
              AND timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC",
  "tables_used": ["security_alerts"],
  "confidence": 0.95,
  "reasoning_steps": [
    "Identified 'high-severity' as requiring severity filter for 'high' and 'critical' levels",
    "Mapped 'security events' to the security_alerts table",
    "Parsed 'last 24 hours' as a time filter using NOW() - INTERVAL '24 hours'",
    "Applied ORDER BY timestamp DESC to show most recent events first"
  ]
}
```

### Example 2: Failed Login Analysis

```bash
python main.py "Which users had the most failed login attempts this month?"
```

**Output:**
```json
{
  "query": "SELECT user_name, COUNT(*) as failed_attempts
            FROM authentication_events
            WHERE status = 'failure'
              AND timestamp >= DATE_TRUNC('month', CURRENT_TIMESTAMP)
            GROUP BY user_name
            ORDER BY failed_attempts DESC",
  "tables_used": ["authentication_events"],
  "confidence": 0.95,
  "reasoning_steps": [
    "Identified that failed login data is in the authentication_events table",
    "Used status='failure' to filter for failed authentication attempts",
    "Applied DATE_TRUNC('month', CURRENT_TIMESTAMP) to filter for current month"
  ]
}
```

### Example 3: File Access Investigation

```bash
python main.py "Find all suspicious file access events related to sensitive documents"
```

**Output:**
```json
{
  "query": "SELECT fe.event_id, fe.timestamp, fe.hostname, fe.event_type,
            fe.file_path, fe.file_name, fe.user_name, fe.process_name
            FROM file_events fe
            WHERE (LOWER(fe.file_path) LIKE '%confidential%'
                   OR LOWER(fe.file_path) LIKE '%sensitive%' ...)
              AND (fe.event_type IN ('copy', 'rename', 'delete')
                   OR fe.access_mask LIKE '%WRITE%')
            ORDER BY fe.timestamp DESC",
  "tables_used": ["file_events"],
  "confidence": 0.75,
  "reasoning_steps": [
    "Identified file_events as the primary table for file system activity",
    "Determined 'sensitive documents' by file path keywords and extensions",
    "Defined 'suspicious' activity as copy, rename, delete, or write operations",
    "Confidence of 0.75 due to ambiguity in what constitutes 'sensitive' and 'suspicious'"
  ]
}
```

### Example 4: Event Type Analysis

```bash
python main.py "What are the top 10 most common security event types?"
```

**Output:**
```json
{
  "query": "SELECT event_type, COUNT(*) as event_count
            FROM endpoint_events
            GROUP BY event_type
            ORDER BY event_count DESC
            LIMIT 10",
  "tables_used": ["endpoint_events"],
  "confidence": 0.95,
  "reasoning_steps": [
    "The endpoint_events table contains core security events with event_type field",
    "Used GROUP BY and COUNT(*) for aggregation",
    "Applied ORDER BY count DESC and LIMIT 10 for top results"
  ]
}
```

### Example 5: Multi-Alert IP Detection

```bash
python main.py "Show me events where the same IP address triggered multiple alerts"
```

**Output:**
```json
{
  "query": "SELECT ip_address, COUNT(*) as alert_count,
            MIN(timestamp) as first_alert, MAX(timestamp) as last_alert,
            ARRAY_AGG(DISTINCT alert_name) as alert_names
            FROM security_alerts
            WHERE ip_address IS NOT NULL
            GROUP BY ip_address
            HAVING COUNT(*) > 1
            ORDER BY alert_count DESC",
  "tables_used": ["security_alerts"],
  "confidence": 0.6,
  "reasoning_steps": [
    "Identified need to find duplicate IP addresses across security alerts",
    "Used GROUP BY with HAVING COUNT(*) > 1 to filter for IPs with multiple alerts",
    "Included temporal information (first/last alert) for context",
    "Confidence reduced to 0.6 due to uncertainty about IP address field location"
  ]
}
```

## Edge Case Handling

The agent handles various edge cases gracefully:

### Empty Input

```bash
python main.py ""
```

**Output:**
```json
{
  "query": null,
  "explanation": "Error processing question: Question cannot be empty",
  "tables_used": [],
  "confidence": 0.0,
  "error": "Question cannot be empty"
}
```

### Ambiguous Questions

```bash
python main.py "Show me stuff"
```

**Output:**
```json
{
  "query": "SELECT asset_id, hostname, asset_type, os_type, location, criticality
            FROM asset_inventory ORDER BY last_updated DESC LIMIT 100",
  "confidence": 0.3,
  "reasoning_steps": [
    "Question is extremely ambiguous with no specific context",
    "Chose asset_inventory as foundational security data",
    "Confidence is very low - user needs to provide more specific requirements"
  ]
}
```

### Dangerous Operations Detection

```bash
python main.py "DROP TABLE users; DELETE FROM security_events"
```

**Output:**
```json
{
  "query": "SELECT 'SECURITY ALERT: SQL injection attempt detected' AS warning",
  "explanation": "This input contains SQL injection attack patterns. Rather than
                  executing destructive commands, returning a safety message.",
  "confidence": 0.5,
  "reasoning_steps": [
    "Detected SQL injection keywords: DROP TABLE, DELETE FROM",
    "Identified destructive intent rather than data query intent",
    "Returning warning message instead of generating destructive SQL",
    "Validation errors: Invalid SQL syntax"
  ]
}
```

### Nonexistent Tables/Fields

```bash
python main.py "Show me foobar_column from xyzzy_table"
```

**Output:**
```json
{
  "query": "SELECT foobar_column, baz_field FROM xyzzy_table",
  "confidence": 0.1,
  "reasoning_steps": [
    "No table named 'xyzzy_table' exists in the database",
    "No columns named 'foobar_column' exist in any available tables",
    "Validation warnings: Unknown fields: foobar_column, baz_field",
    "Validation errors: Unknown tables: xyzzy_table"
  ]
}
```

### Unrelated Queries (No Matching Tables)

```bash
python main.py "Show me all quantum blockchain NFT transactions"
```

**Output:**
```json
{
  "query": "SELECT * FROM network_traffic
            WHERE application LIKE '%blockchain%' OR application LIKE '%NFT%'
            LIMIT 100",
  "confidence": 0.25,
  "reasoning_steps": [
    "Identified that 'quantum blockchain NFT' combines unrelated buzzwords",
    "Database schema has no tables for blockchain/crypto monitoring",
    "Made best-effort attempt using network_traffic table",
    "Very low confidence - query unlikely to return meaningful results"
  ]
}
```

## Testing

### Run All Tests (Fast)

```bash
# Unit tests only (no API key required)
pytest tests/ -v -m "not integration"
```

### Run Integration Tests (Requires API Key)

```bash
# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Run integration tests (validates real API contracts)
pytest tests/integration/ -v -m integration
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

# Test ReAct agents
pytest tests/test_react_agent.py tests/test_react_agent_v2.py -v

# Test experiment framework
pytest tests/test_judges.py tests/test_experiment_configs.py -v
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
│   ├── constants.py             # Configuration constants
│   ├── agents/
│   │   ├── base.py              # Abstract base agent
│   │   ├── sql_agent.py         # Core SQL agent with pluggable retrieval
│   │   ├── keyword_agent.py     # Keyword-based retrieval agent
│   │   ├── semantic_agent.py    # Semantic retrieval agent
│   │   ├── react_agent.py       # ReAct agent with tool-use loop
│   │   ├── react_agent_v2.py    # Enhanced ReAct with LLM judge
│   │   └── agent_cards/         # Agent documentation
│   │       ├── keyword_agent.md
│   │       ├── semantic_agent.md
│   │       ├── react_agent.md
│   │       └── react_agent_v2.md
│   ├── retrieval/
│   │   ├── semantic_retrieval.py  # Embedding-based retrieval
│   │   └── keyword_retrieval.py   # Keyword-based retrieval
│   └── utils/
│       ├── schema_loader.py     # Load and format schemas
│       └── validator.py         # SQL validation
├── tests/
│   ├── test_*.py               # Unit tests
│   └── integration/            # Integration tests
│       ├── test_agent_initialization.py
│       ├── test_llm_judge.py
│       ├── test_caching_parameters.py
│       └── test_run_experiments.py
├── experiments/
│   ├── configs/                 # Experiment configurations
│   │   └── experiment_config.py
│   ├── judges/                  # LLM judge implementations
│   │   ├── base.py
│   │   ├── correctness_judge.py
│   │   ├── categorical_judge.py
│   │   └── integrity_judge.py
│   ├── utils/
│   │   ├── metrics.py          # Correctness, precision, latency
│   │   └── llm_judge.py        # LLM-as-judge evaluation
│   ├── generate_test_cases.py
│   ├── run_experiments.py
│   ├── rejudge.py              # Re-evaluate existing results
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
- **288 total tests**: Unit tests (fast, mocked) + integration tests (real API)
- **Test coverage**: Schema loading, retrieval, validation, all agent types, experiments
- **CI-ready**: Separate test markers for unit vs integration

## Configuration Options

### Command-Line Arguments

```bash
python main.py [question] [options]

Options:
  --agent AGENT     Agent type: semantic, keyword, react, react-v2 (default: semantic)
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
- Unit tests: ~4s for all unit tests
- Integration tests: ~30s for integration tests

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
4. Create an agent card in `src/agents/agent_cards/`
5. Register the agent in `src/agents/registry.py`:
   ```python
   from src.agents.your_agent import YourAgent

   AGENT_REGISTRY = {
       ...
       "your-agent": YourAgent,
   }

   AGENT_DEFAULTS = {
       ...
       "your-agent": {
           "retrieval_type": "semantic",  # or "keyword"
       },
   }
   ```
   This single registration enables the agent in both CLI (`main.py`) and experiments.

**Add a new retrieval strategy:**
1. Create new class in `src/retrieval/`
2. Implement `get_top_k(question, k)` method
3. Add tests in `tests/test_your_retrieval.py`
4. Wire up to agent initialization

## Known Limitations

1. **No database execution**: Agent generates SQL but doesn't execute it
2. **Haiku limitations**: Doesn't support structured outputs (skips some integration tests)

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
