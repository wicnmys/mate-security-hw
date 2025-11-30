# Mate Security SQL Query Agent - Implementation Context

## Project Overview
Take-home assignment for Mate Security AI Lead position: Build an intelligent SQL query generation agent using the Agno framework that converts natural language questions to SQL queries for a large security events database (1,000 tables, up to 300 fields per table).

**Time Budget:** 4-6 hours core + 2 hours experiments + 1.5 hours testing = 7.5-9.5 hours total  
**Framework:** [Agno](https://github.com/agno-agi/agno)  
**Strategic Goal:** Demonstrate evaluation-driven architecture decisions (addresses their pain points: slow/inaccurate agents)

---

## Environment Setup

### Python Version
**Required:** Python 3.10 or higher (for Pydantic 2.x compatibility)

### Package Manager & Virtual Environment
**Use:** `pip` + `venv` (standard library, universal, no extra tools needed)

**Setup Commands:**
```bash
# Create project directory
mkdir sql-query-agent
cd sql-query-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (testing)
pip install -r requirements-dev.txt
```

### Requirements Files Structure
**requirements.txt** (production dependencies):
```txt
agno>=0.1.0
openai>=1.0.0
pydantic>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0  # For .env file loading
```

**requirements-dev.txt** (development/testing dependencies):
```txt
-r requirements.txt  # Include production deps
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0        # Code formatting
ruff>=0.1.0          # Linting
```

### Environment Variables
**Required:** OpenAI API key for embeddings and LLM calls

```bash
# Create .env file (add to .gitignore!)
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Or export directly
export OPENAI_API_KEY="your_api_key_here"
```

### .gitignore
```txt
# Virtual environment
.venv/
venv/
env/

# Environment variables
.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# Embeddings cache (optional)
*.pkl
embeddings_cache/
```

---

## Technical Decisions

### Agno Framework Usage
- **Main Pattern:** Single agent with structured output (`response_model` for JSON schema)
- **Model:** GPT-4 or Claude for SQL generation
- **Chain-of-Thought:** Use detailed instructions to guide multi-step reasoning internally
- **No Conversations:** Single-turn input/output (not multi-turn dialogue)

### Architecture Approach
**Main Submission:** Variant B - Semantic Search + Single Agent
- Two-stage retrieval: embeddings → top-K tables → LLM generation
- Minimize LLM calls (addresses "slow agents" pain point)
- Structured JSON output with confidence scoring

**Experimental Comparison:** 3 Variants
1. **Baseline:** Keyword matching + single agent (simple, fast, less accurate)
2. **Semantic:** Embedding-based retrieval + single agent (balanced - MAIN)
3. **Multi-Agent:** Separate agents for schema selection → SQL generation → validation (accurate, slower)

### Testing & Metrics Strategy
**Test-Driven Development:**
- Generate synthetic test cases using LLM (20-25 cases covering simple/medium/complex)
- Each test case: question + expected SQL + difficulty + required tables

**Evaluation Metrics:**
1. **Correctness:** % semantically correct queries (LLM-as-judge)
2. **Latency:** End-to-end time per query
3. **Cost:** Total tokens used per query
4. **Retrieval Precision:** % retrieved tables actually used in query

### Memory/Storage Approach
- **Schema Storage:** In-memory dict/JSON (no database needed for demo)
- **Embeddings:** Pre-compute table embeddings, store in-memory or pickle file
- **Caching:** Optional - cache schema lookups for repeated tables

### Testing Strategy
**Critical Priority: Test Components in Isolation**
1. **Unit Tests First** - Test retrieval, validation, schema loading independently
2. **Integration Tests Second** - Test full agent pipeline end-to-end
3. **Why This Matters:** 
   - If agent fails, you can isolate whether it's retrieval (wrong tables selected) or generation (bad SQL from correct tables)
   - Fast iteration: retrieval tests run in milliseconds (no LLM calls)
   - Enables measuring "Retrieval Precision" metric with confidence

**Test-Driven Development Flow:**
```
1. Write retrieval tests → Implement retrieval → Verify tests pass
2. Write validation tests → Implement validation → Verify tests pass
3. Write agent tests → Implement agent → Verify tests pass
4. Run experimental evaluator on complete system
```

This approach catches bugs early and makes debugging much faster.

**Why Testing Matters for Mate Security:**
- **Slow Agents:** Unit tests help identify bottlenecks (is retrieval slow or LLM generation?)
- **Inaccurate Agents:** Can isolate whether errors come from retrieval (wrong tables) or generation (bad SQL)
- **Production Readiness:** Demonstrates you build reliable systems, not just working demos
- **In Video Demo:** "I tested retrieval independently, which revealed semantic search had 78% precision vs. keyword's 45% - this gave me confidence in the architecture choice"

---

## Repository Structure

```
sql-query-agent/
├── .venv/                       # Virtual environment (not in git)
├── .env                         # API keys (not in git)
├── .gitignore
├── src/
│   ├── agents/
│   │   ├── base.py              # Abstract base class
│   │   ├── baseline_agent.py    # Variant A: keyword matching
│   │   ├── semantic_agent.py    # Variant B: semantic search (MAIN)
│   │   └── multiagent.py        # Variant C: multi-agent pipeline
│   ├── retrieval/
│   │   ├── keyword_retrieval.py
│   │   └── semantic_retrieval.py
│   ├── generation/
│   │   └── sql_generator.py
│   └── utils/
│       ├── schema_loader.py
│       └── validator.py
├── experiments/                 # Evaluation framework (separate from core)
│   ├── test_generator.py       # Generate synthetic tests
│   ├── evaluator.py            # Metrics collection
│   ├── run_experiments.py      # Compare architectures
│   ├── test_cases/generated_tests.json
│   └── results/comparison.md
├── examples/
│   └── example_queries.py      # 5 required examples + bonus
├── tests/
│   ├── test_retrieval.py       # Retrieval unit tests (CRITICAL)
│   ├── test_agents.py          # Agent integration tests
│   ├── test_validation.py      # SQL validation tests
│   └── test_schema_loader.py   # Schema loading tests
├── schemas/                     # Schema JSON files (provided by Mate)
├── main.py                     # CLI: python main.py "question" --architecture=semantic
├── setup.sh                    # Optional: Quick setup script for Unix systems
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev/test dependencies
├── pyproject.toml              # Optional: project metadata
└── README.md
```

### Setup Verification
After setup, verify everything works:

```bash
# Verify Python version
python --version  # Should be 3.10+

# Verify virtual environment is active
which python  # Should point to .venv/bin/python

# Verify dependencies installed
pip list | grep agno
pip list | grep pytest

# Verify API key configured
python -c "import os; print('✓ API key set' if os.getenv('OPENAI_API_KEY') else '✗ No API key')"

# Run tests to verify installation
pytest tests/ -v --tb=short

# Run a quick example
python main.py "Show me high-severity events" --architecture=semantic
```

### Optional: Quick Setup Script
**setup.sh** (for Unix systems - nice for reviewers):
```bash
#!/bin/bash
set -e

echo "🚀 Setting up SQL Query Agent..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✓ Python version OK: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate and install
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements-dev.txt -q

echo "✓ Dependencies installed"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Create .env file with: OPENAI_API_KEY=your_key"
else
    echo "✓ API key configured"
fi

# Run tests
echo "Running tests..."
pytest tests/ -v --tb=short

echo "✅ Setup complete!"
echo ""
echo "To activate environment: source .venv/bin/activate"
echo "To run agent: python main.py \"your question here\""
```

---

## Implementation Outline

### Phase 1: Core Infrastructure (1.5 hrs)
1. **Schema Loader** (`src/utils/schema_loader.py`)
   - Parse JSON schema files
   - Build in-memory representation: `{table_name: {fields: [...], description: "..."}}`

2. **Semantic Retrieval** (`src/retrieval/semantic_retrieval.py`)
   - Pre-compute embeddings for all tables (OpenAI `text-embedding-3-small`)
   - Implement cosine similarity search: question → top-K tables
   - Return table schemas for LLM context

3. **Keyword Retrieval** (`src/retrieval/keyword_retrieval.py`)
   - Simple: exact match on table/field names in question
   - Fallback: return tables containing any keywords

### Phase 2: Agent Variants (2.5 hrs)

#### Variant B: Semantic Agent (MAIN - 1.5 hrs)
```python
from agno import Agent

class SemanticAgent:
    def __init__(self, schema_path):
        self.schemas = load_schemas(schema_path)
        self.retriever = SemanticRetrieval(self.schemas)
        
        self.agent = Agent(
            name="sql_generator",
            model="gpt-4",
            instructions="""Generate SQL for security event queries.
            
            Process:
            1. Analyze what data is requested
            2. Identify necessary joins between tables
            3. Determine filters and conditions
            4. Construct optimized SQL query
            
            Return structured JSON.""",
            response_model=SQLQueryResponse  # Pydantic model
        )
    
    def run(self, question: str):
        # Retrieve relevant schemas
        relevant_schemas = self.retriever.get_top_k(question, k=5)
        
        # Generate SQL
        response = self.agent.run(
            question=question,
            schemas=relevant_schemas
        )
        
        return response
```

#### Variant A: Baseline (1 hr)
- Use `KeywordRetrieval` instead of `SemanticRetrieval`
- Same SQL generation agent
- Simpler, faster, less accurate

#### Variant C: Multi-Agent (Optional - if time permits)
```python
# Agent 1: Schema selector
schema_agent = Agent(name="schema_selector", ...)

# Agent 2: SQL builder
sql_agent = Agent(name="sql_builder", ...)

# Agent 3: Validator
validator_agent = Agent(name="validator", ...)

# Orchestrator
orchestrator = Agent(
    name="orchestrator",
    instructions="Run pipeline: select schemas → build SQL → validate"
)
```

### Phase 3: Experimental Framework (2 hrs)

#### Test Generator (`experiments/test_generator.py`)
```python
def generate_test_cases(schemas, num_cases=25):
    generator = Agent(
        name="test_generator",
        instructions="""Generate SQL test cases covering:
        - Simple filters (8 cases)
        - Aggregations (5 cases)
        - Time-based queries (5 cases)
        - Multi-table joins (4 cases)
        - Complex queries (3 cases)
        
        For each: question, expected_query, difficulty, required_tables"""
    )
    
    return generator.run(schemas=schemas, num_cases=num_cases)
```

#### Evaluator (`experiments/evaluator.py`)
```python
class Evaluator:
    def evaluate(self, agent, test_cases):
        results = []
        for test in test_cases:
            start = time.time()
            response = agent.run(test['question'])
            latency = time.time() - start
            
            # Check correctness (LLM-as-judge)
            correct = self.check_semantic_equivalence(
                response['query'], 
                test['expected_query']
            )
            
            results.append({
                'correct': correct,
                'latency': latency,
                'tokens': response.get('tokens', 0)
            })
        
        return self.summarize(results)
```

### Phase 4: Unit Testing & Examples (1.5 hrs)

#### Critical Unit Tests (1 hr)

**Retrieval Tests** (`tests/test_retrieval.py`) - HIGHEST PRIORITY
```python
import pytest
from src.retrieval.semantic_retrieval import SemanticRetrieval
from src.retrieval.keyword_retrieval import KeywordRetrieval

class TestSemanticRetrieval:
    @pytest.fixture
    def sample_schemas(self):
        return {
            "login_events": {
                "description": "Records of user authentication attempts",
                "fields": [
                    {"name": "user_id", "type": "INTEGER"},
                    {"name": "status", "type": "STRING"},  # FAILED/SUCCESS
                    {"name": "timestamp", "type": "TIMESTAMP"}
                ]
            },
            "file_access_events": {
                "description": "File access and modification events",
                "fields": [
                    {"name": "user_id", "type": "INTEGER"},
                    {"name": "file_path", "type": "STRING"},
                    {"name": "action", "type": "STRING"}
                ]
            },
            "network_connections": {
                "description": "Network connection logs",
                "fields": [
                    {"name": "source_ip", "type": "STRING"},
                    {"name": "destination_ip", "type": "STRING"}
                ]
            }
        }
    
    def test_semantic_similarity_finds_relevant_table(self, sample_schemas):
        """Semantic search should find 'login_events' for authentication queries"""
        retriever = SemanticRetrieval(sample_schemas)
        
        # Query about failed logins - should retrieve login_events
        results = retriever.get_top_k("show me failed authentication attempts", k=2)
        
        table_names = [r['table_name'] for r in results]
        assert "login_events" in table_names
        assert results[0]['table_name'] == "login_events"  # Should be top result
    
    def test_top_k_returns_correct_count(self, sample_schemas):
        """Should return exactly k tables"""
        retriever = SemanticRetrieval(sample_schemas)
        
        results = retriever.get_top_k("show me all events", k=2)
        assert len(results) == 2
    
    def test_retrieval_ranking_by_relevance(self, sample_schemas):
        """More relevant tables should rank higher"""
        retriever = SemanticRetrieval(sample_schemas)
        
        # Network query should rank network_connections highest
        results = retriever.get_top_k("suspicious IP addresses", k=3)
        assert results[0]['table_name'] == "network_connections"
    
    def test_empty_query_handling(self, sample_schemas):
        """Empty queries should return error or default behavior"""
        retriever = SemanticRetrieval(sample_schemas)
        
        with pytest.raises(ValueError):
            retriever.get_top_k("", k=1)
    
    def test_embedding_dimensions(self, sample_schemas):
        """Embeddings should have consistent dimensions"""
        retriever = SemanticRetrieval(sample_schemas)
        
        # Check all embeddings have same dimension
        dims = [len(emb) for emb in retriever.table_embeddings.values()]
        assert len(set(dims)) == 1  # All same dimension
        assert dims[0] == 1536  # text-embedding-3-small dimension

class TestKeywordRetrieval:
    def test_exact_keyword_match(self, sample_schemas):
        """Keyword 'login' should find 'login_events'"""
        retriever = KeywordRetrieval(sample_schemas)
        
        results = retriever.get_top_k("show me login data", k=1)
        assert "login_events" in [r['table_name'] for r in results]
    
    def test_no_keyword_match_returns_fallback(self, sample_schemas):
        """When no keywords match, should return reasonable fallback"""
        retriever = KeywordRetrieval(sample_schemas)
        
        results = retriever.get_top_k("xyz123nonsense", k=1)
        assert len(results) > 0  # Should return something, not fail
    
    def test_field_name_matching(self, sample_schemas):
        """Should match field names too, not just table names"""
        retriever = KeywordRetrieval(sample_schemas)
        
        # "user_id" appears in multiple tables
        results = retriever.get_top_k("user_id statistics", k=3)
        assert len(results) >= 2  # Should find multiple tables with user_id
```

**Schema Loader Tests** (`tests/test_schema_loader.py`)
```python
from src.utils.schema_loader import load_schemas

def test_load_schemas_from_directory():
    """Should load all schema JSON files from directory"""
    schemas = load_schemas('test_schemas/')
    assert len(schemas) > 0
    assert all('fields' in schema for schema in schemas.values())

def test_schema_validation():
    """Should reject malformed schemas"""
    with pytest.raises(ValueError):
        load_schemas('test_schemas/invalid_schema.json')
```

**Validation Tests** (`tests/test_validation.py`)
```python
from src.utils.validator import SQLValidator

def test_valid_sql_syntax():
    """Should accept syntactically correct SQL"""
    validator = SQLValidator()
    query = "SELECT * FROM users WHERE active = true"
    assert validator.is_valid(query) == True

def test_invalid_sql_syntax():
    """Should reject malformed SQL"""
    validator = SQLValidator()
    query = "SLECT * FORM users"  # Typos
    assert validator.is_valid(query) == False

def test_table_exists_validation():
    """Should verify referenced tables exist in schema"""
    schemas = {"users": {...}, "events": {...}}
    validator = SQLValidator(schemas)
    
    # Valid table reference
    assert validator.table_exists("SELECT * FROM users") == True
    
    # Invalid table reference
    assert validator.table_exists("SELECT * FROM nonexistent") == False
```

**Agent Integration Tests** (`tests/test_agents.py`)
```python
from src.agents.semantic_agent import SemanticAgent

def test_semantic_agent_end_to_end():
    """Full pipeline test: question -> SQL"""
    agent = SemanticAgent(schema_path='test_schemas/')
    
    response = agent.run("Show me failed login attempts")
    
    assert response['query'] is not None
    assert 'SELECT' in response['query'].upper()
    assert response['confidence'] > 0
    assert len(response['tables_used']) > 0

def test_agent_confidence_scoring():
    """Agent should provide confidence scores"""
    agent = SemanticAgent(schema_path='test_schemas/')
    
    # Clear query should have high confidence
    clear_response = agent.run("SELECT all users")
    assert clear_response['confidence'] > 0.7
    
    # Ambiguous query should have lower confidence
    ambiguous_response = agent.run("show me stuff")
    assert ambiguous_response['confidence'] < clear_response['confidence']
```

#### Example Queries (0.5 hrs)
Implement 5 required examples in `examples/example_queries.py` demonstrating:
1. Simple filter query
2. Aggregation query
3. Time-based query
4. Multi-table join
5. Complex conditions

### Phase 5: Documentation (1 hr)
- **Main README:** Setup, architecture, examples, bonus features
- **Experiments README:** Methodology, results, findings
- **Video prep:** Script for 5-min demo

---

## Important Agno Patterns & Gotchas

### Dependencies Management
All dependencies are managed via `requirements.txt` and `requirements-dev.txt`.

**Production dependencies** (requirements.txt):
```txt
agno>=0.1.0
openai>=1.0.0
pydantic>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0  # Optional but recommended for .env loading
```

**Development dependencies** (requirements-dev.txt):
```txt
-r requirements.txt  # Includes production deps
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0
```

**Installation:**
```bash
# Virtual environment must be active
pip install -r requirements.txt        # Production only
pip install -r requirements-dev.txt    # Includes testing tools
```

### Structured Outputs
```python
from pydantic import BaseModel

class SQLQueryResponse(BaseModel):
    query: str
    explanation: str
    tables_used: list[str]
    confidence: float
    reasoning_steps: list[str] = []  # Optional bonus

# Use in agent
agent = Agent(
    model="gpt-4",
    response_model=SQLQueryResponse  # ← Forces JSON schema
)
```

### Context Management
- Keep schema context under ~8K tokens for fast inference
- If using GPT-4: ~128K context, but retrieval keeps it focused
- Pre-filter schemas before sending to LLM

### Error Handling
```python
try:
    response = agent.run(question=q, schemas=schemas)
except Exception as e:
    return {
        "query": None,
        "error": str(e),
        "confidence": 0.0
    }
```

### Embeddings
```python
from openai import OpenAI
import os

# Load API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Pre-compute once, cache
table_embeddings = {
    table_name: embed_text(f"{table_name}: {description}")
    for table_name, description in schemas.items()
}
```

**Note:** For production, consider using `python-dotenv` to auto-load .env file:
```python
from dotenv import load_dotenv
load_dotenv()  # Automatically loads .env variables

# Now os.getenv("OPENAI_API_KEY") works
```

---

## Deliverables Checklist

**Core (Required):**
- [ ] **Environment Setup:**
  - [ ] .gitignore (exclude .venv, .env, __pycache__)
  - [ ] requirements.txt (production dependencies)
  - [ ] requirements-dev.txt (testing dependencies)
  - [ ] Setup verification script or instructions
- [ ] Working semantic agent (Variant B)
- [ ] 5 example queries demonstrating functionality
- [ ] **Unit tests for all components:**
  - [ ] Retrieval tests (keyword & semantic) - CRITICAL
  - [ ] Schema loader tests
  - [ ] Validation tests
  - [ ] Agent integration tests
- [ ] README with setup, architecture, examples
  - [ ] Include virtual environment setup instructions
  - [ ] Include Python version requirement (3.10+)

**Bonus Points:**
- [ ] Confidence scoring
- [ ] Query explanation in natural language
- [ ] Multi-step reasoning (visible in output)
- [ ] Schema similarity search (semantic retrieval)
- [ ] Query validation

**Experimental (Differentiator):**
- [ ] Test case generator
- [ ] Evaluation harness
- [ ] Comparison of 2-3 architectures
- [ ] Results analysis (comparison.md)
- [ ] Experiments README

**Submission:**
- [ ] GitHub repo with clean commit history
- [ ] 5-min video demo

---

## Key Messages for Video Demo

1. **Core Solution:** "Semantic search + single agent balances accuracy and speed"
2. **Bonus Features:** "Confidence scoring and explanations aid debugging"
3. **Experimental Framework:** "Data-driven architecture decisions - semantic search improves correctness 16% over baseline with acceptable latency increase"
4. **Production Thinking:** "This evaluation approach mirrors how I'd tackle Mate Security's slow/inaccurate agent challenges"

---

## Quick Reference: CLI Usage

```bash
# Main submission (semantic agent)
python main.py "Show me high-severity events from last 24 hours"

# Compare architectures
python main.py "..." --architecture=baseline
python main.py "..." --architecture=semantic

# Run unit tests (DO THIS FIRST)
pytest tests/ -v                          # All tests
pytest tests/test_retrieval.py -v         # Just retrieval tests
pytest tests/test_retrieval.py::test_semantic_similarity_finds_relevant_table -v  # Single test

# Run experiments
python experiments/test_generator.py --num-cases=25
python experiments/run_experiments.py
```

---

## Development Workflow (Recommended Order)

1. **Setup** → Install dependencies, load schemas
2. **Unit Tests** → Write and pass retrieval tests (TDD approach)
3. **Core Agent** → Build semantic agent with test coverage
4. **Integration** → Verify end-to-end with example queries
5. **Experiments** → Generate test cases, run evaluations
6. **Documentation** → README and video prep

**Key Principle:** Test early, test often. Don't wait until the end.
