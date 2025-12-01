# Python Code Review: SQL Query Agent

**Date**: 2025-12-01
**Reviewer**: Claude
**Scope**: Full codebase review for structure, readability, and best practices

---

## 1. Code Structure

### Strengths

**Logical Organization** - The project follows a clean layered architecture:
```
src/
├── agents/      # Agent implementations (generation layer)
├── retrieval/   # Retrieval strategies (data access layer)
└── utils/       # Shared utilities (cross-cutting concerns)
```

**Good Separation of Concerns**:
- Abstract base class (`src/agents/base.py`) defines contract for agents
- Retrieval strategies are pluggable (semantic vs keyword)
- Validation is isolated in its own module
- Schema loading separated from business logic

**Appropriate Use of Classes vs Functions**:
- Classes for stateful components (`SemanticAgent`, `SQLValidator`, `SemanticRetrieval`)
- Module-level functions for stateless utilities (`load_schemas`, `format_schema_for_llm`)

### Issues

| Priority | Issue | Location |
|----------|-------|----------|
| **Important** | Code Duplication | `semantic_agent.py` and `keyword_agent.py` are ~95% identical |

**Details**: The only difference between the two agent classes is the retriever type. Lines 102-156 in both files are nearly identical.

**Suggestion**: Use dependency injection or a factory pattern:
```python
class SQLQueryAgent(BaseAgent):
    def __init__(self, schema_path: str, retriever: BaseRetriever, ...):
        self.retriever = retriever  # Inject the retrieval strategy
```

---

## 2. Readability

### Strengths

**Clear Naming**: Variable and function names are descriptive:
- `get_top_k()`, `validate_fields()`, `check_dangerous_operations()`
- `table_embeddings`, `relevant_tables`, `schema_context`

**Consistent Docstrings**: All public methods have comprehensive docstrings following Google style:
```python
def get_top_k(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant tables based on semantic similarity.

    Args:
        question: Natural language question
        k: Number of tables to retrieve

    Returns:
        List of table schemas with metadata, sorted by relevance
    """
```

**Good Formatting**: Consistent 4-space indentation, reasonable line lengths.

### Issues

| Priority | Issue | Location |
|----------|-------|----------|
| **Nice-to-have** | Overly Long Function | `schema_loader.py:91-191` |
| **Nice-to-have** | Magic Numbers | Multiple locations |

**Details**:
- `get_table_description()` is 100 lines with nested conditionals. Could be split into helper functions.
- `keyword_retrieval.py:72`: `len(t) > 2` - why 2?
- `semantic_agent.py:62`: `cache_ttl=3600` - what's the rationale?

---

## 3. Python Best Practices

### Strengths

**Type Hints**: Comprehensive and accurate throughout:
```python
def validate(self, query: str, strict: bool = False) -> Dict[str, Any]:
```

**Context Awareness**: Proper use of `Path` for file operations (`schema_loader.py:22`)

**Pydantic Models**: Clean data validation with `SQLQueryResponse`:
```python
confidence: float = Field(ge=0.0, le=1.0)  # Built-in validation
```

### Issues

| Priority | Issue | Location |
|----------|-------|----------|
| **Important** | Import at Runtime | `semantic_retrieval.py:66` |
| **Important** | Bare Exception Catches | Multiple locations |
| **Nice-to-have** | Unused Import | `semantic_agent.py:3`, `keyword_agent.py:3` |

**Details**:

1. **Late Import**:
```python
def _precompute_embeddings(self) -> None:
    from src.utils.schema_loader import get_table_description  # Late import
```
This creates a hidden dependency. Move to top of file.

2. **Bare Exceptions**:
- `semantic_retrieval.py:62-63`
- `semantic_agent.py:160-161`

Should catch specific exceptions or at least log the exception type.

3. **Unused Import**:
```python
import os  # Never used in semantic_agent.py and keyword_agent.py
```

---

## 4. Potential Issues

### Critical

| Issue | Location | Description |
|-------|----------|-------------|
| None identified | - | - |

### Important

| Issue | Location | Description |
|-------|----------|-------------|
| **Pickle Deserialization Risk** | `semantic_retrieval.py:51-52` | If an attacker can modify the cache file, arbitrary code execution is possible |
| **Print Statements in Library Code** | `semantic_retrieval.py:37-39` | Library code should use `logging` instead of `print` |

**Pickle Risk Details**:
```python
with open(self.cache_path, 'rb') as f:
    cache = pickle.load(f)
```
Consider using `safetensors` or JSON for embeddings.

**Print Statement Details**:
```python
print(f"Loading embedding model: {embedding_model}...")
print(f"✓ Model loaded")
```

### Nice-to-have

| Issue | Location | Description |
|-------|----------|-------------|
| No Input Sanitization | `validator.py` | Queries aren't executed, but displaying them could be risky in web contexts |

---

## 5. Suggestions

### Critical Priority

#### 1. Extract Common Agent Logic

Create a single `SQLQueryAgent` class with pluggable retriever:

```python
# src/agents/sql_agent.py
from abc import ABC
from typing import Protocol

class Retriever(Protocol):
    def get_top_k(self, question: str, k: int) -> list[dict]: ...

class SQLQueryAgent(BaseAgent):
    def __init__(
        self,
        schema_path: str,
        retriever: Retriever,
        model: str = "claude-sonnet-4-5",
        top_k_tables: int = 5
    ):
        self.schemas = load_schemas(schema_path)
        self.retriever = retriever
        self.top_k_tables = top_k_tables
        # ... rest of initialization

# Usage
semantic_retriever = SemanticRetrieval(schemas, cache_path="...")
agent = SQLQueryAgent(schema_path, retriever=semantic_retriever)

keyword_retriever = KeywordRetrieval(schemas)
agent = SQLQueryAgent(schema_path, retriever=keyword_retriever)
```

### Important Priority

#### 2. Replace Pickle with Safer Serialization

Use numpy's native format or JSON:

```python
import numpy as np

# Saving
np.savez(
    cache_path,
    model_name=np.array([self.embedding_model_name]),
    **{f"emb_{name}": emb for name, emb in self.table_embeddings.items()}
)

# Loading
cache = np.load(cache_path, allow_pickle=False)
```

Or use `safetensors`:
```python
from safetensors.numpy import save_file, load_file

save_file(self.table_embeddings, cache_path)
embeddings = load_file(cache_path)
```

#### 3. Use Logging Instead of Print

```python
import logging

logger = logging.getLogger(__name__)

class SemanticRetrieval:
    def __init__(self, ...):
        # Replace
        # print(f"Loading embedding model: {embedding_model}...")
        # With
        logger.info("Loading embedding model: %s", embedding_model)

        self.model = SentenceTransformer(embedding_model)

        # Replace
        # print(f"✓ Model loaded")
        # With
        logger.info("Model loaded successfully")
```

#### 4. Add Constants File

Create `src/constants.py`:

```python
"""Constants for SQL Query Agent."""

# Retrieval settings
MIN_KEYWORD_LENGTH = 3
DEFAULT_TOP_K_TABLES = 5

# Caching
DEFAULT_CACHE_TTL_SECONDS = 3600

# Embedding model
DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# Validation
DANGEROUS_SQL_OPERATIONS = frozenset({
    'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
    'ALTER', 'CREATE', 'GRANT', 'REVOKE'
})
```

### Nice-to-have Priority

#### 5. Consider Using `@dataclass` for Simple Data Classes

Some dict returns could be dataclasses for better type safety:

```python
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    table_name: str
    schema: dict
    score: float
    match_type: str
```

#### 6. Add `__all__` exports

In `__init__.py` files to make public API explicit:

```python
# src/agents/__init__.py
__all__ = ['BaseAgent', 'SemanticAgent', 'KeywordAgent', 'SQLQueryResponse']
```

#### 7. Move Late Import to Top of File

In `semantic_retrieval.py`:
```python
# Move this to the top of the file with other imports
from src.utils.schema_loader import get_table_description
```

---

## Summary

| Category | Rating | Notes |
|----------|--------|-------|
| Code Structure | **B+** | Good layering, but significant duplication between agents |
| Readability | **A-** | Clear naming, good docstrings, some long functions |
| Python Best Practices | **B+** | Good type hints, some anti-patterns (late imports, bare exceptions) |
| Potential Issues | **B** | Pickle security concern, print statements in library code |

### Priority Action Items

- [ ] **Critical**: Extract common agent logic to eliminate duplication
- [ ] **Important**: Replace pickle with safer serialization (numpy or safetensors)
- [ ] **Important**: Convert print statements to proper logging
- [ ] **Important**: Move late imports to module level
- [ ] **Important**: Catch specific exceptions instead of bare `except`
- [ ] **Nice-to-have**: Extract constants to dedicated file
- [ ] **Nice-to-have**: Remove unused imports
- [ ] **Nice-to-have**: Split long functions into smaller helpers

---

## Files Reviewed

| File | Lines | Notes |
|------|-------|-------|
| `src/agents/base.py` | 76 | Clean abstract base class |
| `src/agents/semantic_agent.py` | 209 | Main agent, duplicated code |
| `src/agents/keyword_agent.py` | 204 | Baseline agent, duplicated code |
| `src/retrieval/semantic_retrieval.py` | 230 | Good embedding logic, pickle concern |
| `src/retrieval/keyword_retrieval.py` | 176 | Clean keyword matching |
| `src/utils/schema_loader.py` | 228 | Long function needs refactoring |
| `src/utils/validator.py` | 323 | Comprehensive validation |
| `main.py` | 165 | Clean CLI entrypoint |
