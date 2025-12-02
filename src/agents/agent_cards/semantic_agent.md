# Semantic Agent Card

## Overview

The **SemanticAgent** is an advanced SQL query generation agent that uses embedding-based semantic retrieval. It leverages sentence-transformers to understand the semantic meaning of questions and match them with relevant database tables.

---

## System Components

### Core Classes & Modules

| Component | Location | Description |
|-----------|----------|-------------|
| `SemanticAgent` | `src/agents/semantic_agent.py` | Main agent class, convenience wrapper |
| `SQLQueryAgent` | `src/agents/sql_agent.py` | Parent class providing query generation |
| `SemanticRetrieval` | `src/retrieval/semantic_retrieval.py` | Embedding-based table search |
| `SQLValidator` | `src/utils/validator.py` | SQL syntax and schema validation |
| `SchemaLoader` | `src/utils/schema_loader.py` | Table definition loading |

### Data Models

| Model | Purpose |
|-------|---------|
| `SQLQueryResponse` | Pydantic model for structured output (query, explanation, tables_used, confidence, reasoning_steps) |

### External Dependencies

- **Agno Framework**: LLM agent orchestration
- **Claude (Anthropic)**: Language model for SQL generation
- **sentence-transformers**: Local embedding generation
- **NumPy**: Embedding computation and caching
- **Pydantic**: Data validation and serialization

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       SemanticAgent                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SQLQueryAgent                             ││
│  │  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐    ││
│  │  │SemanticRetrieval│  │  Agno Agent │  │ SQLValidator  │    ││
│  │  │   (Injected)    │  │  (Claude)   │  │  (Optional)   │    ││
│  │  └───────┬─────────┘  └──────┬──────┘  └───────┬───────┘    ││
│  │          │                   │                  │            ││
│  │  ┌───────▼─────────┐         ▼                  ▼            ││
│  │  │ Sentence-       │   SQL Generation      Validation        ││
│  │  │ Transformers    │                                         ││
│  │  │ (Embeddings)    │                                         ││
│  │  └───────┬─────────┘                                         ││
│  │          │                                                    ││
│  │  ┌───────▼─────────┐                                         ││
│  │  │ Embedding Cache │                                         ││
│  │  │   (.npz file)   │                                         ││
│  │  └─────────────────┘                                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Pre-computed Embeddings
All table embeddings are computed at initialization and cached, avoiding redundant computation during query time.

```python
# Embeddings computed once at init
self.table_embeddings = self._compute_embeddings(tables)
```

### 2. Cache-First Approach
Embedding cache is checked before recomputing:
- Cache stored in NumPy `.npz` format
- Model name validation ensures cache compatibility
- Significant speedup on subsequent loads

### 3. Sentence-Transformers Model
Uses `multi-qa-mpnet-base-dot-v1` model optimized for:
- Question-answering tasks
- Semantic similarity
- Dot product scoring

### 4. Cosine Similarity Matching
Questions and table descriptions are compared using cosine similarity in the embedding space.

### 5. Table Description Generation
Each table generates a rich description for embedding:
- Table name and category
- Description text
- Field names and their descriptions

---

## Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **Adapter Pattern** | Wraps `SemanticRetrieval` to provide `Retriever` protocol interface |
| **Factory Pattern** | Creates fully configured `SQLQueryAgent` instance |
| **Strategy Pattern** | `SemanticRetrieval` is injected as the retrieval strategy |
| **Memoization Pattern** | Embedding caching for performance optimization |
| **Template Method** | Inherits `run()` flow from parent class |

---

## Processing Flow

```
1. Question Input
       │
       ▼
2. Question Embedding
   └── Encode with sentence-transformers
   └── Generate dense vector representation
       │
       ▼
3. Similarity Computation
   └── Compare against all table embeddings
   └── Compute cosine similarity scores
       │
       ▼
4. Table Retrieval
   └── Rank by similarity score
   └── Return top-k with relevance scores
       │
       ▼
5. Schema Context Building
   └── Format retrieved table schemas
   └── Include field definitions
       │
       ▼
6. SQL Generation (Claude LLM)
   └── Apply security database context
   └── Generate query with confidence score
       │
       ▼
7. Validation (Optional)
   └── Syntax check
   └── Schema validation
   └── Adjust confidence if issues found
       │
       ▼
8. Structured Response
   └── SQLQueryResponse with query, explanation, tables_used, confidence
```

---

## Embedding Cache Structure

```
cache.npz
├── model_name: str          # For validation
├── table_names: ndarray     # Array of table names
├── descriptions: ndarray    # Array of descriptions
├── emb_0: ndarray          # Table 0 embedding
├── emb_1: ndarray          # Table 1 embedding
└── emb_N: ndarray          # Table N embedding
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema_path` | Required | Path to table schema files |
| `top_k` | 5 | Number of tables to retrieve |
| `model_id` | `claude-sonnet-4-20250514` | LLM model for generation |
| `embedding_model` | `multi-qa-mpnet-base-dot-v1` | Sentence-transformers model |
| `cache_path` | Optional | Path to embedding cache file |
| `validator` | Auto-created | Optional custom SQL validator |

---

## Strengths & Limitations

### Strengths
- Understands semantic meaning and intent
- Handles synonyms and paraphrasing
- Better for natural language variations
- Finds semantically related tables even without exact keyword matches
- Supports JOIN suggestions via `search_similar_tables`

### Limitations
- Requires sentence-transformers dependency
- Initial embedding computation takes time
- Higher memory footprint than keyword-based
- Embedding model may not understand domain-specific terms
- Non-deterministic (floating point variations)

---

## Usage Example

```python
from src.agents.semantic_agent import SemanticAgent

agent = SemanticAgent(
    schema_path="schemas/",
    top_k=5,
    cache_path="embeddings_cache.npz",
    model_id="claude-sonnet-4-20250514"
)

result = agent.run("What are the most common attack vectors this month?")
print(result["query"])
print(result["confidence"])
```

---

## Advanced Features

### Similar Table Search
Useful for identifying potential JOIN candidates:

```python
similar = retriever.search_similar_tables("users", top_k=3)
# Returns tables semantically related to "users"
```

---

## Related Components

- [KeywordAgent](./keyword_agent.md) - Lightweight keyword-based alternative
- [ReActAgent](./react_agent.md) - Iterative self-correcting agent
- [ReActAgentV2](./react_agent_v2.md) - Enhanced with LLM judge
