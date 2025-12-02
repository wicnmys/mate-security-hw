# Keyword Agent Card

## Overview

The **KeywordAgent** is a lightweight SQL query generation agent that uses keyword-based table retrieval. It provides fast, deterministic table matching without requiring ML models or embeddings.

---

## System Components

### Core Classes & Modules

| Component | Location | Description |
|-----------|----------|-------------|
| `KeywordAgent` | `src/agents/keyword_agent.py` | Main agent class, convenience wrapper |
| `SQLQueryAgent` | `src/agents/sql_agent.py` | Parent class providing query generation |
| `KeywordRetrieval` | `src/retrieval/keyword_retrieval.py` | Keyword-based table search |
| `SQLValidator` | `src/utils/validator.py` | SQL syntax and schema validation |
| `SchemaLoader` | `src/utils/schema_loader.py` | Table definition loading |

### Data Models

| Model | Purpose |
|-------|---------|
| `SQLQueryResponse` | Pydantic model for structured output (query, explanation, tables_used, confidence, reasoning_steps) |

### External Dependencies

- **Agno Framework**: LLM agent orchestration
- **Claude (Anthropic)**: Language model for SQL generation
- **Pydantic**: Data validation and serialization

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      KeywordAgent                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   SQLQueryAgent                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ││
│  │  │KeywordRetrieval│  │  Agno Agent │  │ SQLValidator  │  ││
│  │  │   (Injected)   │  │  (Claude)   │  │  (Optional)   │  ││
│  │  └───────┬────────┘  └──────┬──────┘  └───────┬───────┘  ││
│  │          │                  │                  │          ││
│  │          ▼                  ▼                  ▼          ││
│  │    Table Schemas      SQL Generation      Validation      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Convenience Wrapper Pattern
The KeywordAgent is a thin wrapper that preconfigures `SQLQueryAgent` with keyword-based retrieval. This promotes code reuse while providing a simple interface.

```python
class KeywordAgent(SQLQueryAgent):
    def __init__(self, schema_path: str, ...):
        retriever = KeywordRetrieval(tables)
        super().__init__(retriever=retriever, ...)
```

### 2. Keyword Index Construction
Builds searchable index from multiple sources:
- Table names (tokenized)
- Category metadata
- Description text
- Field names and descriptions

### 3. Token Filtering
Filters keywords under 3 characters (`MIN_KEYWORD_LENGTH`) to reduce noise and improve match quality.

### 4. Fallback Strategy
Returns largest tables by field count when no keyword matches are found, ensuring the agent always has context to work with.

### 5. Deterministic Retrieval
Unlike semantic agents, keyword matching is fully deterministic - same input always produces same output.

---

## Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **Adapter Pattern** | Wraps `KeywordRetrieval` to provide `Retriever` protocol interface |
| **Factory Pattern** | Creates fully configured `SQLQueryAgent` instance |
| **Strategy Pattern** | `KeywordRetrieval` is injected as the retrieval strategy |
| **Template Method** | Inherits `run()` flow from parent class |

---

## Processing Flow

```
1. Question Input
       │
       ▼
2. Keyword Extraction
   └── Tokenize question
   └── Filter short tokens
       │
       ▼
3. Table Retrieval
   └── Match keywords against index
   └── Score and rank tables
   └── Return top-k with relevance scores
       │
       ▼
4. Schema Context Building
   └── Format retrieved table schemas
   └── Include field definitions
       │
       ▼
5. SQL Generation (Claude LLM)
   └── Apply security database context
   └── Generate query with confidence score
       │
       ▼
6. Validation (Optional)
   └── Syntax check
   └── Schema validation
   └── Adjust confidence if issues found
       │
       ▼
7. Structured Response
   └── SQLQueryResponse with query, explanation, tables_used, confidence
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema_path` | Required | Path to table schema files |
| `top_k` | 5 | Number of tables to retrieve |
| `model_id` | `claude-sonnet-4-20250514` | LLM model for generation |
| `validator` | Auto-created | Optional custom SQL validator |

---

## Strengths & Limitations

### Strengths
- Fast initialization (no embedding computation)
- No ML model dependencies
- Deterministic and debuggable
- Low memory footprint
- Simple to understand and maintain

### Limitations
- Cannot handle synonyms or paraphrasing
- Exact keyword matching only
- May miss semantically related tables
- Less robust to natural language variations

---

## Usage Example

```python
from src.agents.keyword_agent import KeywordAgent

agent = KeywordAgent(
    schema_path="schemas/",
    top_k=5,
    model_id="claude-sonnet-4-20250514"
)

result = agent.run("Show me all failed login attempts in the last 24 hours")
print(result["query"])
print(result["confidence"])
```

---

## Related Components

- [SemanticAgent](./semantic_agent.md) - Embedding-based alternative
- [ReActAgent](./react_agent.md) - Iterative self-correcting agent
- [ReActAgentV2](./react_agent_v2.md) - Enhanced with LLM judge
