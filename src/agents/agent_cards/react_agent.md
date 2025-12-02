# ReAct Agent Card

## Overview

The **ReActAgent** is an iterative, self-correcting SQL query generation agent that implements the ReAct (Reasoning + Acting) paradigm. It uses a tool-based approach with explicit reasoning traces, allowing the agent to retrieve tables, validate queries, and iteratively improve until reaching a satisfactory answer.

---

## System Components

### Core Classes & Modules

| Component | Location | Description |
|-----------|----------|-------------|
| `ReActAgent` | `src/agents/react_agent.py` | Main agent class with tool-use loop |
| `AgentState` | `src/agents/react_agent.py` | Dataclass tracking iteration state |
| `BaseAgent` | `src/agents/base.py` | Abstract base class |
| `SemanticRetrieval` | `src/retrieval/semantic_retrieval.py` | Embedding-based table search |
| `KeywordRetrieval` | `src/retrieval/keyword_retrieval.py` | Keyword-based table search |
| `SQLValidator` | `src/utils/validator.py` | SQL syntax and schema validation |

### Data Models

| Model | Purpose |
|-------|---------|
| `AgentState` | Tracks question, iteration, retrieved tables, queries, validation results, reasoning trace |
| `SQLQueryResponse` | Pydantic model for structured output |

### Tools

| Tool | Purpose |
|------|---------|
| `retrieve_tables` | Search and retrieve relevant table schemas |
| `validate_sql` | Validate SQL syntax and schema correctness |
| `submit_answer` | Submit final answer and terminate loop |

### External Dependencies

- **Agno Framework**: LLM agent orchestration with tool support
- **Claude (Anthropic)**: Language model for reasoning and generation
- **sentence-transformers**: Optional, for semantic retrieval mode
- **Pydantic**: Data validation and serialization

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         ReActAgent                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                     AgentState                              │  │
│  │  question | iteration | retrieved_tables | generated_queries│  │
│  │  validation_results | reasoning_trace | final_answer        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │                    Agno Agent (Claude)                      │  │
│  │              with ReAct Instructions                        │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ retrieve_   │     │ validate_   │     │ submit_     │        │
│  │   tables    │     │    sql      │     │  answer     │        │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘        │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  Semantic/  │     │    SQL      │     │   Final     │        │
│  │  Keyword    │     │  Validator  │     │  Response   │        │
│  │  Retrieval  │     │             │     │             │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. ReAct Paradigm Implementation
The agent follows the Thought-Action-Observation loop:

```
Thought: Analyze question and current state
Action: Call a tool (retrieve, validate, or submit)
Observation: Process tool result
(Repeat until submit_answer is called)
```

### 2. Stateful Iteration Tracking
`AgentState` dataclass maintains complete state across iterations:

```python
@dataclass
class AgentState:
    question: str
    iteration: int = 0
    retrieved_tables: list = field(default_factory=list)
    generated_queries: list = field(default_factory=list)
    validation_results: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    final_answer: Optional[dict] = None
    retrieval_calls: int = 0
    validation_attempts: int = 0
```

### 3. Loop Control Mechanisms
Prevents infinite loops and resource abuse:

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_iterations` | 10 | Total tool calls allowed |
| `max_retrieval_calls` | 3 | Prevents over-searching |
| `max_validation_attempts` | 4 | Prevents infinite validation loops |

### 4. Tool-Scoped State via Closures
Tools access agent state through closures, enabling state updates within the tool execution context.

### 5. Validation with Actionable Feedback
When validation fails, the agent receives specific error messages and suggestions to guide correction.

### 6. Dual Retriever Support
Can use either semantic or keyword retrieval based on configuration.

---

## Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **ReAct Pattern** | Core reasoning + acting loop |
| **State Machine Pattern** | Iteration and state tracking |
| **Tool Use Pattern** | Agno framework tool integration |
| **Closure Pattern** | Tools access agent state |
| **Strategy Pattern** | Pluggable retrieval strategies |
| **Template Method** | Base `run()` method structure |

---

## Processing Flow

```
1. Question Input
       │
       ▼
2. Initialize AgentState
   └── Set question
   └── Reset counters
       │
       ▼
┌──────▼──────────────────────────────────────────┐
│            ReAct Loop (max 10 iterations)        │
│  ┌─────────────────────────────────────────────┐│
│  │ Thought: Agent reasons about next step      ││
│  └──────────────────┬──────────────────────────┘│
│                     │                            │
│  ┌──────────────────▼──────────────────────────┐│
│  │ Action: Select and call tool                ││
│  │  ├── retrieve_tables (if need more context) ││
│  │  ├── validate_sql (if query generated)      ││
│  │  └── submit_answer (if ready)               ││
│  └──────────────────┬──────────────────────────┘│
│                     │                            │
│  ┌──────────────────▼──────────────────────────┐│
│  │ Observation: Process tool result            ││
│  │  ├── Tables with schemas and scores         ││
│  │  ├── Validation success/errors              ││
│  │  └── (Loop terminates on submit_answer)     ││
│  └──────────────────┬──────────────────────────┘│
│                     │                            │
│            (Continue loop)                       │
└──────────────────────────────────────────────────┘
       │
       ▼
3. Final Response
   └── SQLQueryResponse with reasoning trace
```

---

## Tool Specifications

### `retrieve_tables(question: str, top_k: int = 5)`
- Searches for relevant tables based on question
- Returns formatted schemas with relevance scores
- Tracks retrieval count, enforces limits
- Uses configured retriever (semantic or keyword)

### `validate_sql(query: str)`
- Validates SQL syntax using sqlglot
- Checks table and field existence against schema
- Detects dangerous operations (DROP, DELETE, etc.)
- Returns validation status with detailed errors
- Provides actionable suggestions for fixing issues

### `submit_answer(query: str, explanation: str, confidence: float)`
- Submits final answer, terminates ReAct loop
- Performs final validation
- Adjusts confidence based on validation results
- Stores complete reasoning trace

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema_path` | Required | Path to table schema files |
| `retrieval_type` | `"semantic"` | `"semantic"` or `"keyword"` |
| `top_k` | 5 | Number of tables to retrieve |
| `model_id` | `claude-sonnet-4-20250514` | LLM model for reasoning |
| `max_iterations` | 10 | Maximum tool calls per query |
| `max_retrieval_calls` | 3 | Maximum retrieval operations |
| `max_validation_attempts` | 4 | Maximum validation attempts |

---

## Strengths & Limitations

### Strengths
- Self-correcting through iterative validation
- Explicit reasoning traces for transparency
- Handles complex queries requiring multiple steps
- Validation feedback guides improvements
- Complete audit trail of agent decisions
- Configurable retrieval strategy

### Limitations
- Higher latency due to multiple LLM calls
- More expensive (multiple tool iterations)
- May get stuck in validation loops
- Requires careful prompt engineering for ReAct instructions
- Loop limits may terminate before finding solution

---

## Usage Example

```python
from src.agents.react_agent import ReActAgent

agent = ReActAgent(
    schema_path="schemas/",
    retrieval_type="semantic",
    top_k=5,
    max_iterations=10
)

result = agent.run("Find users with more than 5 failed login attempts who haven't been blocked")
print(result["query"])
print(result["reasoning_steps"])  # Full reasoning trace
print(result["confidence"])
```

---

## ReAct Prompt Structure

The agent is initialized with ReAct-specific instructions:

```
You are a SQL query generation agent using the ReAct framework.

For each question, follow this process:
1. THOUGHT: Analyze what you need to do
2. ACTION: Use a tool to make progress
3. OBSERVATION: Review the result
4. Repeat until you can submit a final answer

Available tools:
- retrieve_tables: Get relevant table schemas
- validate_sql: Check SQL syntax and correctness
- submit_answer: Submit your final SQL query
```

---

## Related Components

- [KeywordAgent](./keyword_agent.md) - Simple keyword-based agent
- [SemanticAgent](./semantic_agent.md) - Semantic retrieval agent
- [ReActAgentV2](./react_agent_v2.md) - Enhanced with LLM judge validation
