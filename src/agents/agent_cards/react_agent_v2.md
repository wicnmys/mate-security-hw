# ReAct Agent V2 Card

## Overview

The **ReActAgentV2** is an enhanced version of the ReAct agent that implements a dual validation system combining structural validation with semantic LLM-based evaluation. It uses a dedicated "judge" LLM to assess whether generated SQL queries correctly answer the original question.

---

## System Components

### Core Classes & Modules

| Component | Location | Description |
|-----------|----------|-------------|
| `ReActAgentV2` | `src/agents/react_agent_v2.py` | Main agent class with dual validation |
| `AgentStateV2` | `src/agents/react_agent_v2.py` | Extended state with judge tracking |
| `LLMJudgeOutput` | `src/agents/react_agent_v2.py` | Pydantic model for judge evaluation |
| `BaseAgent` | `src/agents/base.py` | Abstract base class |
| `SemanticRetrieval` | `src/retrieval/semantic_retrieval.py` | Embedding-based table search |
| `KeywordRetrieval` | `src/retrieval/keyword_retrieval.py` | Keyword-based table search |
| `SQLValidator` | `src/utils/validator.py` | SQL syntax and schema validation |

### Data Models

| Model | Purpose |
|-------|---------|
| `AgentStateV2` | Extended state with judge_results, judge_calls, judge_scores |
| `LLMJudgeOutput` | is_correct, correctness_score, issues, suggestions, reasoning |
| `SQLQueryResponse` | Pydantic model for structured output |

### Tools

| Tool | Purpose |
|------|---------|
| `retrieve_tables` | Search and retrieve relevant table schemas |
| `validate_sql` | Structural validation (syntax, schema) |
| `llm_judge_evaluate` | **NEW** - Semantic correctness evaluation |
| `submit_answer` | Submit final answer with combined scores |

### External Dependencies

- **Agno Framework**: LLM agent orchestration with tool support
- **Claude (Anthropic)**: Main agent + separate judge agent
- **sentence-transformers**: Optional, for semantic retrieval mode
- **Pydantic**: Data validation and serialization

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                           ReActAgentV2                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       AgentStateV2                                │  │
│  │  question | iteration | retrieved_tables | generated_queries     │  │
│  │  validation_results | reasoning_trace | final_answer             │  │
│  │  judge_results | judge_calls | judge_scores                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│  ┌─────────────────────────────▼────────────────────────────────────┐  │
│  │                      Agno Agent (Claude)                          │  │
│  │                  with Enhanced ReAct Instructions                 │  │
│  └─────────────────────────────┬────────────────────────────────────┘  │
│                                │                                        │
│       ┌────────────────────────┼────────────────────────┐              │
│       ▼                        ▼                        ▼              │
│ ┌───────────┐          ┌─────────────┐          ┌─────────────┐        │
│ │ retrieve_ │          │ validate_   │          │ submit_     │        │
│ │   tables  │          │    sql      │          │  answer     │        │
│ └─────┬─────┘          └──────┬──────┘          └──────┬──────┘        │
│       │                       │                        │                │
│       ▼                       ▼                        ▼                │
│ ┌───────────┐          ┌─────────────┐          ┌─────────────┐        │
│ │ Semantic/ │          │    SQL      │          │   Final     │        │
│ │ Keyword   │          │  Validator  │          │  Response   │        │
│ │ Retrieval │          │ (Structural)│          │             │        │
│ └───────────┘          └─────────────┘          └─────────────┘        │
│                                                                         │
│                        ┌─────────────┐                                 │
│                        │ llm_judge_  │                                 │
│                        │  evaluate   │                                 │
│                        └──────┬──────┘                                 │
│                               │                                         │
│                               ▼                                         │
│                  ┌─────────────────────────┐                           │
│                  │    Separate Judge Agent  │                           │
│                  │        (Claude)          │                           │
│                  │   - Semantic Evaluation  │                           │
│                  │   - Correctness Scoring  │                           │
│                  │   - Issue Identification │                           │
│                  └─────────────────────────┘                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Dual Validation System
Two complementary validation approaches:

| Validation Type | Tool | What It Checks |
|-----------------|------|----------------|
| **Structural** | `validate_sql` | Syntax, schema, field existence, dangerous ops |
| **Semantic** | `llm_judge_evaluate` | Does the query answer the question correctly? |

### 2. Separate Judge Agent
A dedicated Claude instance evaluates semantic correctness, preventing self-evaluation bias:

```python
self.judge_agent = Agent(
    model=Claude(id=model_id),
    instructions=JUDGE_INSTRUCTIONS,
    response_model=LLMJudgeOutput
)
```

### 3. Judge Rate Limiting
Maximum 3 judge calls per query to control costs (LLM judge is expensive).

### 4. Weighted Confidence Computation
Final confidence combines user-provided confidence with judge score:

```python
final_confidence = (user_confidence * 0.4) + (judge_score * 0.6)
# Further capped by structural validation results
```

### 5. Validation Pipeline Order
1. Always validate structurally first (fast, cheap)
2. Only evaluate semantically if structural validation passes
3. Judge feedback guides improvements if score < 0.7

### 6. Comprehensive Scoring Rubric
Judge uses detailed rubric for consistent evaluation (see below).

---

## Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **ReAct Pattern** | Enhanced reasoning + acting loop |
| **Validator Pattern** | Two validators for different concerns |
| **State Machine Pattern** | Extended iteration tracking |
| **Adapter Pattern** | Judge evaluation integration |
| **Strategy Pattern** | Pluggable retrieval strategies |
| **Separation of Concerns** | Structural vs semantic validation |
| **Chain of Responsibility** | Validation pipeline |

---

## Processing Flow

```
1. Question Input
       │
       ▼
2. Initialize AgentStateV2
   └── Set question
   └── Reset all counters (including judge_calls)
       │
       ▼
┌──────▼──────────────────────────────────────────────────────────┐
│                 ReAct Loop (max 10 iterations)                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Thought: Agent reasons about next step                     │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│  ┌─────────────────────▼──────────────────────────────────────┐ │
│  │ Action: Select and call tool                               │ │
│  │  ├── retrieve_tables (if need more context)                │ │
│  │  ├── validate_sql (structural check)                       │ │
│  │  ├── llm_judge_evaluate (semantic check) ◄── NEW           │ │
│  │  └── submit_answer (if ready)                              │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│  ┌─────────────────────▼──────────────────────────────────────┐ │
│  │ Observation: Process tool result                           │ │
│  │  ├── Tables with schemas and scores                        │ │
│  │  ├── Structural validation success/errors                  │ │
│  │  ├── Judge score with issues/suggestions ◄── NEW           │ │
│  │  └── (Loop terminates on submit_answer)                    │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│              (Continue loop)                                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
3. Final Response
   └── SQLQueryResponse with combined scores and reasoning trace
```

---

## Dual Validation Flow (Detailed)

```
                    Question
                        │
                        ▼
               ┌─────────────────┐
               │ retrieve_tables │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Generate SQL    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ validate_sql    │ ◄── Structural Validation
               └────────┬────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
        ┌───▼───┐              ┌────▼────┐
        │ FAIL  │              │  PASS   │
        └───┬───┘              └────┬────┘
            │                       │
            ▼                       ▼
    ┌───────────────┐      ┌─────────────────┐
    │ Fix Errors    │      │llm_judge_evaluate│ ◄── Semantic Validation
    │ Based on      │      └────────┬────────┘
    │ Suggestions   │               │
    └───────────────┘      ┌────────┴────────┐
                           │                  │
                      ┌────▼────┐        ┌────▼────┐
                      │Score<0.7│        │Score≥0.7│
                      └────┬────┘        └────┬────┘
                           │                  │
                           ▼                  ▼
                   ┌───────────────┐  ┌───────────────┐
                   │ Fix Based on  │  │ submit_answer │
                   │ Judge Feedback│  │ (Final)       │
                   └───────────────┘  └───────────────┘
```

---

## LLM Judge Scoring Rubric

| Score | Meaning |
|-------|---------|
| **1.0** | Perfect - fully answers the question with correct logic |
| **0.9** | Correct with minor cosmetic differences |
| **0.8** | Correct approach with minor issues |
| **0.7** | Mostly correct, one significant issue |
| **0.5-0.6** | Partially correct, missing elements |
| **0.3-0.4** | Wrong approach but related to question |
| **0.0-0.2** | Completely wrong or irrelevant |

---

## LLMJudgeOutput Model

```python
class LLMJudgeOutput(BaseModel):
    is_correct: bool           # Binary correctness
    correctness_score: float   # 0.0 to 1.0
    issues: list[str]          # Identified problems
    suggestions: list[str]     # Improvement recommendations
    reasoning: str             # Judge's reasoning explanation
```

---

## Tool Specifications

### `retrieve_tables(question: str, top_k: int = 5)`
Same as ReActAgent V1.

### `validate_sql(query: str)`
Same as ReActAgent V1 (structural validation).

### `llm_judge_evaluate(query: str, question: str)` **NEW**
- Evaluates if query correctly answers the question
- Uses separate Claude instance as judge
- Returns correctness score (0-1) with reasoning
- Provides specific issues and suggestions
- Rate-limited to 3 calls per query

### `submit_answer(query: str, explanation: str, confidence: float)`
Enhanced from V1:
- Combines user confidence with judge scores
- Includes judge feedback in final response
- Weighted average: 40% user, 60% judge

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema_path` | Required | Path to table schema files |
| `retrieval_type` | `"semantic"` | `"semantic"` or `"keyword"` |
| `top_k` | 5 | Number of tables to retrieve |
| `model_id` | `claude-sonnet-4-20250514` | LLM model for main agent |
| `judge_model_id` | `claude-sonnet-4-20250514` | LLM model for judge |
| `max_iterations` | 10 | Maximum tool calls per query |
| `max_retrieval_calls` | 3 | Maximum retrieval operations |
| `max_validation_attempts` | 4 | Maximum validation attempts |
| `max_judge_calls` | 3 | Maximum LLM judge evaluations |

---

## Strengths & Limitations

### Strengths
- Catches semantic errors that structural validation misses
- Two-tier error detection provides comprehensive validation
- Judge feedback guides meaningful improvements
- Separate judge prevents self-evaluation bias
- Combined confidence reflects both structural and semantic quality
- Complete audit trail with judge reasoning

### Limitations
- Higher latency (additional LLM calls for judge)
- More expensive (judge LLM calls add cost)
- Judge may have its own biases
- Rate limits may prevent thorough evaluation
- Complex queries may exceed judge's evaluation capability
- Requires careful prompt engineering for judge consistency

---

## Usage Example

```python
from src.agents.react_agent_v2 import ReActAgentV2

agent = ReActAgentV2(
    schema_path="schemas/",
    retrieval_type="semantic",
    top_k=5,
    max_iterations=10,
    max_judge_calls=3
)

result = agent.run(
    "Find the top 10 users by failed login attempts who have never been blocked"
)
print(result["query"])
print(result["confidence"])      # Combined confidence
print(result["judge_score"])     # Semantic correctness score
print(result["reasoning_steps"]) # Full trace including judge feedback
```

---

## Comparison: V1 vs V2

| Feature | ReActAgent (V1) | ReActAgentV2 |
|---------|-----------------|--------------|
| Structural Validation | Yes | Yes |
| Semantic Validation | No | Yes (LLM Judge) |
| Tools | 3 | 4 |
| State Tracking | AgentState | AgentStateV2 (extended) |
| Confidence Source | Validation only | Validation + Judge |
| Cost | Lower | Higher |
| Latency | Lower | Higher |
| Error Detection | Syntax/Schema | Syntax/Schema + Logic |

---

## Related Components

- [KeywordAgent](./keyword_agent.md) - Simple keyword-based agent
- [SemanticAgent](./semantic_agent.md) - Semantic retrieval agent
- [ReActAgent](./react_agent.md) - Base ReAct agent (V1)
