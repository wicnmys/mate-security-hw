# Experiment 1: ReAct Agent with Retrieval and Validation Tools

## Overview

Build an autonomous agent that uses the ReAct (Reasoning + Acting) pattern to iteratively refine SQL query generation. The agent has control over when to retrieve tables and when to validate queries, enabling self-correction loops.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ReAct Agent Loop                            │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │  Think   │───▶│   Act    │───▶│ Observe  │───▶│  Think   │    │
│   │(Reason)  │    │(Use Tool)│    │(Get Result)   │(Reason)  │    │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    │
│        │                                               │            │
│        └───────────────────────────────────────────────┘            │
│                           (iterate until done)                      │
└─────────────────────────────────────────────────────────────────────┘

Tools Available:
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  retrieve_tables    │  │   validate_sql      │  │   submit_answer     │
│  (semantic/keyword) │  │   (syntax + schema) │  │   (final response)  │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## Tool Definitions

### Tool 1: `retrieve_tables`

Retrieves relevant tables from the schema based on query.

```python
from pydantic import BaseModel, Field
from typing import List

class RetrieveTablesInput(BaseModel):
    """Input for table retrieval tool."""
    query: str = Field(description="Search query to find relevant tables")
    top_k: int = Field(default=5, description="Number of tables to retrieve")
    retrieval_type: str = Field(
        default="semantic",
        description="Type of retrieval: 'semantic' or 'keyword'"
    )

class RetrieveTablesOutput(BaseModel):
    """Output from table retrieval."""
    tables: List[dict] = Field(description="List of retrieved table schemas")
    scores: List[float] = Field(description="Relevance scores for each table")
    total_available: int = Field(description="Total tables in database")

def retrieve_tables(input: RetrieveTablesInput) -> RetrieveTablesOutput:
    """
    Retrieve relevant tables based on natural language query.

    Returns table schemas with relevance scores.
    """
    if input.retrieval_type == "semantic":
        retriever = SemanticRetrieval(schemas)
        results = retriever.get_top_k(input.query, k=input.top_k)
    else:
        retriever = KeywordRetrieval(schemas)
        results = retriever.get_top_k(input.query, k=input.top_k)

    return RetrieveTablesOutput(
        tables=[r['schema'] for r in results],
        scores=[r['score'] for r in results],
        total_available=len(schemas)
    )
```

### Tool 2: `validate_sql`

Validates a SQL query for syntax and schema correctness.

```python
class ValidateSQLInput(BaseModel):
    """Input for SQL validation tool."""
    sql_query: str = Field(description="The SQL query to validate")
    strict: bool = Field(default=False, description="Use strict validation mode")

class ValidationIssue(BaseModel):
    """Single validation issue."""
    type: str = Field(description="Type: 'error' or 'warning'")
    message: str = Field(description="Description of the issue")
    suggestion: str | None = Field(description="How to fix the issue")

class ValidateSQLOutput(BaseModel):
    """Output from SQL validation."""
    valid: bool = Field(description="Whether the query is valid")
    issues: List[ValidationIssue] = Field(description="List of issues found")
    tables_used: List[str] = Field(description="Tables referenced in query")
    fields_used: List[str] = Field(description="Fields referenced in query")
    is_dangerous: bool = Field(description="Contains DROP/DELETE/TRUNCATE")

def validate_sql(input: ValidateSQLInput) -> ValidateSQLOutput:
    """
    Validate SQL query against schema and syntax rules.

    Returns detailed validation results with suggestions.
    """
    validator = SQLValidator(schemas)
    result = validator.validate(input.sql_query, strict=input.strict)

    issues = []
    for error in result['errors']:
        issues.append(ValidationIssue(
            type='error',
            message=error,
            suggestion=_get_suggestion(error)
        ))
    for warning in result['warnings']:
        issues.append(ValidationIssue(
            type='warning',
            message=warning,
            suggestion=_get_suggestion(warning)
        ))

    return ValidateSQLOutput(
        valid=result['valid'],
        issues=issues,
        tables_used=result.get('tables_used', []),
        fields_used=result.get('fields_used', []),
        is_dangerous=result.get('is_dangerous', False)
    )
```

### Tool 3: `submit_answer`

Submit the final SQL query and explanation.

```python
class SubmitAnswerInput(BaseModel):
    """Input for submitting final answer."""
    sql_query: str = Field(description="The final SQL query")
    explanation: str = Field(description="Explanation of what the query does")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning_steps: List[str] = Field(description="Steps taken to arrive at answer")

def submit_answer(input: SubmitAnswerInput) -> str:
    """
    Submit the final answer. This ends the agent loop.
    """
    return f"Answer submitted with confidence {input.confidence}"
```

---

## Agent System Prompt

```python
REACT_AGENT_SYSTEM_PROMPT = """You are an expert SQL query generation agent. Your task is to generate accurate SQL queries from natural language questions.

## Your Tools

You have access to three tools:

1. **retrieve_tables**: Search for relevant database tables
   - Use this to find tables that might be relevant to the question
   - You can adjust top_k to retrieve more or fewer tables
   - You can search again with different queries to refine results

2. **validate_sql**: Validate a SQL query
   - Use this to check if your query is syntactically correct
   - It will tell you about missing tables, invalid fields, and syntax errors
   - Use the suggestions to fix issues

3. **submit_answer**: Submit your final answer
   - Only use this when you are confident in your query
   - Include your reasoning steps and confidence score

## ReAct Process

Follow this reasoning pattern:

**Thought**: What do I need to do next? What have I learned?
**Action**: Choose a tool and provide inputs
**Observation**: Analyze the tool's output
**Thought**: What does this tell me? Do I need to adjust?
... (repeat as needed)
**Action**: submit_answer with final query

## Guidelines

1. **Start with retrieval**: Always retrieve tables first to understand the schema
2. **Generate iteratively**: Create a query, validate it, fix issues
3. **Refine search if needed**: If validation shows missing tables, retrieve more
4. **Be thorough**: Check that all fields exist in the schema
5. **Know when to stop**: Don't loop forever - if you can't find a solution after 3-4 attempts, submit your best effort with low confidence

## Confidence Scoring

- **0.9-1.0**: Query validated successfully, all tables/fields confirmed
- **0.7-0.9**: Query validated with minor warnings, made reasonable assumptions
- **0.5-0.7**: Some validation issues remain, multiple interpretations possible
- **0.3-0.5**: Significant uncertainty, key tables/fields may be wrong
- **0.0-0.3**: Unable to generate valid query, question may be unanswerable

## Example Reasoning Trace

Question: "Show me all high-severity endpoint events from today"

Thought: I need to find tables related to endpoint events. Let me search for relevant tables.
Action: retrieve_tables(query="endpoint events severity", top_k=3)
Observation: Found endpoint_events table with fields: id, timestamp, severity, event_type...

Thought: Good, I found the endpoint_events table. It has severity and timestamp fields. Let me construct a query.
Action: validate_sql(sql_query="SELECT * FROM endpoint_events WHERE severity = 'high' AND timestamp >= CURRENT_DATE")
Observation: Valid query. No issues found. Tables used: [endpoint_events].

Thought: The query validated successfully. I'm confident this is correct.
Action: submit_answer(sql_query="...", confidence=0.95, ...)
"""
```

---

## ReAct Pattern Considerations

### 1. Observation Handling

The agent must properly interpret tool outputs:

```python
def format_observation(tool_name: str, result: Any) -> str:
    """Format tool result as observation for the agent."""
    if tool_name == "retrieve_tables":
        tables_summary = "\n".join([
            f"- {t['table_name']} (score: {s:.2f}): {t.get('description', 'No description')}"
            for t, s in zip(result.tables, result.scores)
        ])
        return f"Retrieved {len(result.tables)} tables:\n{tables_summary}"

    elif tool_name == "validate_sql":
        if result.valid:
            return f"✅ Query is valid. Tables used: {result.tables_used}"
        else:
            issues = "\n".join([f"- [{i.type}] {i.message}" for i in result.issues])
            return f"❌ Validation failed:\n{issues}"

    return str(result)
```

### 2. Loop Control & Stopping Conditions

Prevent infinite loops:

```python
class AgentConfig:
    max_iterations: int = 10          # Hard limit on tool calls
    max_retrieval_calls: int = 3      # Max times to retrieve tables
    max_validation_attempts: int = 4  # Max times to validate same query
    timeout_seconds: int = 60         # Overall timeout

def should_stop(state: AgentState) -> bool:
    """Check if agent should stop iterating."""
    return (
        state.iteration >= config.max_iterations or
        state.has_submitted_answer or
        state.elapsed_time >= config.timeout_seconds or
        state.consecutive_validation_failures >= config.max_validation_attempts
    )
```

### 3. State Management

Track agent state across iterations:

```python
@dataclass
class AgentState:
    question: str
    iteration: int = 0
    retrieved_tables: List[str] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    validation_results: List[dict] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    has_submitted_answer: bool = False
    final_answer: dict | None = None
```

### 4. Error Recovery

Handle tool failures gracefully:

```python
def execute_tool_with_recovery(tool_name: str, input: dict, state: AgentState) -> str:
    """Execute tool with error recovery."""
    try:
        result = tools[tool_name](input)
        return format_observation(tool_name, result)

    except ValidationError as e:
        return f"Tool input error: {e}. Please check your parameters."

    except TimeoutError:
        return "Tool timed out. Try a simpler query or different approach."

    except Exception as e:
        state.reasoning_trace.append(f"Tool error: {e}")
        return f"Tool failed: {e}. Consider trying a different approach."
```

### 5. Thought-Action Parsing

Parse agent output to extract thoughts and actions:

```python
import re

def parse_agent_response(response: str) -> tuple[str, str, dict]:
    """
    Parse agent response to extract thought, action, and parameters.

    Returns: (thought, action_name, action_params)
    """
    # Extract thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    # Extract action
    action_match = re.search(r'Action:\s*(\w+)\((.*?)\)', response, re.DOTALL)
    if action_match:
        action_name = action_match.group(1)
        params_str = action_match.group(2)
        # Parse parameters (simplified)
        action_params = parse_params(params_str)
        return thought, action_name, action_params

    return thought, None, {}
```

---

## Implementation with Agno

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

@tool
def retrieve_tables(query: str, top_k: int = 5, retrieval_type: str = "semantic") -> dict:
    """Retrieve relevant database tables based on search query."""
    # Implementation here
    pass

@tool
def validate_sql(sql_query: str, strict: bool = False) -> dict:
    """Validate SQL query for syntax and schema correctness."""
    # Implementation here
    pass

@tool
def submit_answer(sql_query: str, explanation: str, confidence: float, reasoning_steps: list) -> str:
    """Submit the final SQL query answer. This ends the agent loop."""
    # Implementation here
    pass

# Create agent
react_agent = Agent(
    name="sql_react_agent",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[retrieve_tables, validate_sql, submit_answer],
    instructions=REACT_AGENT_SYSTEM_PROMPT,
    show_tool_calls=True,  # For debugging
    markdown=False
)

# Run agent
result = react_agent.run("Show me all high-severity endpoint events from today")
```

---

## Metrics to Track

| Metric | Description |
|--------|-------------|
| `iterations_to_answer` | How many tool calls before submitting |
| `retrieval_calls` | Number of retrieve_tables calls |
| `validation_attempts` | Number of validate_sql calls |
| `final_validation_passed` | Did the final query pass validation? |
| `self_correction_count` | Times agent fixed query after validation failure |
| `total_latency_ms` | End-to-end time |
| `correctness_score` | LLM-as-judge evaluation (external) |

---

## Expected Behavior Patterns

### Happy Path
```
retrieve_tables → validate_sql (pass) → submit_answer
```

### Self-Correction Path
```
retrieve_tables → validate_sql (fail: missing field) →
retrieve_tables (refined) → validate_sql (pass) → submit_answer
```

### Refinement Path
```
retrieve_tables (top_k=3) → validate_sql (fail: wrong table) →
retrieve_tables (top_k=5, different query) → validate_sql (pass) → submit_answer
```

### Failure Path
```
retrieve_tables → validate_sql (fail) →
validate_sql (fail) → validate_sql (fail) →
submit_answer (low confidence)
```

---

## Comparison with Baseline

| Aspect | Baseline (Single-Pass) | ReAct Agent |
|--------|----------------------|-------------|
| Retrieval | One-shot, fixed top_k | Iterative, adaptive |
| Validation | Post-hoc, non-blocking | In-loop, blocking |
| Self-correction | None | Yes, based on validation |
| Latency | Lower (single LLM call) | Higher (multiple calls) |
| Correctness | Depends on first attempt | Can improve with iterations |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/react_agent.py` | ReAct agent implementation |
| `src/tools/retrieval_tool.py` | Retrieval tool wrapper |
| `src/tools/validation_tool.py` | Validation tool wrapper |
| `tests/test_react_agent.py` | Unit tests for ReAct agent |
| `experiments/run_react_experiment.py` | Experiment runner |

---

## Open Questions

1. **Retrieval strategy**: Should the agent be able to switch between semantic and keyword retrieval?
2. **Validation strictness**: Should strict mode be the default or agent-controlled?
3. **Observation verbosity**: How much detail should tool outputs include?
4. **Early stopping**: Should we force submission after N validation failures?

---

*Created: 2025-12-01*
