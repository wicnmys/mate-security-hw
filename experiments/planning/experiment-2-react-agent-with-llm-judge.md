# Experiment 2: ReAct Agent with LLM-as-Judge Validation

## Overview

Extends Experiment 1 by adding a second validation mechanism: an LLM-as-Judge that semantically evaluates whether the generated SQL correctly answers the original question. This creates a two-tier validation system:

1. **Structural Validation** (Tool): Syntax, schema, field existence
2. **Semantic Validation** (LLM Judge): Does the query actually answer the question?

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ReAct Agent with Dual Validation                      │
│                                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Think   │───▶│   Act    │───▶│ Observe  │───▶│  Think   │         │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                         │                                                │
│                         ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Available Tools                             │   │
│   │                                                                  │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│   │  │   retrieve   │  │   validate   │  │  llm_judge   │          │   │
│   │  │   _tables    │  │    _sql      │  │  _evaluate   │          │   │
│   │  │  (semantic)  │  │  (syntax)    │  │  (semantic)  │          │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│   │                                                                  │   │
│   │                    ┌──────────────┐                             │   │
│   │                    │   submit     │                             │   │
│   │                    │   _answer    │                             │   │
│   │                    └──────────────┘                             │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## New Tool: `llm_judge_evaluate`

### Purpose

The LLM Judge answers: **"Does this SQL query correctly answer the original question?"**

Unlike structural validation which checks syntax, the judge:
- Evaluates semantic correctness
- Identifies logical errors (wrong filters, missing conditions)
- Catches misunderstandings of the question
- Provides actionable feedback for improvement

### Implementation

```python
class LLMJudgeInput(BaseModel):
    """Input for LLM-as-judge evaluation."""
    original_question: str = Field(description="The user's original question")
    sql_query: str = Field(description="The generated SQL query")
    table_schemas: List[dict] = Field(description="Schemas of tables used in query")
    agent_explanation: str = Field(description="Agent's explanation of the query")

class LLMJudgeOutput(BaseModel):
    """Output from LLM-as-judge evaluation."""
    is_correct: bool = Field(description="Whether query correctly answers question")
    correctness_score: float = Field(description="Score from 0.0 to 1.0")
    issues: List[str] = Field(description="List of semantic issues found")
    suggestions: List[str] = Field(description="Suggestions for improvement")
    reasoning: str = Field(description="Judge's reasoning")

@tool
def llm_judge_evaluate(
    original_question: str,
    sql_query: str,
    table_schemas: list,
    agent_explanation: str
) -> dict:
    """
    Evaluate if the SQL query correctly answers the original question.

    Uses a separate LLM call to act as an impartial judge.
    Returns semantic correctness assessment with detailed feedback.
    """
    judge_prompt = f"""You are an expert SQL evaluator. Assess whether this SQL query correctly answers the question.

## Original Question
{original_question}

## Generated SQL Query
```sql
{sql_query}
```

## Agent's Explanation
{agent_explanation}

## Available Table Schemas
{json.dumps(table_schemas, indent=2)}

## Your Task
Evaluate the query and provide:

1. **is_correct**: Does the query correctly answer the question? (true/false)
2. **correctness_score**: How correct is it? (0.0 to 1.0)
   - 1.0: Perfect, answers exactly what was asked
   - 0.8-0.9: Correct with minor differences (e.g., extra ORDER BY)
   - 0.5-0.7: Partially correct, missing some aspects
   - 0.2-0.4: Significant issues, wrong interpretation
   - 0.0-0.1: Completely wrong or doesn't answer the question
3. **issues**: List any semantic problems (wrong filters, missing conditions, wrong table)
4. **suggestions**: Specific improvements to fix the issues
5. **reasoning**: Explain your evaluation

Be strict but fair. Focus on whether the query would return the data the user actually wants."""

    # Use a smaller/faster model for judging
    judge_agent = Agent(
        name="sql_judge",
        model=Claude(id="claude-haiku-3"),  # Faster for iteration
        output_schema=LLMJudgeOutput,
        markdown=False
    )

    result = judge_agent.run(judge_prompt)
    return result.content.model_dump()
```

---

## Updated Agent System Prompt

```python
REACT_AGENT_WITH_JUDGE_PROMPT = """You are an expert SQL query generation agent with self-evaluation capabilities.

## Your Tools

You have access to four tools:

1. **retrieve_tables**: Search for relevant database tables
   - Use this to find tables relevant to the question
   - Can refine search with different queries

2. **validate_sql**: Validate SQL syntax and schema
   - Checks if query is syntactically correct
   - Verifies tables and fields exist
   - Fast, rule-based validation

3. **llm_judge_evaluate**: Semantic correctness evaluation (NEW)
   - Evaluates if your query actually answers the question
   - Catches logical errors and misunderstandings
   - Provides suggestions for improvement
   - Use this AFTER structural validation passes

4. **submit_answer**: Submit your final answer
   - Only use when confident in your query
   - Ideally after both validations pass

## Dual Validation Strategy

Always validate in two stages:

```
Stage 1: Structural Validation (validate_sql)
├── Syntax correct?
├── Tables exist?
├── Fields exist?
└── No dangerous operations?

Stage 2: Semantic Validation (llm_judge_evaluate)
├── Answers the question?
├── Correct filters/conditions?
├── Right interpretation?
└── Complete solution?
```

## When to Use Each Validator

| Situation | Use validate_sql | Use llm_judge_evaluate |
|-----------|------------------|------------------------|
| First query attempt | ✅ Always first | ✅ If structural passes |
| After fixing syntax | ✅ Yes | Only if it passes |
| Unsure about logic | Not needed | ✅ Yes |
| Final check before submit | ✅ Quick sanity check | ✅ For confidence |

## ReAct Process with Dual Validation

**Thought**: Analyze the question, plan approach
**Action**: retrieve_tables(...)
**Observation**: Found relevant tables

**Thought**: Construct initial query
**Action**: validate_sql(sql_query="...")
**Observation**: ✅ Syntax valid

**Thought**: Syntax is good, but does it answer the question correctly?
**Action**: llm_judge_evaluate(question="...", sql_query="...", ...)
**Observation**: Score 0.6 - Missing time filter

**Thought**: Need to add time filter. Let me fix and re-validate.
**Action**: validate_sql(sql_query="...updated...")
**Observation**: ✅ Valid

**Action**: llm_judge_evaluate(...)
**Observation**: Score 0.95 - Correct!

**Thought**: Both validations pass with high score. Ready to submit.
**Action**: submit_answer(...)

## Confidence Scoring with Dual Validation

Your confidence should reflect BOTH validation results:

| Structural | Semantic (Judge) | Confidence |
|------------|------------------|------------|
| ✅ Pass | Score >= 0.9 | 0.9 - 1.0 |
| ✅ Pass | Score 0.7-0.9 | 0.7 - 0.9 |
| ✅ Pass | Score 0.5-0.7 | 0.5 - 0.7 |
| ✅ Pass | Score < 0.5 | 0.3 - 0.5 |
| ❌ Fail | Any | 0.0 - 0.3 |

## Loop Limits

- Maximum iterations: 10
- Maximum judge calls: 3 (expensive)
- If you can't get judge score > 0.7 after 3 attempts, submit best effort
"""
```

---

## Key Differences from Experiment 1

| Aspect | Experiment 1 | Experiment 2 |
|--------|--------------|--------------|
| Validation | Structural only | Structural + Semantic |
| Self-correction | Based on syntax errors | Based on logical errors too |
| Feedback quality | "Field X doesn't exist" | "Query misses time filter the user asked for" |
| Cost | Lower (rule-based validation) | Higher (extra LLM call for judge) |
| Latency | Lower | Higher |
| Correctness | Good for syntax | Better for semantic accuracy |

---

## ReAct Considerations for LLM Judge

### 1. Judge Model Selection

```python
# Options for judge model
JUDGE_MODEL_OPTIONS = {
    "fast": "claude-haiku-3",      # ~100ms, cheaper, good for iteration
    "balanced": "claude-sonnet-4-5", # ~500ms, more accurate
    "accurate": "claude-opus-4-5",  # ~2s, most accurate, expensive
}

# Recommendation: Use faster model during iteration, accurate for final check
def get_judge_model(is_final_check: bool) -> str:
    return "claude-sonnet-4-5" if is_final_check else "claude-haiku-3"
```

### 2. Preventing Infinite Judge Loops

The agent might keep calling the judge hoping for a better score:

```python
class JudgeLoopPrevention:
    """Prevent infinite loops with LLM judge."""

    def __init__(self, max_judge_calls: int = 3):
        self.max_calls = max_judge_calls
        self.call_count = 0
        self.previous_scores = []

    def can_call_judge(self) -> bool:
        return self.call_count < self.max_calls

    def record_call(self, score: float):
        self.call_count += 1
        self.previous_scores.append(score)

    def is_improving(self) -> bool:
        """Check if scores are improving."""
        if len(self.previous_scores) < 2:
            return True
        return self.previous_scores[-1] > self.previous_scores[-2]

    def should_stop_trying(self) -> bool:
        """Stop if not improving after 2 attempts."""
        if len(self.previous_scores) >= 2:
            # Not improving and already tried twice
            if not self.is_improving():
                return True
        return self.call_count >= self.max_calls
```

### 3. Cost Management

```python
class CostTracker:
    """Track costs of agent execution."""

    # Approximate costs per 1K tokens (as of 2024)
    COSTS = {
        "claude-haiku-3": {"input": 0.00025, "output": 0.00125},
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    }

    def __init__(self):
        self.total_cost = 0.0
        self.calls_by_model = {}

    def add_call(self, model: str, input_tokens: int, output_tokens: int):
        cost = (
            self.COSTS[model]["input"] * input_tokens / 1000 +
            self.COSTS[model]["output"] * output_tokens / 1000
        )
        self.total_cost += cost
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

    def get_summary(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "calls_by_model": self.calls_by_model
        }
```

### 4. Handling Judge Disagreement

Sometimes structural validation passes but judge gives low score:

```python
def handle_validation_disagreement(
    structural_result: ValidateSQLOutput,
    judge_result: LLMJudgeOutput,
    state: AgentState
) -> str:
    """
    Handle cases where structural passes but semantic fails.

    Returns guidance for the agent.
    """
    if structural_result.valid and judge_result.correctness_score < 0.5:
        # Structural OK but semantic bad - likely wrong interpretation
        return f"""
        Your query is syntactically correct but doesn't answer the question properly.

        Judge feedback:
        - Score: {judge_result.correctness_score}
        - Issues: {', '.join(judge_result.issues)}
        - Suggestions: {', '.join(judge_result.suggestions)}

        Consider:
        1. Re-reading the original question
        2. Checking if you're using the right filters
        3. Verifying you selected the correct table
        """

    elif not structural_result.valid:
        # Structural failed - fix that first
        return "Fix structural issues before semantic evaluation."

    else:
        # Both pass - good to go
        return "Both validations passed. Consider submitting."
```

### 5. Observation Formatting for Judge

```python
def format_judge_observation(result: LLMJudgeOutput) -> str:
    """Format judge result as clear observation."""

    status = "✅ CORRECT" if result.is_correct else "❌ NEEDS IMPROVEMENT"
    score_bar = "█" * int(result.correctness_score * 10) + "░" * (10 - int(result.correctness_score * 10))

    observation = f"""
## Semantic Evaluation: {status}
Score: [{score_bar}] {result.correctness_score:.1%}

### Issues Found
{chr(10).join(f"- {issue}" for issue in result.issues) or "None"}

### Suggestions
{chr(10).join(f"- {s}" for s in result.suggestions) or "Query looks good!"}

### Judge Reasoning
{result.reasoning}
"""
    return observation.strip()
```

---

## Implementation

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

# Import tools from Experiment 1
from experiments.exp1.tools import retrieve_tables, validate_sql, submit_answer

@tool
def llm_judge_evaluate(
    original_question: str,
    sql_query: str,
    table_schemas: list,
    agent_explanation: str
) -> dict:
    """
    Evaluate if SQL query correctly answers the question using LLM-as-judge.

    This provides semantic validation beyond syntax checking.
    Use after structural validation passes.

    Returns:
        - is_correct: Boolean indicating if query answers the question
        - correctness_score: Float 0.0-1.0
        - issues: List of semantic issues found
        - suggestions: List of improvements
        - reasoning: Judge's explanation
    """
    # Implementation as shown above
    pass

# Create agent with dual validation
react_agent_with_judge = Agent(
    name="sql_react_agent_v2",
    model=Claude(id="claude-sonnet-4-5"),
    tools=[retrieve_tables, validate_sql, llm_judge_evaluate, submit_answer],
    instructions=REACT_AGENT_WITH_JUDGE_PROMPT,
    show_tool_calls=True,
    markdown=False
)
```

---

## Expected Behavior Patterns

### Happy Path (Both Pass)
```
retrieve_tables → validate_sql (✅) → llm_judge_evaluate (0.95) → submit_answer
```

### Semantic Correction Path
```
retrieve_tables → validate_sql (✅) → llm_judge_evaluate (0.4: "wrong filter") →
[Agent fixes filter] → validate_sql (✅) → llm_judge_evaluate (0.9) → submit_answer
```

### Structural Then Semantic Path
```
retrieve_tables → validate_sql (❌: field missing) →
[Agent fixes field] → validate_sql (✅) → llm_judge_evaluate (0.85) → submit_answer
```

### Judge Limit Reached
```
retrieve_tables → validate_sql (✅) → llm_judge_evaluate (0.5) →
[Agent tries fix] → validate_sql (✅) → llm_judge_evaluate (0.55) →
[Agent tries again] → validate_sql (✅) → llm_judge_evaluate (0.6) →
submit_answer (confidence: 0.6, note: "reached judge limit")
```

---

## Metrics to Track

| Metric | Description |
|--------|-------------|
| `structural_validation_calls` | Number of validate_sql calls |
| `judge_calls` | Number of llm_judge_evaluate calls |
| `final_judge_score` | Last judge score before submission |
| `judge_improvement` | Score delta from first to last judge call |
| `semantic_corrections` | Times agent fixed logic after judge feedback |
| `total_cost` | Combined cost of all LLM calls |
| `correctness_score` | External evaluation (ground truth) |

---

## Comparison: Experiment 1 vs 2

| Metric | Exp 1 (Structural Only) | Exp 2 (+ Judge) |
|--------|------------------------|-----------------|
| Latency | Lower | Higher (+1-3 LLM calls) |
| Cost | Lower | Higher (~2-3x for judge) |
| Syntax errors caught | ✅ Yes | ✅ Yes |
| Logic errors caught | ❌ No | ✅ Yes |
| Self-correction quality | Syntax-based | Semantic-based |
| Confidence calibration | Based on validation | Based on judge score |

---

## Hypothesis

We expect Experiment 2 to:
1. Have **higher correctness** (catches semantic errors)
2. Have **higher latency** (more LLM calls)
3. Have **higher cost** (judge calls)
4. Show **better self-correction** on logic errors
5. Have **better calibrated confidence** (judge score correlates with correctness)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/react_agent_v2.py` | ReAct agent with judge |
| `src/tools/llm_judge_tool.py` | LLM judge tool |
| `tests/test_react_agent_v2.py` | Unit tests |
| `experiments/run_react_v2_experiment.py` | Experiment runner |

---

## Open Questions

1. **Judge model**: Use same model as agent or different? (Haiku vs Sonnet)
2. **Judge prompt**: How detailed should schemas be in judge context?
3. **When to judge**: After every query attempt or only when structural passes?
4. **Score threshold**: What judge score is "good enough" to submit?
5. **Cost vs accuracy tradeoff**: Is the extra cost worth the accuracy gain?

---

*Created: 2025-12-01*
