"""Correctness judge for SQL evaluation (0.0-1.0 scoring)."""

from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field

from experiments.judges.base import BaseJudge


class CorrectnessEvaluation(BaseModel):
    """Structured correctness evaluation from LLM judge."""
    score: float = Field(description="Correctness score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of the score")
    issues: list[str] = Field(default_factory=list, description="List of issues found")


class CorrectnessJudge(BaseJudge):
    """LLM-as-a-judge for SQL correctness evaluation.

    Returns a continuous score between 0.0 and 1.0 indicating
    how well the generated SQL matches the reference.

    Scoring Rubric:
    - 1.0: Perfectly correct, fully equivalent to reference
    - 0.9: Correct logic, minor cosmetic differences
    - 0.8: Correct approach, minor issues
    - 0.7: Mostly correct, one significant issue
    - 0.5-0.6: Partially correct
    - 0.3-0.4: Wrong approach but related tables
    - 0.0-0.2: Completely wrong
    """

    judge_id = "correctness_v1"

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize correctness judge.

        Args:
            model: LLM model to use for evaluation
        """
        self.model_name = model
        self._agent = Agent(
            name="sql_correctness_judge",
            model=Claude(
                id=model,
                cache_system_prompt=True,
                cache_ttl=3600
            ),
            instructions=self._get_instructions(),
            output_schema=CorrectnessEvaluation,
            markdown=False
        )

    def _get_instructions(self) -> str:
        """Get instructions for the correctness judge."""
        return """You are an expert SQL evaluator. Your task is to assess whether a generated SQL query correctly answers a given question by comparing it to a reference SQL query.

**Evaluation Criteria:**
1. **Table Selection**: Does it query the right tables?
2. **Filtering/Conditions**: Does it use the right WHERE clauses?
3. **Columns**: Does it select the right columns?
4. **Aggregations**: Are GROUP BY, HAVING, COUNT, etc. used correctly?
5. **Joins**: Are multi-table joins done correctly?
6. **Ordering/Limiting**: Are ORDER BY and LIMIT used appropriately?

**Important Notes:**
- SQL queries can be semantically equivalent even if syntactically different
- Different JOIN styles (INNER JOIN vs WHERE) may be acceptable
- Alias names don't matter as long as logic is correct
- Column order in SELECT doesn't matter unless ORDER BY is specified
- Extra columns are minor issues if core logic is correct

**Scoring Rubric:**
- **1.0**: Perfectly correct, fully equivalent to reference
- **0.9**: Correct logic, minor cosmetic differences (extra columns, different aliases)
- **0.8**: Correct approach, minor issues (missing ORDER BY, suboptimal but valid)
- **0.7**: Mostly correct, one significant issue (wrong filter value, missing join condition)
- **0.5-0.6**: Partially correct (right tables, wrong aggregation or joins)
- **0.3-0.4**: Wrong approach but related tables (wrong filters or logic)
- **0.0-0.2**: Completely wrong (wrong tables, nonsensical query)

Return a structured evaluation with:
- **score**: Float between 0 and 1
- **reasoning**: Clear explanation of your assessment
- **issues**: List of specific problems found (empty if perfect)

Be strict but fair. Minor syntactic differences should not lower the score if the semantic meaning is the same."""

    def evaluate(
        self,
        question: str,
        reference_sql: str,
        generated_sql: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate if generated SQL correctly answers the question.

        Args:
            question: Natural language question
            reference_sql: Reference SQL query (ground truth)
            generated_sql: Generated SQL query to evaluate
            **kwargs: Additional context (ignored for correctness evaluation)

        Returns:
            Dictionary with 'score' (float 0.0-1.0), 'reasoning', and 'issues'
        """
        if not generated_sql or not generated_sql.strip():
            return {
                'score': 0.0,
                'reasoning': 'Generated SQL is empty',
                'issues': ['No SQL query was generated']
            }

        prompt = f"""Evaluate this SQL query:

**Question**: {question}

**Reference SQL** (ground truth):
```sql
{reference_sql}
```

**Generated SQL** (to evaluate):
```sql
{generated_sql}
```

Compare the generated SQL to the reference SQL and assess whether it correctly answers the question. Provide a score, reasoning, and any issues found."""

        try:
            run_output = self._agent.run(prompt)
            evaluation = run_output.content

            return {
                'score': evaluation.score,
                'reasoning': evaluation.reasoning,
                'issues': evaluation.issues
            }

        except Exception as e:
            return {
                'score': 0.0,
                'reasoning': f'Evaluation failed: {str(e)}',
                'issues': ['LLM evaluation error']
            }
