"""LLM-based evaluation utilities for SQL correctness."""

import json
from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field


class CorrectnessEvaluation(BaseModel):
    """Structured correctness evaluation from LLM judge."""
    score: float = Field(description="Correctness score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of the score")
    issues: list[str] = Field(default_factory=list, description="List of issues found")


class LLMJudge:
    """LLM-as-a-judge for SQL correctness evaluation."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize LLM judge.

        Args:
            model: LLM model to use for evaluation
        """
        self.model_name = model
        self.agent = Agent(
            name="sql_correctness_judge",
            model=Claude(id=model),
            instructions=self._get_judge_instructions(),
            output_schema=CorrectnessEvaluation,
            markdown=False
        )

    def _get_judge_instructions(self) -> str:
        """Get instructions for the LLM judge."""
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

    def evaluate_correctness(
        self,
        question: str,
        reference_sql: str,
        generated_sql: str
    ) -> Dict[str, Any]:
        """
        Evaluate if generated SQL correctly answers the question.

        Args:
            question: Natural language question
            reference_sql: Reference SQL query (ground truth)
            generated_sql: Generated SQL query to evaluate

        Returns:
            Dictionary with 'score', 'reasoning', and 'issues'
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
            run_output = self.agent.run(prompt)
            evaluation = run_output.content

            return {
                'score': evaluation.score,
                'reasoning': evaluation.reasoning,
                'issues': evaluation.issues
            }

        except Exception as e:
            # Fallback if LLM call fails
            return {
                'score': 0.0,
                'reasoning': f'Evaluation failed: {str(e)}',
                'issues': ['LLM evaluation error']
            }

    def batch_evaluate(
        self,
        evaluations: list[Dict[str, str]]
    ) -> list[Dict[str, Any]]:
        """
        Evaluate multiple SQL queries.

        Args:
            evaluations: List of dicts with 'question', 'reference_sql', 'generated_sql'

        Returns:
            List of evaluation results
        """
        results = []

        for eval_case in evaluations:
            result = self.evaluate_correctness(
                question=eval_case['question'],
                reference_sql=eval_case['reference_sql'],
                generated_sql=eval_case['generated_sql']
            )
            results.append(result)

        return results
