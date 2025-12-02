"""Categorical judge for SQL evaluation (1-5 integer scoring)."""

from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field

from experiments.judges.base import BaseJudge


class CategoricalEvaluation(BaseModel):
    """Structured categorical evaluation from LLM judge."""
    score: int = Field(description="Categorical score from 1 to 5", ge=1, le=5)
    category: str = Field(description="Category name: WRONG, POOR, PARTIAL, GOOD, or PERFECT")
    reasoning: str = Field(description="Explanation of the categorization")
    issues: list[str] = Field(default_factory=list, description="List of issues found")


class CategoricalJudge(BaseJudge):
    """LLM-as-a-judge for SQL categorical evaluation.

    Returns an integer score from 1-5 with SQL-specific meaning:

    Category Definitions:
    - 1 (WRONG): Wrong tables or syntax errors - query won't run or queries completely wrong data
    - 2 (POOR): Right tables but wrong logic - correct tables but joins, filters, or aggregations are incorrect
    - 3 (PARTIAL): Some correct but incomplete - partially answers the question, missing key components
    - 4 (GOOD): Functionally correct with minor issues - right answer, minor cosmetic issues
    - 5 (PERFECT): Semantically equivalent - fully correct, no issues
    """

    judge_id = "categorical_v1"

    # Category name mappings
    CATEGORIES = {
        1: "WRONG",
        2: "POOR",
        3: "PARTIAL",
        4: "GOOD",
        5: "PERFECT"
    }

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize categorical judge.

        Args:
            model: LLM model to use for evaluation
        """
        self.model_name = model
        self._agent = Agent(
            name="sql_categorical_judge",
            model=Claude(
                id=model,
                cache_system_prompt=True,
                cache_ttl=3600
            ),
            instructions=self._get_instructions(),
            output_schema=CategoricalEvaluation,
            markdown=False
        )

    def _get_instructions(self) -> str:
        """Get instructions for the categorical judge."""
        return """You are an expert SQL evaluator. Your task is to categorize a generated SQL query into one of five quality categories by comparing it to a reference SQL query.

**Category Definitions:**

**1 - WRONG**: Wrong tables or syntax errors
- Query uses completely incorrect tables
- Query has syntax errors that would prevent execution
- Query is nonsensical or unrelated to the question

**2 - POOR**: Right tables but wrong logic
- Uses correct tables but joins are wrong
- Filters/WHERE clauses have incorrect logic
- Aggregations are fundamentally wrong
- Would return wrong data from correct sources

**3 - PARTIAL**: Some correct but incomplete
- Partially answers the question
- Missing key columns, joins, or filters
- Returns correct but incomplete results
- Some logic is right, some is wrong

**4 - GOOD**: Functionally correct with minor issues
- Query would return the correct answer
- Minor cosmetic differences (aliases, column order)
- Slightly suboptimal but valid approach
- Missing non-essential ORDER BY or LIMIT

**5 - PERFECT**: Semantically equivalent
- Fully correct, answers the question completely
- Equivalent logic to reference (even if syntax differs)
- No issues or improvements needed

**Evaluation Process:**
1. Check if tables are correct (wrong tables = score 1)
2. Check if basic logic is correct (wrong logic = score 2)
3. Check if query is complete (missing parts = score 3)
4. Check for minor issues (minor issues = score 4)
5. Perfect match = score 5

Return a structured evaluation with:
- **score**: Integer from 1 to 5
- **category**: The category name (WRONG, POOR, PARTIAL, GOOD, or PERFECT)
- **reasoning**: Clear explanation of your categorization
- **issues**: List of specific problems found (empty if perfect)"""

    def evaluate(
        self,
        question: str,
        reference_sql: str,
        generated_sql: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate and categorize the generated SQL query.

        Args:
            question: Natural language question
            reference_sql: Reference SQL query (ground truth)
            generated_sql: Generated SQL query to evaluate
            **kwargs: Additional context (ignored for categorical evaluation)

        Returns:
            Dictionary with 'score' (int 1-5), 'category', 'reasoning', and 'issues'
        """
        if not generated_sql or not generated_sql.strip():
            return {
                'score': 1,
                'category': 'WRONG',
                'reasoning': 'Generated SQL is empty',
                'issues': ['No SQL query was generated']
            }

        prompt = f"""Categorize this SQL query:

**Question**: {question}

**Reference SQL** (ground truth):
```sql
{reference_sql}
```

**Generated SQL** (to evaluate):
```sql
{generated_sql}
```

Categorize the generated SQL into one of the five quality categories (1-WRONG to 5-PERFECT). Provide the score, category name, reasoning, and any issues found."""

        try:
            run_output = self._agent.run(prompt)
            evaluation = run_output.content

            return {
                'score': evaluation.score,
                'category': evaluation.category,
                'reasoning': evaluation.reasoning,
                'issues': evaluation.issues
            }

        except Exception as e:
            return {
                'score': 1,
                'category': 'WRONG',
                'reasoning': f'Evaluation failed: {str(e)}',
                'issues': ['LLM evaluation error']
            }

    @classmethod
    def score_to_category(cls, score: int) -> str:
        """Convert a score to its category name."""
        return cls.CATEGORIES.get(score, "UNKNOWN")

    @classmethod
    def category_to_score(cls, category: str) -> int:
        """Convert a category name to its score."""
        reverse_map = {v: k for k, v in cls.CATEGORIES.items()}
        return reverse_map.get(category.upper(), 1)
