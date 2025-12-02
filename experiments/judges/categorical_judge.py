"""Categorical judge for SQL evaluation (1-5 integer scoring)."""

from typing import Dict, Any, List
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

    @classmethod
    def generate_report_sections(cls, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate categorical-specific report sections.

        Args:
            results: List of result dicts with 'score' (1-5), 'category', 'agent', etc.

        Returns:
            Dict with 'methodology', 'results_table', 'distribution',
            'failure_analysis' sections.
        """
        sections = {}

        # Methodology section
        sections['methodology'] = cls._generate_methodology()

        # Get unique agents
        agents = sorted(set(r.get('agent', 'unknown') for r in results))

        # Results table with category distribution
        sections['results_table'] = cls._generate_results_table(results, agents)

        # Score distribution
        sections['distribution'] = cls._generate_distribution(results, agents)

        # Failure analysis (score <= 2)
        sections['failure_analysis'] = cls._generate_failure_analysis(results, agents)

        return sections

    @classmethod
    def _generate_methodology(cls) -> str:
        """Generate methodology section describing the categorical rubric."""
        return """## Categorical Evaluation

### Category Definitions (1-5)

| Score | Category | Description |
|-------|----------|-------------|
| 5 | PERFECT | Semantically equivalent - fully correct, no issues |
| 4 | GOOD | Functionally correct with minor issues |
| 3 | PARTIAL | Some correct but incomplete |
| 2 | POOR | Right tables but wrong logic |
| 1 | WRONG | Wrong tables or syntax errors |

### Evaluation Process
1. Check if tables are correct (wrong tables = 1)
2. Check if basic logic is correct (wrong logic = 2)
3. Check if query is complete (missing parts = 3)
4. Check for minor issues (minor issues = 4)
5. Perfect match = 5
"""

    @classmethod
    def _generate_results_table(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate results table with category distribution."""
        lines = [
            "## Categorical Results\n",
            "| Agent | Avg Score | PERFECT | GOOD | PARTIAL | POOR | WRONG | Count |",
            "|-------|-----------|---------|------|---------|------|-------|-------|",
        ]

        for agent in agents:
            agent_results = [r for r in results if r.get('agent') == agent]
            if not agent_results:
                continue

            # Count by category
            scores = [r.get('score', 1) for r in agent_results]
            avg_score = sum(scores) / len(scores) if scores else 0

            category_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for r in agent_results:
                score = r.get('score', 1)
                if score in category_counts:
                    category_counts[score] += 1

            lines.append(
                f"| {agent.upper()} | {avg_score:.2f} | "
                f"{category_counts[5]} | {category_counts[4]} | {category_counts[3]} | "
                f"{category_counts[2]} | {category_counts[1]} | {len(scores)} |"
            )

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def _generate_distribution(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate score distribution breakdown."""
        lines = ["## Score Distribution\n"]

        for agent in agents:
            agent_results = [r for r in results if r.get('agent') == agent]
            if not agent_results:
                continue

            lines.append(f"### {agent.upper()}\n")

            total = len(agent_results)
            category_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for r in agent_results:
                score = r.get('score', 1)
                if score in category_counts:
                    category_counts[score] += 1

            for score in [5, 4, 3, 2, 1]:
                count = category_counts[score]
                pct = (count / total * 100) if total > 0 else 0
                category = cls.CATEGORIES.get(score, "UNKNOWN")
                bar = "█" * int(pct / 5)  # Visual bar
                lines.append(f"- **{score} ({category})**: {count} ({pct:.1f}%) {bar}")

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _generate_failure_analysis(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate failure analysis for low-scoring results (score <= 2)."""
        lines = ["## Failure Analysis (Score <= 2)\n"]

        for agent in agents:
            agent_failures = [
                r for r in results
                if r.get('agent') == agent and r.get('score', 5) <= 2
            ]

            lines.append(f"### {agent.upper()} ({len(agent_failures)} failures)\n")

            if not agent_failures:
                lines.append("No significant failures.\n")
                continue

            # Count by category
            wrong_count = sum(1 for r in agent_failures if r.get('score') == 1)
            poor_count = sum(1 for r in agent_failures if r.get('score') == 2)

            lines.append(f"- WRONG (1): {wrong_count} cases")
            lines.append(f"- POOR (2): {poor_count} cases\n")

            # Group by issue type
            issue_counts: Dict[str, int] = {}
            for failure in agent_failures:
                for issue in failure.get('issues', []):
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

            if issue_counts:
                lines.append("**Common Issues:**")
                for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"- {issue} ({count} cases)")
                lines.append("")

            # Show worst cases
            worst_cases = sorted(agent_failures, key=lambda x: x.get('score', 5))[:3]
            if worst_cases:
                lines.append("**Worst Cases:**")
                for case in worst_cases:
                    question = case.get('question', 'N/A')[:80]
                    score = case.get('score', 0)
                    category = case.get('category', 'UNKNOWN')
                    lines.append(f"- [{score} {category}] {question}...")
                lines.append("")

        return "\n".join(lines)
