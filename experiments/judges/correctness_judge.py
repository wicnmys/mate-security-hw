"""Correctness judge for SQL evaluation (0.0-1.0 scoring)."""

from typing import Dict, Any, List
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

    @classmethod
    def generate_report_sections(cls, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate correctness-specific report sections.

        Args:
            results: List of result dicts with 'correctness_score', 'agent', etc.

        Returns:
            Dict with 'methodology', 'results_table', 'complexity_breakdown',
            'category_breakdown', 'failure_analysis' sections.
        """
        sections = {}

        # Methodology section
        sections['methodology'] = cls._generate_methodology()

        # Get unique agents
        agents = sorted(set(r.get('agent', 'unknown') for r in results))

        # Results table
        sections['results_table'] = cls._generate_results_table(results, agents)

        # Complexity breakdown
        sections['complexity_breakdown'] = cls._generate_complexity_breakdown(results, agents)

        # Category breakdown
        sections['category_breakdown'] = cls._generate_category_breakdown(results, agents)

        # Failure analysis
        sections['failure_analysis'] = cls._generate_failure_analysis(results, agents)

        return sections

    @classmethod
    def _generate_methodology(cls) -> str:
        """Generate methodology section describing the scoring rubric."""
        return """## Correctness Evaluation

### Scoring Rubric (0.0 - 1.0)

| Score | Description |
|-------|-------------|
| 1.0 | Perfectly correct, fully equivalent to reference |
| 0.9 | Correct logic, minor cosmetic differences |
| 0.8 | Correct approach, minor issues |
| 0.7 | Mostly correct, one significant issue |
| 0.5-0.6 | Partially correct |
| 0.3-0.4 | Wrong approach but related tables |
| 0.0-0.2 | Completely wrong |

### Evaluation Criteria
1. **Table Selection**: Does it query the right tables?
2. **Filtering/Conditions**: Does it use the right WHERE clauses?
3. **Columns**: Does it select the right columns?
4. **Aggregations**: Are GROUP BY, HAVING, COUNT, etc. used correctly?
5. **Joins**: Are multi-table joins done correctly?
6. **Ordering/Limiting**: Are ORDER BY and LIMIT used appropriately?
"""

    @classmethod
    def _generate_results_table(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate overall results table."""
        lines = [
            "## Overall Correctness Results\n",
            "| Agent | Avg Score | Min | Max | Count |",
            "|-------|-----------|-----|-----|-------|",
        ]

        for agent in agents:
            agent_results = [r for r in results if r.get('agent') == agent]
            if not agent_results:
                continue

            scores = [r.get('correctness_score', 0) for r in agent_results]
            avg_score = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0

            lines.append(
                f"| {agent.upper()} | {avg_score:.1%} | {min_score:.2f} | {max_score:.2f} | {len(scores)} |"
            )

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def _generate_complexity_breakdown(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate results breakdown by complexity level."""
        lines = ["## Results by Complexity\n"]

        complexities = ['simple', 'medium', 'complex']

        for complexity in complexities:
            complexity_results = [r for r in results if r.get('complexity') == complexity]
            if not complexity_results:
                continue

            lines.append(f"### {complexity.title()} Queries\n")
            lines.append("| Agent | Avg Score | Count |")
            lines.append("|-------|-----------|-------|")

            for agent in agents:
                agent_results = [r for r in complexity_results if r.get('agent') == agent]
                if not agent_results:
                    lines.append(f"| {agent.upper()} | N/A | 0 |")
                    continue

                scores = [r.get('correctness_score', 0) for r in agent_results]
                avg_score = sum(scores) / len(scores) if scores else 0
                lines.append(f"| {agent.upper()} | {avg_score:.1%} | {len(scores)} |")

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _generate_category_breakdown(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate results breakdown by category."""
        lines = ["## Results by Category\n"]

        # Get all unique categories (excluding integrity)
        categories = sorted(set(
            r.get('category', 'unknown')
            for r in results
            if r.get('category') and r.get('category') != 'integrity'
        ))

        for category in categories:
            category_results = [r for r in results if r.get('category') == category]
            if not category_results:
                continue

            lines.append(f"### {category.title()}\n")
            lines.append("| Agent | Avg Score | Count |")
            lines.append("|-------|-----------|-------|")

            for agent in agents:
                agent_results = [r for r in category_results if r.get('agent') == agent]
                if not agent_results:
                    lines.append(f"| {agent.upper()} | N/A | 0 |")
                    continue

                scores = [r.get('correctness_score', 0) for r in agent_results]
                avg_score = sum(scores) / len(scores) if scores else 0
                lines.append(f"| {agent.upper()} | {avg_score:.1%} | {len(scores)} |")

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _generate_failure_analysis(cls, results: List[Dict[str, Any]], agents: List[str]) -> str:
        """Generate failure analysis for low-scoring results."""
        lines = ["## Failure Analysis (Score < 0.5)\n"]

        threshold = 0.5

        for agent in agents:
            agent_failures = [
                r for r in results
                if r.get('agent') == agent and r.get('correctness_score', 1) < threshold
            ]

            lines.append(f"### {agent.upper()} ({len(agent_failures)} failures)\n")

            if not agent_failures:
                lines.append("No significant failures.\n")
                continue

            # Group by issue type
            issue_counts: Dict[str, int] = {}
            for failure in agent_failures:
                for issue in failure.get('correctness_issues', []):
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

            if issue_counts:
                lines.append("**Common Issues:**")
                for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"- {issue} ({count} cases)")
                lines.append("")

            # Show worst cases
            worst_cases = sorted(agent_failures, key=lambda x: x.get('correctness_score', 0))[:3]
            if worst_cases:
                lines.append("**Worst Cases:**")
                for case in worst_cases:
                    question = case.get('question', 'N/A')[:80]
                    score = case.get('correctness_score', 0)
                    lines.append(f"- [{score:.2f}] {question}...")
                lines.append("")

        return "\n".join(lines)
