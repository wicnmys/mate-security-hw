#!/usr/bin/env python3
"""Generate comparison report from experiment results."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class ReportGenerator:
    """Generate markdown report from experiment results."""

    def __init__(self, results_path: str):
        """
        Initialize report generator.

        Args:
            results_path: Path to experiment results JSON
        """
        with open(results_path, 'r') as f:
            self.data = json.load(f)

        self.metadata = self.data.get('metadata', {})
        self.results = self.data.get('results', [])
        self.summary = self.data.get('summary', {})
        self.agents = self.metadata.get('agents', [])

    def _format_percentage(self, value: float) -> str:
        """Format value as percentage."""
        return f"{value * 100:.1f}%"

    def _format_metric(self, value: float, decimals: int = 2) -> str:
        """Format numeric metric."""
        return f"{value:.{decimals}f}"

    def generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        lines = [
            "## Executive Summary\n",
            f"**Experiment Date:** {self.metadata.get('timestamp', 'N/A')}\n",
            f"**Agents Compared:** {', '.join(self.agents)}\n",
            f"**Test Cases:** {self.metadata.get('total_test_cases', 0)}\n",
            f"**Evaluation Model:** {self.metadata.get('judge_model', 'N/A')}\n",
        ]

        # Find best agent by correctness
        if self.summary:
            best_agent = max(
                self.summary.items(),
                key=lambda x: x[1]['overall']['avg_correctness']
            )
            best_name = best_agent[0]
            best_score = best_agent[1]['overall']['avg_correctness']

            lines.append(f"\n**Winner:** {best_name.upper()} (Correctness: {self._format_percentage(best_score)})\n")

        return "\n".join(lines)

    def generate_methodology(self) -> str:
        """Generate methodology section."""
        lines = [
            "## Methodology\n",
            "### Test Case Generation",
            "- Generated synthetic test cases using LLM (Claude Sonnet 4.5)",
            "- Three complexity levels: simple (single table), medium (aggregations), complex (joins/subqueries)",
            "- Each test case includes: question, reference SQL, expected tables, semantic intent\n",
            "### Evaluation Metrics",
            "1. **Correctness**: LLM-as-judge semantic evaluation (0.0-1.0 score)",
            "2. **Latency**: End-to-end query generation time (milliseconds)",
            "3. **Token Usage**: Estimated tokens for cost comparison",
            "4. **Retrieval Precision**: % of retrieved tables actually used in SQL\n",
            "### Agent Architectures",
        ]

        for agent in self.agents:
            if agent == "keyword":
                lines.append("- **Keyword Agent**: Token-based keyword matching retrieval")
            elif agent == "semantic":
                lines.append("- **Semantic Agent**: Embedding-based semantic similarity retrieval")
            else:
                lines.append(f"- **{agent.title()} Agent**: Custom retrieval strategy")

        lines.append("")
        return "\n".join(lines)

    def generate_overall_results(self) -> str:
        """Generate overall results table."""
        lines = [
            "## Overall Results\n",
            "| Agent | Correctness | Latency (ms) | Tokens | Retrieval Precision |",
            "|-------|-------------|--------------|--------|---------------------|",
        ]

        for agent_name in self.agents:
            if agent_name not in self.summary:
                continue

            metrics = self.summary[agent_name]['overall']
            row = (
                f"| {agent_name.upper()} | "
                f"{self._format_percentage(metrics['avg_correctness'])} | "
                f"{self._format_metric(metrics['avg_latency_ms'], 0)} | "
                f"{self._format_metric(metrics['avg_total_tokens'], 0)} | "
                f"{self._format_percentage(metrics['avg_retrieval_precision'])} |"
            )
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def generate_complexity_breakdown(self) -> str:
        """Generate results by complexity level."""
        lines = [
            "## Results by Complexity\n",
        ]

        complexities = ['simple', 'medium', 'complex']

        for complexity in complexities:
            lines.append(f"### {complexity.title()} Queries\n")
            lines.append("| Agent | Correctness | Latency (ms) | Tokens | Precision |")
            lines.append("|-------|-------------|--------------|--------|-----------|")

            for agent_name in self.agents:
                if agent_name not in self.summary:
                    continue

                by_complexity = self.summary[agent_name].get('by_complexity', {})
                if complexity not in by_complexity:
                    lines.append(f"| {agent_name.upper()} | N/A | N/A | N/A | N/A |")
                    continue

                metrics = by_complexity[complexity]
                row = (
                    f"| {agent_name.upper()} | "
                    f"{self._format_percentage(metrics['avg_correctness'])} | "
                    f"{self._format_metric(metrics['avg_latency_ms'], 0)} | "
                    f"{self._format_metric(metrics['avg_total_tokens'], 0)} | "
                    f"{self._format_percentage(metrics['avg_retrieval_precision'])} |"
                )
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def generate_category_breakdown(self) -> str:
        """Generate results by category."""
        lines = [
            "## Results by Category\n",
        ]

        # Get all unique categories
        all_categories = set()
        for agent_name in self.agents:
            if agent_name in self.summary:
                all_categories.update(self.summary[agent_name].get('by_category', {}).keys())

        for category in sorted(all_categories):
            lines.append(f"### {category.title()}\n")
            lines.append("| Agent | Correctness | Latency (ms) | Tokens | Precision |")
            lines.append("|-------|-------------|--------------|--------|-----------|")

            for agent_name in self.agents:
                if agent_name not in self.summary:
                    continue

                by_category = self.summary[agent_name].get('by_category', {})
                if category not in by_category:
                    lines.append(f"| {agent_name.upper()} | N/A | N/A | N/A | N/A |")
                    continue

                metrics = by_category[category]
                row = (
                    f"| {agent_name.upper()} | "
                    f"{self._format_percentage(metrics['avg_correctness'])} | "
                    f"{self._format_metric(metrics['avg_latency_ms'], 0)} | "
                    f"{self._format_metric(metrics['avg_total_tokens'], 0)} | "
                    f"{self._format_percentage(metrics['avg_retrieval_precision'])} |"
                )
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def generate_failure_analysis(self) -> str:
        """Analyze cases where agents performed poorly."""
        lines = [
            "## Failure Analysis\n",
        ]

        # Find cases with low correctness scores
        threshold = 0.5
        failures_by_agent = {agent: [] for agent in self.agents}

        for result in self.results:
            if result['correctness_score'] < threshold:
                agent = result['agent']
                if agent in failures_by_agent:
                    failures_by_agent[agent].append(result)

        for agent_name in self.agents:
            failures = failures_by_agent.get(agent_name, [])
            lines.append(f"### {agent_name.upper()} Failures ({len(failures)} cases < {threshold})\n")

            if not failures:
                lines.append("No significant failures.\n")
                continue

            # Group by issue type
            issue_counts = {}
            for failure in failures:
                for issue in failure.get('correctness_issues', []):
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

            if issue_counts:
                lines.append("**Common Issues:**")
                for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"- {issue} ({count} cases)")
                lines.append("")

            # Show worst case examples
            worst_cases = sorted(failures, key=lambda x: x['correctness_score'])[:3]
            if worst_cases:
                lines.append("**Example Failures:**")
                for case in worst_cases:
                    lines.append(f"\n**Q:** {case['question']}")
                    lines.append(f"**Score:** {case['correctness_score']:.2f}")
                    lines.append(f"**Issue:** {case['correctness_reasoning'][:150]}...")
                    lines.append("")

        return "\n".join(lines)

    def generate_insights(self) -> str:
        """Generate key insights and recommendations."""
        lines = [
            "## Key Insights\n",
        ]

        if len(self.agents) < 2:
            lines.append("Single agent tested - no comparative insights available.\n")
            return "\n".join(lines)

        # Compare first two agents (typically keyword vs semantic)
        agent1_name = self.agents[0]
        agent2_name = self.agents[1]

        agent1_metrics = self.summary[agent1_name]['overall']
        agent2_metrics = self.summary[agent2_name]['overall']

        # Correctness comparison
        correctness_diff = agent2_metrics['avg_correctness'] - agent1_metrics['avg_correctness']
        if abs(correctness_diff) > 0.05:
            winner = agent2_name if correctness_diff > 0 else agent1_name
            diff_pct = abs(correctness_diff) * 100
            lines.append(f"1. **{winner.upper()} is more accurate** by {diff_pct:.1f} percentage points")

        # Latency comparison
        latency_diff = agent2_metrics['avg_latency_ms'] - agent1_metrics['avg_latency_ms']
        if abs(latency_diff) > 50:
            faster = agent1_name if latency_diff > 0 else agent2_name
            diff_ms = abs(latency_diff)
            lines.append(f"2. **{faster.upper()} is faster** by {diff_ms:.0f}ms on average")

        # Retrieval precision comparison
        precision_diff = agent2_metrics['avg_retrieval_precision'] - agent1_metrics['avg_retrieval_precision']
        if abs(precision_diff) > 0.05:
            winner = agent2_name if precision_diff > 0 else agent1_name
            diff_pct = abs(precision_diff) * 100
            lines.append(f"3. **{winner.upper()} has better retrieval** by {diff_pct:.1f} percentage points")

        # Complexity analysis
        lines.append("\n### Performance by Complexity")
        for complexity in ['simple', 'medium', 'complex']:
            agent1_complex = self.summary[agent1_name].get('by_complexity', {}).get(complexity, {})
            agent2_complex = self.summary[agent2_name].get('by_complexity', {}).get(complexity, {})

            if agent1_complex and agent2_complex:
                diff = agent2_complex['avg_correctness'] - agent1_complex['avg_correctness']
                if abs(diff) > 0.05:
                    winner = agent2_name if diff > 0 else agent1_name
                    lines.append(f"- **{complexity.title()}:** {winner.upper()} performs better ({abs(diff)*100:.1f}pp advantage)")

        lines.append("")
        return "\n".join(lines)

    def generate_recommendations(self) -> str:
        """Generate recommendations based on results."""
        lines = [
            "## Recommendations\n",
        ]

        if len(self.agents) < 2:
            lines.append("Insufficient agents for comparative recommendations.\n")
            return "\n".join(lines)

        # Find overall best agent
        best_agent = max(
            self.summary.items(),
            key=lambda x: x[1]['overall']['avg_correctness']
        )
        best_name = best_agent[0]

        lines.append(f"### Production Deployment")
        lines.append(f"**Recommended Agent:** {best_name.upper()}\n")

        best_metrics = best_agent[1]['overall']
        lines.append(f"- **Correctness:** {self._format_percentage(best_metrics['avg_correctness'])}")
        lines.append(f"- **Latency:** {self._format_metric(best_metrics['avg_latency_ms'], 0)}ms")
        lines.append(f"- **Retrieval Precision:** {self._format_percentage(best_metrics['avg_retrieval_precision'])}\n")

        lines.append("### Future Improvements")
        lines.append("1. **Hybrid Retrieval**: Combine keyword and semantic approaches")
        lines.append("2. **Query Refinement**: Add iterative self-correction based on validation")
        lines.append("3. **Example-Based Learning**: Few-shot prompting with similar queries")
        lines.append("4. **Contextual Ranking**: Re-rank retrieved tables based on query complexity")
        lines.append("5. **Schema Optimization**: Add table relationship metadata for better JOIN handling\n")

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate complete markdown report."""
        sections = [
            "# SQL Agent Comparison Report\n",
            self.generate_executive_summary(),
            self.generate_methodology(),
            self.generate_overall_results(),
            self.generate_complexity_breakdown(),
            self.generate_category_breakdown(),
            self.generate_failure_analysis(),
            self.generate_insights(),
            self.generate_recommendations(),
            "---",
            f"\n*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*"
        ]

        return "\n".join(sections)

    def save_report(self, output_path: str):
        """Save report to markdown file."""
        report = self.generate_full_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✅ Report generated: {output_path}")
        print(f"   Lines: {len(report.splitlines())}")
        print(f"   Size: {len(report)} characters")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comparison report from experiment results")
    parser.add_argument(
        "--results",
        default="experiments/results/experiment_results.json",
        help="Path to experiment results JSON"
    )
    parser.add_argument(
        "--output",
        default="experiments/comparison.md",
        help="Output path for markdown report"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SQL AGENT REPORT GENERATOR")
    print("=" * 70)
    print(f"Results: {args.results}")
    print(f"Output: {args.output}")
    print("=" * 70)

    generator = ReportGenerator(args.results)
    generator.save_report(args.output)

    print("\n✅ Report generation complete!")
    print(f"   View report: cat {args.output}")


if __name__ == "__main__":
    main()
