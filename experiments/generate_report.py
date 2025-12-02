#!/usr/bin/env python3
"""Generate comparison report from experiment results.

This orchestrator loads result files and delegates section generation
to the appropriate judge classes. Shared sections (executive summary,
agent overview, latency comparison) are generated here, while
judge-specific sections come from each judge's generate_report_sections().
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Type

from experiments.judges.base import BaseJudge
from experiments.judges.correctness_judge import CorrectnessJudge
from experiments.judges.categorical_judge import CategoricalJudge
from experiments.judges.integrity_judge import IntegrityJudge


# Judge registry for looking up classes by type
# NOTE: This is brittle - see docs/development.md for improvement ideas
JUDGE_REGISTRY: Dict[str, Type[BaseJudge]] = {
    'correctness': CorrectnessJudge,
    'categorical': CategoricalJudge,
    'integrity': IntegrityJudge,
}


def get_judge_type(data: Dict[str, Any], file_path: str) -> str:
    """Get judge type from metadata or infer from path.

    Supports both new format (metadata.judge.type) and legacy format
    (inferred from file path).

    Args:
        data: Loaded result JSON data
        file_path: Path to the result file

    Returns:
        Judge type string (e.g., 'correctness_v1')
    """
    # New format: explicit judge type in metadata
    metadata = data.get('metadata', {})
    if 'judge' in metadata and 'type' in metadata['judge']:
        return metadata['judge']['type']

    # Legacy format: infer from file path
    if '/integrity/' in file_path:
        return 'integrity_v1'
    else:
        return 'correctness_v1'  # Default for main experiments


def get_judge_class(judge_type: str) -> Type[BaseJudge]:
    """Get judge class from type string.

    NOTE: This is a brittle lookup that strips version suffix.
    See docs/development.md for improvement ideas.

    Args:
        judge_type: e.g., 'correctness_v1'

    Returns:
        Judge class (e.g., CorrectnessJudge)
    """
    # Strip version suffix (e.g., 'correctness_v1' -> 'correctness')
    base_type = judge_type.replace('_v1', '').replace('_v2', '')
    return JUDGE_REGISTRY.get(base_type, CorrectnessJudge)


class ReportGenerator:
    """Generate markdown report from experiment results."""

    def __init__(self, results_paths: List[str]):
        """
        Initialize report generator from one or more result files.

        Args:
            results_paths: List of paths to experiment results JSON files
        """
        self.results_by_judge: Dict[str, List[Dict[str, Any]]] = {}
        self.agents: List[str] = []
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

        for path in results_paths:
            with open(path, 'r') as f:
                data = json.load(f)

            # Determine judge type for this file
            judge_type = get_judge_type(data, path)

            # Group results by judge type
            results = data.get('results', [])
            if judge_type not in self.results_by_judge:
                self.results_by_judge[judge_type] = []
            self.results_by_judge[judge_type].extend(results)

            # Merge agent metadata
            file_metadata = data.get('metadata', {})
            agents_data = file_metadata.get('agents', {})

            if isinstance(agents_data, list):
                # Old format: just a list of agent names
                for agent_name in agents_data:
                    if agent_name not in self.agents:
                        self.agents.append(agent_name)
                        self.agent_configs[agent_name] = {'type': agent_name}
            else:
                # New format: dict with agent configs
                for agent_name, config in agents_data.items():
                    if agent_name not in self.agents:
                        self.agents.append(agent_name)
                        self.agent_configs[agent_name] = config

            # Merge metadata (use first file's metadata as base)
            if not self.metadata:
                self.metadata = file_metadata

        # Count total results
        self.total_results = sum(len(r) for r in self.results_by_judge.values())

    def _format_percentage(self, value: float) -> str:
        """Format value as percentage."""
        return f"{value * 100:.1f}%"

    def generate_executive_summary(self) -> str:
        """Generate executive summary section (factual, no recommendations)."""
        lines = [
            "## Executive Summary\n",
            f"**Experiment Date:** {self.metadata.get('timestamp', 'N/A')}\n",
            f"**Agents Compared:** {', '.join(self.agents)}\n",
            f"**Total Results:** {self.total_results}\n",
            f"**Judge Types:** {', '.join(sorted(self.results_by_judge.keys()))}\n",
        ]

        return "\n".join(lines)

    def generate_agent_overview(self) -> str:
        """Generate agent configuration overview."""
        lines = [
            "## Agent Configuration\n",
        ]

        for agent_name in self.agents:
            config = self.agent_configs.get(agent_name, {})
            lines.append(f"### {agent_name.upper()}\n")

            if config:
                for key, value in config.items():
                    lines.append(f"- **{key}**: {value}")
            else:
                lines.append("- No configuration data available")

            lines.append("")

        return "\n".join(lines)

    def generate_latency_token_comparison(self) -> str:
        """Generate latency and token usage comparison."""
        lines = [
            "## Performance Metrics\n",
            "| Agent | Avg Latency (ms) | Avg Tokens | Results |",
            "|-------|------------------|------------|---------|",
        ]

        # Aggregate across all judge types
        all_results = []
        for results in self.results_by_judge.values():
            all_results.extend(results)

        for agent_name in self.agents:
            agent_results = [r for r in all_results if r.get('agent') == agent_name]
            if not agent_results:
                continue

            latencies = [r.get('latency_ms', 0) for r in agent_results]
            tokens = [r.get('total_tokens', 0) for r in agent_results]

            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_tokens = sum(tokens) / len(tokens) if tokens else 0

            lines.append(
                f"| {agent_name.upper()} | {avg_latency:.0f} | {avg_tokens:.0f} | {len(agent_results)} |"
            )

        lines.append("")
        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate complete markdown report."""
        sections = [
            "# SQL Agent Comparison Report\n",
        ]

        # Shared sections
        sections.append(self.generate_executive_summary())
        sections.append(self.generate_agent_overview())
        sections.append(self.generate_latency_token_comparison())

        # Judge-specific sections
        for judge_type, results in sorted(self.results_by_judge.items()):
            sections.append(f"\n---\n\n# {judge_type.replace('_', ' ').title()} Evaluation\n")

            # Get judge class and generate its sections
            judge_class = get_judge_class(judge_type)
            judge_sections = judge_class.generate_report_sections(results)

            # Add all sections returned by the judge
            for section_name, content in judge_sections.items():
                sections.append(content)

        # Footer
        sections.append("---")
        sections.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")

        return "\n".join(sections)

    def save_report(self, output_path: str):
        """Save report to markdown file."""
        report = self.generate_full_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Report generated: {output_path}")
        print(f"  Lines: {len(report.splitlines())}")
        print(f"  Size: {len(report)} characters")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comparison report from experiment results"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to experiment results JSON files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for markdown report (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Load results first to get agent names for auto-generated filename
    generator = ReportGenerator(args.results)

    # Auto-generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        agents_str = "_vs_".join(sorted(generator.agents))
        args.output = f"experiments/reports/{agents_str}_{timestamp}.md"

    print("=" * 70)
    print("SQL AGENT REPORT GENERATOR")
    print("=" * 70)
    if len(args.results) == 1:
        print(f"Results: {args.results[0]}")
    else:
        print(f"Results: {len(args.results)} files")
        for path in args.results:
            print(f"  - {path}")
    print(f"Output: {args.output}")
    print(f"Judge types: {', '.join(sorted(generator.results_by_judge.keys()))}")
    print("=" * 70)

    generator.save_report(args.output)

    print("\nReport generation complete!")
    print(f"View report: cat {args.output}")


if __name__ == "__main__":
    main()
