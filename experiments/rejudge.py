#!/usr/bin/env python3
"""Re-evaluate existing experiment results with a different judge."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from experiments.judges import BaseJudge, CorrectnessJudge, CategoricalJudge, IntegrityJudge
from experiments.run_experiments import JUDGE_REGISTRY, create_judge, generate_output_filename
from experiments.utils.metrics import calculate_aggregate_metrics


class Rejudger:
    """Re-evaluate experiment results with a different judge."""

    def __init__(
        self,
        results_path: str,
        judge: BaseJudge,
    ):
        """
        Initialize rejudger.

        Args:
            results_path: Path to existing experiment results JSON
            judge: Judge instance to use for re-evaluation
        """
        self.original_results = self._load_results(results_path)
        self.judge = judge
        self.results_path = results_path

        original_judge = self.original_results['metadata'].get('judge', {})
        print(f"✅ Loaded {len(self.original_results['results'])} results from: {results_path}")
        print(f"   Original judge: {original_judge.get('identifier', 'unknown')}")
        print(f"   New judge: {self.judge.identifier}\n")

    def _load_results(self, path: str) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def rejudge_all(self) -> Dict[str, Any]:
        """
        Re-evaluate all results with the new judge.

        Returns:
            New experiment results with updated evaluations
        """
        results = self.original_results['results']
        rejudged_results = []

        print(f"Re-judging {len(results)} results with {self.judge.identifier}...")

        for result in tqdm(results, desc="REJUDGING", unit="test"):
            rejudged = self._rejudge_single(result)
            rejudged_results.append(rejudged)

        # Build new experiment results
        new_results = {
            'metadata': {
                **self.original_results['metadata'],
                'rejudged_at': datetime.now().isoformat(),
                'original_judge': self.original_results['metadata'].get('judge', {}),
                'judge': {
                    'type': self.judge.judge_id,
                    'model': self.judge.model_name,
                    'identifier': self.judge.identifier
                }
            },
            'results': rejudged_results,
            'summary': self._generate_summary(rejudged_results)
        }

        return new_results

    def _rejudge_single(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-evaluate a single result with the new judge.

        Args:
            result: Original result dictionary

        Returns:
            Updated result with new evaluation
        """
        # Call judge with the same inputs
        judge_eval = self.judge.evaluate(
            question=result['question'],
            reference_sql=result.get('reference_sql', ''),
            generated_sql=result.get('generated_sql', ''),
            expected_behavior=result.get('expected_behavior'),
            integrity_type=result.get('integrity_type')
        )

        # Create updated result
        rejudged = {
            **result,
            'judge_type': self.judge.judge_id,
            'judge_identifier': self.judge.identifier,
            'judge_evaluation': judge_eval,
        }

        # Update backward-compatible fields
        if 'score' in judge_eval:
            rejudged['correctness_score'] = judge_eval['score']
        elif 'passed' in judge_eval:
            rejudged['correctness_score'] = 1.0 if judge_eval['passed'] else 0.0
            # Also add passed and confidence at top level for IntegrityJudge report generation
            rejudged['passed'] = judge_eval['passed']
            if 'confidence' in judge_eval:
                rejudged['confidence'] = judge_eval['confidence']

        rejudged['correctness_reasoning'] = judge_eval.get('reasoning', '')
        rejudged['correctness_issues'] = judge_eval.get('issues', [])

        return rejudged

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics by agent, complexity, and category."""
        summary = {}

        # Get unique agents
        agents = set(r['agent'] for r in results)

        for agent_name in agents:
            agent_results = [r for r in results if r['agent'] == agent_name]

            if not agent_results:
                continue

            agent_summary = {
                'overall': calculate_aggregate_metrics(agent_results),
                'by_complexity': {},
                'by_category': {}
            }

            # By complexity
            for complexity in ['simple', 'medium', 'complex']:
                complexity_results = [r for r in agent_results if r.get('complexity') == complexity]
                if complexity_results:
                    agent_summary['by_complexity'][complexity] = calculate_aggregate_metrics(complexity_results)

            # By category
            categories = set(r.get('category', 'unknown') for r in agent_results)
            for category in categories:
                category_results = [r for r in agent_results if r.get('category') == category]
                if category_results:
                    agent_summary['by_category'][category] = calculate_aggregate_metrics(category_results)

            summary[agent_name] = agent_summary

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save rejudged results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"✅ Rejudged results saved to: {output_path}")
        print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing experiment results with a different judge"
    )
    parser.add_argument(
        "results_path",
        help="Path to existing experiment results JSON"
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=list(JUDGE_REGISTRY.keys()),
        help="Judge type to use for re-evaluation"
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5",
        help="LLM model for judge evaluation"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (auto-generated if not specified)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT REJUDGER")
    print("=" * 70)
    print(f"Results: {args.results_path}")
    print(f"Judge: {args.judge}")
    print(f"Judge Model: {args.judge_model}")
    print("=" * 70)

    # Create judge
    judge = create_judge(args.judge, args.judge_model)

    # Generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        # Parse original filename and modify
        original_path = Path(args.results_path)
        # Replace judge type in filename
        stem = original_path.stem
        # Try to extract components from original filename
        parts = stem.split('_')
        if len(parts) >= 4:
            # Format: model_agents_judge_timestamp
            model = parts[0]
            agents = parts[1:-2]  # Everything between model and judge_timestamp
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            new_filename = f"{model}_{'_'.join(agents)}_{args.judge}_{timestamp}.json"
        else:
            # Fallback: append judge type
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            new_filename = f"{stem}_rejudged_{args.judge}_{timestamp}.json"

        output_path = str(original_path.parent / new_filename)

    print(f"Output: {output_path}")
    print()

    # Run rejudging
    rejudger = Rejudger(
        results_path=args.results_path,
        judge=judge
    )

    results = rejudger.rejudge_all()
    rejudger.save_results(results, output_path)

    # Print summary
    print("\n📊 REJUDGING SUMMARY")
    print("=" * 70)
    for agent_name, agent_summary in results['summary'].items():
        overall = agent_summary['overall']
        print(f"\n{agent_name.upper()}:")
        print(f"  Avg Score: {overall['avg_correctness']:.3f}")

    print("\n✅ Rejudging complete!")


if __name__ == "__main__":
    main()
