#!/usr/bin/env python3
"""Run experiments comparing SQL agents."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm

from src.agents.base import BaseAgent
from src.constants import DEFAULT_LLM_MODEL, DEFAULT_TOP_K_TABLES, DEFAULT_EMBEDDING_MODEL
from experiments.utils.llm_judge import LLMJudge
from experiments.utils.metrics import (
    calculate_retrieval_precision,
    extract_tables_from_sql,
    calculate_aggregate_metrics
)


class ExperimentRunner:
    """Run experiments comparing multiple SQL agents."""

    def __init__(
        self,
        test_cases_path: str,
        agents: Dict[str, BaseAgent],
        agent_configs: Dict[str, Dict[str, Any]] = None,
        judge_model: str = "claude-sonnet-4-5"
    ):
        """
        Initialize experiment runner.

        Args:
            test_cases_path: Path to generated test cases JSON
            agents: Dictionary mapping agent names to agent instances
                   e.g., {'keyword': KeywordAgent(...), 'semantic': SemanticAgent(...)}
            agent_configs: Dictionary mapping agent names to their configurations
                   e.g., {'keyword_v1': {'type': 'keyword', 'llm_model': 'claude-sonnet-4-5', ...}}
            judge_model: LLM model to use for correctness evaluation
        """
        self.test_cases = self._load_test_cases(test_cases_path)
        self.agents = agents
        self.agent_configs = agent_configs or {}
        self.judge = LLMJudge(model=judge_model)

        print(f"✅ Loaded {len(self.test_cases)} test cases")
        print(f"✅ Running experiments with {len(agents)} agents: {', '.join(agents.keys())}")
        print(f"✅ Using {judge_model} for correctness evaluation\n")

    def _load_test_cases(self, path: str) -> List[Dict[str, Any]]:
        """Load test cases from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('test_cases', [])

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        More accurate would be to use tiktoken or anthropic's tokenizer,
        but this is a reasonable estimate for comparison purposes.
        """
        # Rough estimate: ~4 characters per token for English text
        return len(text) // 4

    def run_single_test(
        self,
        agent_name: str,
        agent: BaseAgent,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single test case through an agent.

        Args:
            agent_name: Name of the agent being tested
            agent: Agent instance
            test_case: Test case dictionary

        Returns:
            Result dictionary with metrics
        """
        question = test_case['question']
        reference_sql = test_case['reference_sql']
        reference_tables = test_case['reference_tables']

        # Measure latency
        start_time = time.time()
        try:
            response = agent.run(question)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            print(f"  ❌ Error running agent: {e}")
            return {
                'agent': agent_name,
                'test_case_id': test_case.get('id', 'unknown'),
                'question': question,
                'generated_sql': None,
                'agent_explanation': f"Error: {str(e)}",
                'agent_reasoning_steps': [],
                'correctness_score': 0.0,
                'correctness_reasoning': f"Agent error: {str(e)}",
                'correctness_issues': ['Agent execution failed'],
                'latency_ms': 0.0,
                'total_tokens': 0,
                'retrieval_precision': 0.0,
                'complexity': test_case.get('complexity'),
                'category': test_case.get('category', 'unknown'),
                'integrity_type': test_case.get('integrity_type'),
                'expected_behavior': test_case.get('expected_behavior'),
                'confidence': 0.0
            }

        generated_sql = response.get('query', '')

        # Evaluate correctness with LLM judge
        correctness_eval = self.judge.evaluate_correctness(
            question=question,
            reference_sql=reference_sql,
            generated_sql=generated_sql or ''
        )

        # Calculate retrieval precision
        retrieved_tables = response.get('tables_used', [])
        # Also extract tables from generated SQL to be more accurate
        if generated_sql:
            extracted_tables = extract_tables_from_sql(generated_sql)
            if extracted_tables:
                retrieved_tables = extracted_tables

        retrieval_precision = calculate_retrieval_precision(
            retrieved=retrieved_tables,
            used=reference_tables
        )

        # Estimate token usage
        # Input tokens: question + schema context
        # Output tokens: generated SQL + explanation
        input_text = question + str(response.get('explanation', ''))
        output_text = generated_sql or ''
        total_tokens = self._count_tokens(input_text) + self._count_tokens(output_text)

        return {
            'agent': agent_name,
            'test_case_id': test_case.get('id', 'unknown'),
            'question': question,
            'reference_sql': reference_sql,
            'generated_sql': generated_sql,
            'agent_explanation': response.get('explanation', ''),
            'agent_reasoning_steps': response.get('reasoning_steps', []),
            'correctness_score': correctness_eval['score'],
            'correctness_reasoning': correctness_eval['reasoning'],
            'correctness_issues': correctness_eval['issues'],
            'latency_ms': latency_ms,
            'total_tokens': total_tokens,
            'retrieval_precision': retrieval_precision,
            'retrieved_tables': retrieved_tables,
            'reference_tables': reference_tables,
            'complexity': test_case.get('complexity'),
            'category': test_case.get('category', 'unknown'),
            'integrity_type': test_case.get('integrity_type'),
            'expected_behavior': test_case.get('expected_behavior'),
            'confidence': response.get('confidence', 0.0)
        }

    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all test cases through all agents.

        Returns:
            Experiment results with individual test results and aggregate metrics
        """
        all_results = []

        # Calculate total tests across all agents
        total_tests = len(self.test_cases) * len(self.agents)

        # Create overall progress bar
        overall_pbar = tqdm(
            total=total_tests,
            desc="OVERALL PROGRESS",
            position=0,
            unit="test",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            colour='blue',
            leave=True
        )

        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            print(f"\n{'=' * 70}")
            print(f"Running experiments for: {agent_name.upper()}")
            print(f"{'=' * 70}\n")

            agent_results = []

            # Create progress bar for this agent (nested under overall)
            agent_pbar = tqdm(
                self.test_cases,
                desc=f"{agent_name.upper()}",
                position=1,
                unit="test",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                colour='green',
                leave=False
            )

            for test_case in agent_pbar:
                # Update progress bar description with current test info
                # Use integrity_type for integrity tests, complexity for regular tests
                label = test_case.get('integrity_type') or test_case.get('complexity') or '?'
                agent_pbar.set_description(f"{agent_name.upper()} [{label.upper()}]")

                result = self.run_single_test(agent_name, agent, test_case)
                agent_results.append(result)

                # Update progress bar postfix with last result
                score = result['correctness_score']
                score_emoji = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
                agent_pbar.set_postfix({
                    'score': f"{score:.2f}",
                    'latency': f"{result['latency_ms']:.0f}ms",
                    'status': score_emoji
                })

                # Update overall progress
                overall_pbar.update(1)
                overall_pbar.set_postfix({
                    'agent': agent_name,
                    'avg_score': f"{calculate_aggregate_metrics(all_results + agent_results)['avg_correctness']:.2f}" if (all_results or agent_results) else "N/A"
                })

            agent_pbar.close()

            # Calculate aggregate metrics for this agent
            agent_metrics = calculate_aggregate_metrics(agent_results)
            print(f"\n{agent_name.upper()} Summary:")
            print(f"  Avg Correctness: {agent_metrics['avg_correctness']:.3f}")
            print(f"  Avg Latency: {agent_metrics['avg_latency_ms']:.0f}ms")
            print(f"  Avg Tokens: {agent_metrics['avg_total_tokens']:.0f}")
            print(f"  Avg Retrieval Precision: {agent_metrics['avg_retrieval_precision']:.3f}")

            all_results.extend(agent_results)

        # Close overall progress bar
        overall_pbar.close()

        # Build complete experiment report
        experiment_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'agents': self.agent_configs if self.agent_configs else {
                    name: {'type': name} for name in self.agents.keys()
                },
                'total_test_cases': len(self.test_cases),
                'judge_model': self.judge.model_name
            },
            'results': all_results,
            'summary': self._generate_summary(all_results)
        }

        return experiment_results

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics by agent, complexity, and category."""
        summary = {}

        # Group results by agent
        for agent_name in self.agents.keys():
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
                complexity_results = [r for r in agent_results if r['complexity'] == complexity]
                if complexity_results:
                    agent_summary['by_complexity'][complexity] = calculate_aggregate_metrics(complexity_results)

            # By category
            categories = set(r['category'] for r in agent_results)
            for category in categories:
                category_results = [r for r in agent_results if r['category'] == category]
                if category_results:
                    agent_summary['by_category'][category] = calculate_aggregate_metrics(category_results)

            summary[agent_name] = agent_summary

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save experiment results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"✅ Results saved to: {output_path}")
        print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SQL agent experiments")
    parser.add_argument(
        "--test-cases",
        default="experiments/test_cases/generated_test_cases.json",
        help="Path to generated test cases"
    )
    parser.add_argument(
        "--schema-path",
        default="schemas/dataset.json",
        help="Path to schema file"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["keyword", "semantic"],
        choices=["keyword", "semantic", "react", "react_v2"],
        help="Which agents to run experiments on"
    )
    parser.add_argument(
        "--output",
        default="experiments/results/experiment_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5",
        help="LLM model for correctness evaluation"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SQL AGENT EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Test Cases: {args.test_cases}")
    print(f"Schema: {args.schema_path}")
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Top-K Tables: {args.top_k}")
    print("=" * 70)

    # Dynamically import and instantiate requested agents
    agents = {}
    agent_configs = {}

    if "keyword" in args.agents:
        from src.agents.keyword_agent import KeywordAgent
        print("🔄 Initializing KeywordAgent...")
        agents['keyword'] = KeywordAgent(
            schema_path=args.schema_path,
            top_k_tables=args.top_k
        )
        agent_configs['keyword'] = {
            'type': 'keyword',
            'llm_model': DEFAULT_LLM_MODEL,
            'top_k': args.top_k,
            'schema_path': args.schema_path
        }

    if "semantic" in args.agents:
        from src.agents.semantic_agent import SemanticAgent
        print("🔄 Initializing SemanticAgent...")
        agents['semantic'] = SemanticAgent(
            schema_path=args.schema_path,
            top_k_tables=args.top_k
        )
        agent_configs['semantic'] = {
            'type': 'semantic',
            'llm_model': DEFAULT_LLM_MODEL,
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'top_k': args.top_k,
            'schema_path': args.schema_path
        }

    if "react" in args.agents:
        from src.agents.react_agent import ReActAgent
        print("🔄 Initializing ReActAgent...")
        agents['react'] = ReActAgent(
            schema_path=args.schema_path,
            top_k_tables=args.top_k,
            retrieval_type="semantic"
        )
        agent_configs['react'] = {
            'type': 'react',
            'llm_model': DEFAULT_LLM_MODEL,
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'top_k': args.top_k,
            'schema_path': args.schema_path,
            'retrieval_type': 'semantic'
        }

    if "react_v2" in args.agents:
        from src.agents.react_agent_v2 import ReActAgentV2
        print("🔄 Initializing ReActAgentV2 (with LLM judge)...")
        agents['react_v2'] = ReActAgentV2(
            schema_path=args.schema_path,
            top_k_tables=args.top_k,
            retrieval_type="semantic",
            judge_model=DEFAULT_LLM_MODEL
        )
        agent_configs['react_v2'] = {
            'type': 'react_v2',
            'llm_model': DEFAULT_LLM_MODEL,
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'top_k': args.top_k,
            'schema_path': args.schema_path,
            'retrieval_type': 'semantic',
            'judge_model': DEFAULT_LLM_MODEL
        }

    print()

    # Run experiments
    runner = ExperimentRunner(
        test_cases_path=args.test_cases,
        agents=agents,
        agent_configs=agent_configs,
        judge_model=args.judge_model
    )

    results = runner.run_all_experiments()
    runner.save_results(results, args.output)

    # Print final summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 70)
    for agent_name, agent_summary in results['summary'].items():
        overall = agent_summary['overall']
        print(f"\n{agent_name.upper()}:")
        print(f"  Correctness: {overall['avg_correctness']:.3f}")
        print(f"  Latency: {overall['avg_latency_ms']:.0f}ms")
        print(f"  Tokens: {overall['avg_total_tokens']:.0f}")
        print(f"  Retrieval Precision: {overall['avg_retrieval_precision']:.3f}")

    print("\n✅ Experiments complete!")
    print(f"   Results: {args.output}")
    print(f"   Next: python experiments/generate_report.py")


if __name__ == "__main__":
    main()
