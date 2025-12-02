#!/usr/bin/env python3
"""Run experiments comparing SQL agents."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from src.agents.base import BaseAgent
from src.agents.registry import get_agent_names, create_agent, get_agent_config
from src.constants import DEFAULT_LLM_MODEL, DEFAULT_TOP_K_TABLES, DEFAULT_EMBEDDING_MODEL
from experiments.judges import BaseJudge, CorrectnessJudge, CategoricalJudge, IntegrityJudge
from experiments.configs import get_experiment_config, list_experiment_types, ExperimentConfig
from experiments.utils.metrics import (
    calculate_retrieval_precision,
    extract_tables_from_sql,
    calculate_aggregate_metrics
)


# Judge registry
JUDGE_REGISTRY = {
    "correctness": CorrectnessJudge,
    "categorical": CategoricalJudge,
    "integrity": IntegrityJudge,
}


def create_judge(judge_type: str, model: str) -> BaseJudge:
    """Create a judge instance of the specified type.

    Args:
        judge_type: Type of judge (correctness, categorical, integrity)
        model: LLM model to use for evaluation

    Returns:
        Judge instance

    Raises:
        ValueError: If judge_type is not recognized
    """
    if judge_type not in JUDGE_REGISTRY:
        valid_types = ", ".join(JUDGE_REGISTRY.keys())
        raise ValueError(f"Unknown judge type: {judge_type}. Valid types: {valid_types}")
    return JUDGE_REGISTRY[judge_type](model=model)


def generate_output_filename(
    model: str,
    agents: List[str],
    judge_identifier: str,
    experiment_type: str
) -> str:
    """Generate standardized output filename.

    Format: {model}_{agents}_{judge_id}_{timestamp}.json

    Args:
        model: LLM model name (will be sanitized)
        agents: List of agent names
        judge_identifier: Judge identifier (e.g., "claude-sonnet-4-5_correctness_v1")
        experiment_type: Type of experiment (main, integrity, etc.)

    Returns:
        Filename string
    """
    # Sanitize model name (remove special chars)
    model_short = model.replace("claude-", "").replace("-", "_")

    # Join agent names
    agents_str = "-".join(sorted(agents))

    # Extract just the judge type from identifier
    judge_type = judge_identifier.split("_")[-2] if "_" in judge_identifier else judge_identifier

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    return f"{model_short}_{agents_str}_{judge_type}_{timestamp}.json"


class ExperimentRunner:
    """Run experiments comparing multiple SQL agents."""

    def __init__(
        self,
        test_cases_path: str,
        agents: Dict[str, BaseAgent],
        agent_configs: Dict[str, Dict[str, Any]] = None,
        judge: BaseJudge = None,
        judge_type: str = "correctness",
        judge_model: str = "claude-sonnet-4-5",
        limit: int = None,
        experiment_type: str = "main"
    ):
        """
        Initialize experiment runner.

        Args:
            test_cases_path: Path to generated test cases JSON
            agents: Dictionary mapping agent names to agent instances
                   e.g., {'keyword': KeywordAgent(...), 'semantic': SemanticAgent(...)}
            agent_configs: Dictionary mapping agent names to their configurations
                   e.g., {'keyword_v1': {'type': 'keyword', 'llm_model': 'claude-sonnet-4-5', ...}}
            judge: Pre-configured judge instance (takes precedence over judge_type/judge_model)
            judge_type: Type of judge to use (correctness, categorical, integrity)
            judge_model: LLM model to use for evaluation
            limit: Optional limit on number of test cases to run (for testing)
            experiment_type: Type of experiment (main, integrity, consistency)
        """
        self.test_cases = self._load_test_cases(test_cases_path)

        # Apply limit if specified
        if limit is not None:
            self.test_cases = self.test_cases[:limit]

        self.agents = agents
        self.agent_configs = agent_configs or {}
        self.experiment_type = experiment_type

        # Create or use provided judge
        if judge is not None:
            self.judge = judge
        else:
            self.judge = create_judge(judge_type, judge_model)

        limit_msg = f" (limited to {limit})" if limit else ""
        print(f"✅ Loaded {len(self.test_cases)} test cases{limit_msg}")
        print(f"✅ Running experiments with {len(agents)} agents: {', '.join(agents.keys())}")
        print(f"✅ Using judge: {self.judge.identifier}\n")

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

        # Evaluate with judge (supports different judge types)
        judge_eval = self.judge.evaluate(
            question=question,
            reference_sql=reference_sql,
            generated_sql=generated_sql or '',
            # Additional context for integrity judge
            expected_behavior=test_case.get('expected_behavior'),
            integrity_type=test_case.get('integrity_type')
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

        # Build result with judge-specific fields
        result = {
            'agent': agent_name,
            'test_case_id': test_case.get('id', 'unknown'),
            'question': question,
            'reference_sql': reference_sql,
            'generated_sql': generated_sql,
            'agent_explanation': response.get('explanation', ''),
            'agent_reasoning_steps': response.get('reasoning_steps', []),
            'latency_ms': latency_ms,
            'total_tokens': total_tokens,
            'retrieval_precision': retrieval_precision,
            'retrieved_tables': retrieved_tables,
            'reference_tables': reference_tables,
            'complexity': test_case.get('complexity'),
            'category': test_case.get('category', 'unknown'),
            'integrity_type': test_case.get('integrity_type'),
            'expected_behavior': test_case.get('expected_behavior'),
            'confidence': response.get('confidence', 0.0),
            'judge_type': self.judge.judge_id,
            'judge_identifier': self.judge.identifier,
        }

        # Add judge-specific evaluation fields
        # Correctness judge: score (0.0-1.0), reasoning, issues
        # Categorical judge: score (1-5), category, reasoning, issues
        # Integrity judge: passed, confidence, reasoning, issues
        result['judge_evaluation'] = judge_eval

        # For backward compatibility and easy access, add common fields
        if 'score' in judge_eval:
            result['correctness_score'] = judge_eval['score']
        elif 'passed' in judge_eval:
            # Convert integrity pass/fail to score for aggregation
            result['correctness_score'] = 1.0 if judge_eval['passed'] else 0.0
        result['correctness_reasoning'] = judge_eval.get('reasoning', '')
        result['correctness_issues'] = judge_eval.get('issues', [])

        return result

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
                'experiment_type': self.experiment_type,
                'agents': self.agent_configs if self.agent_configs else {
                    name: {'type': name} for name in self.agents.keys()
                },
                'total_test_cases': len(self.test_cases),
                'judge': {
                    'type': self.judge.judge_id,
                    'model': self.judge.model_name,
                    'identifier': self.judge.identifier
                }
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
        "--experiment-type",
        default="main",
        choices=list_experiment_types(),
        help="Type of experiment to run (determines test cases and default judge)"
    )
    parser.add_argument(
        "--test-cases",
        default=None,
        help="Path to test cases (overrides experiment type default)"
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
        choices=get_agent_names(),
        help="Which agents to run experiments on"
    )
    parser.add_argument(
        "--judge",
        default=None,
        choices=list(JUDGE_REGISTRY.keys()),
        help="Judge type (overrides experiment type default)"
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tables to retrieve"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N test cases (for testing)"
    )

    args = parser.parse_args()

    # Get experiment configuration
    exp_config = get_experiment_config(args.experiment_type)

    # Resolve test cases path (CLI override or experiment default)
    test_cases_path = args.test_cases or exp_config.test_cases_path

    # Resolve judge type (CLI override or experiment default)
    judge_type = args.judge or exp_config.default_judge

    # Create judge
    judge = create_judge(judge_type, args.judge_model)

    # Generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        output_dir = exp_config.get_output_dir()
        filename = generate_output_filename(
            model=args.judge_model,
            agents=args.agents,
            judge_identifier=judge.identifier,
            experiment_type=args.experiment_type
        )
        output_path = f"{output_dir}/{filename}"

    print("=" * 70)
    print("SQL AGENT EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Experiment Type: {args.experiment_type}")
    print(f"Test Cases: {test_cases_path}")
    print(f"Schema: {args.schema_path}")
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Judge: {judge.identifier}")
    print(f"Top-K Tables: {args.top_k}")
    print(f"Output: {output_path}")
    if args.limit:
        print(f"Limit: {args.limit} test cases")
    print("=" * 70)

    # Dynamically instantiate requested agents using the registry
    agents = {}
    agent_configs = {}

    for agent_name in args.agents:
        print(f"🔄 Initializing {agent_name} agent...")
        agents[agent_name] = create_agent(
            name=agent_name,
            schema_path=args.schema_path,
            top_k_tables=args.top_k,
            judge_model=DEFAULT_LLM_MODEL if agent_name == "react-v2" else None,
        )
        agent_configs[agent_name] = get_agent_config(
            name=agent_name,
            schema_path=args.schema_path,
            top_k=args.top_k,
            llm_model=DEFAULT_LLM_MODEL,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            judge_model=DEFAULT_LLM_MODEL if agent_name == "react-v2" else None,
        )

    print()

    # Run experiments
    runner = ExperimentRunner(
        test_cases_path=test_cases_path,
        agents=agents,
        agent_configs=agent_configs,
        judge=judge,
        limit=args.limit,
        experiment_type=args.experiment_type
    )

    results = runner.run_all_experiments()
    runner.save_results(results, output_path)

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
    print(f"   Results: {output_path}")
    print(f"   Next: python experiments/generate_report.py")


if __name__ == "__main__":
    main()
