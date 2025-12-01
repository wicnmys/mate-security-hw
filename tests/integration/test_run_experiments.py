"""Integration tests for experiment runner."""

import pytest
import os
import json
import tempfile
from pathlib import Path

from experiments.run_experiments import ExperimentRunner
from src.agents.keyword_agent import KeywordAgent
from src.agents.semantic_agent import SemanticAgent


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY required for integration tests"
)
class TestExperimentRunner:
    """Integration tests for experiment runner."""

    @pytest.fixture
    def test_cases_file(self):
        """Create a temporary test cases file."""
        test_cases = {
            "metadata": {
                "generated_at": "2025-01-01T00:00:00",
                "total_cases": 2
            },
            "test_cases": [
                {
                    "id": "test_001",
                    "question": "Show all critical endpoint events",
                    "reference_sql": "SELECT * FROM endpoint_events WHERE severity = 'critical'",
                    "reference_tables": ["endpoint_events"],
                    "complexity": "simple",
                    "category": "endpoint"
                },
                {
                    "id": "test_002",
                    "question": "Find failed login attempts",
                    "reference_sql": "SELECT * FROM authentication_events WHERE status = 'failure'",
                    "reference_tables": ["authentication_events"],
                    "complexity": "simple",
                    "category": "authentication"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_cases, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_run_single_test_with_keyword_agent(self, integration_model, test_cases_file):
        """Test running one agent with one query."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        # Create agent
        agent = KeywordAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        # Create runner
        runner = ExperimentRunner(
            test_cases_path=test_cases_file,
            agents={'keyword': agent},
            agent_configs={
                'keyword': {
                    'type': 'keyword',
                    'llm_model': integration_model,
                    'top_k': 5
                }
            },
            judge_model=integration_model
        )

        # Run single test
        test_case = runner.test_cases[0]
        result = runner.run_single_test('keyword', agent, test_case)

        # Assertions
        assert result is not None
        assert result['agent'] == 'keyword'
        assert result['test_case_id'] == 'test_001'
        assert 'generated_sql' in result
        assert 'correctness_score' in result
        assert 0.0 <= result['correctness_score'] <= 1.0
        assert result['complexity'] == 'simple'
        assert result['category'] == 'endpoint'

    def test_run_experiment_multiple_queries(self, integration_model, test_cases_file):
        """Test running one agent with multiple queries."""
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = KeywordAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        runner = ExperimentRunner(
            test_cases_path=test_cases_file,
            agents={'keyword': agent},
            agent_configs={
                'keyword': {
                    'type': 'keyword',
                    'llm_model': integration_model,
                    'top_k': 5
                }
            },
            judge_model=integration_model
        )

        results = runner.run_all_experiments()

        # Check metadata
        assert 'metadata' in results
        assert 'agents' in results['metadata']
        assert 'keyword' in results['metadata']['agents']
        assert results['metadata']['total_test_cases'] == 2

        # Check results
        assert 'results' in results
        assert len(results['results']) == 2

        # Check summary
        assert 'summary' in results
        assert 'keyword' in results['summary']
        assert 'overall' in results['summary']['keyword']

    def test_run_experiment_multiple_agents(self, integration_model, test_cases_file):
        """Test running 2 agents with 2 queries each."""
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agents = {
            'keyword': KeywordAgent(
                schema_path="schemas/dataset.json",
                model=integration_model,
                top_k_tables=5
            ),
            'semantic': SemanticAgent(
                schema_path="schemas/dataset.json",
                model=integration_model,
                top_k_tables=5
            )
        }

        agent_configs = {
            'keyword': {
                'type': 'keyword',
                'llm_model': integration_model,
                'top_k': 5
            },
            'semantic': {
                'type': 'semantic',
                'llm_model': integration_model,
                'embedding_model': 'multi-qa-mpnet-base-dot-v1',
                'top_k': 5
            }
        }

        runner = ExperimentRunner(
            test_cases_path=test_cases_file,
            agents=agents,
            agent_configs=agent_configs,
            judge_model=integration_model
        )

        results = runner.run_all_experiments()

        # Check we have results for both agents
        assert len(results['results']) == 4  # 2 agents x 2 queries
        keyword_results = [r for r in results['results'] if r['agent'] == 'keyword']
        semantic_results = [r for r in results['results'] if r['agent'] == 'semantic']
        assert len(keyword_results) == 2
        assert len(semantic_results) == 2

        # Check both agents in summary
        assert 'keyword' in results['summary']
        assert 'semantic' in results['summary']

    def test_save_results(self, integration_model, test_cases_file):
        """Test saving experiment results to file."""
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = KeywordAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        runner = ExperimentRunner(
            test_cases_path=test_cases_file,
            agents={'keyword': agent},
            judge_model=integration_model
        )

        results = runner.run_all_experiments()

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            runner.save_results(results, output_path)

            # Verify file was created and is valid JSON
            assert Path(output_path).exists()

            with open(output_path, 'r') as f:
                loaded_results = json.load(f)

            assert 'metadata' in loaded_results
            assert 'results' in loaded_results
            assert 'summary' in loaded_results
        finally:
            Path(output_path).unlink(missing_ok=True)
