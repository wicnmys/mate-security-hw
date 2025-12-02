"""Unit tests for rejudge module."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.rejudge import Rejudger


class TestRejudger:
    """Tests for Rejudger class."""

    @pytest.fixture
    def sample_results(self, tmp_path):
        """Create sample results file for testing."""
        results = {
            'metadata': {
                'timestamp': '2025-01-01T00:00:00',
                'experiment_type': 'main',
                'judge': {
                    'type': 'correctness_v1',
                    'model': 'claude-sonnet-4-5',
                    'identifier': 'claude-sonnet-4-5_correctness_v1'
                }
            },
            'results': [
                {
                    'agent': 'keyword',
                    'test_case_id': 'test_1',
                    'question': 'Show all users',
                    'reference_sql': 'SELECT * FROM users',
                    'generated_sql': 'SELECT * FROM users',
                    'complexity': 'simple',
                    'category': 'authentication',
                    'correctness_score': 1.0,
                    'correctness_reasoning': 'Perfect match',
                    'correctness_issues': []
                },
                {
                    'agent': 'keyword',
                    'test_case_id': 'test_2',
                    'question': 'Count events',
                    'reference_sql': 'SELECT COUNT(*) FROM events',
                    'generated_sql': 'SELECT * FROM events',
                    'complexity': 'medium',
                    'category': 'analytics',
                    'correctness_score': 0.5,
                    'correctness_reasoning': 'Missing aggregation',
                    'correctness_issues': ['Missing COUNT']
                }
            ],
            'summary': {}
        }

        results_file = tmp_path / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)

        return str(results_file)

    @patch('experiments.rejudge.create_judge')
    def test_rejudger_loads_results(self, mock_create_judge, sample_results):
        """Test Rejudger loads results correctly."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        assert len(rejudger.original_results['results']) == 2
        assert rejudger.judge == mock_judge

    @patch('experiments.rejudge.create_judge')
    def test_rejudge_single_updates_fields(self, mock_create_judge, sample_results):
        """Test _rejudge_single updates result fields."""
        mock_judge = Mock()
        mock_judge.identifier = "new_judge"
        mock_judge.judge_id = "categorical_v1"
        mock_judge.evaluate.return_value = {
            'score': 4,
            'category': 'GOOD',
            'reasoning': 'Functionally correct',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        original = rejudger.original_results['results'][0]
        rejudged = rejudger._rejudge_single(original)

        assert rejudged['judge_type'] == "categorical_v1"
        assert rejudged['judge_identifier'] == "new_judge"
        assert rejudged['judge_evaluation']['score'] == 4
        assert rejudged['correctness_score'] == 4

    @patch('experiments.rejudge.create_judge')
    def test_rejudge_single_handles_integrity_judge(self, mock_create_judge, sample_results):
        """Test _rejudge_single handles integrity judge pass/fail."""
        mock_judge = Mock()
        mock_judge.identifier = "integrity_judge"
        mock_judge.judge_id = "integrity_v1"
        mock_judge.evaluate.return_value = {
            'passed': True,
            'confidence': 0.9,
            'reasoning': 'Correctly refused',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        original = rejudger.original_results['results'][0]
        rejudged = rejudger._rejudge_single(original)

        # passed=True should convert to correctness_score=1.0
        assert rejudged['correctness_score'] == 1.0

    @patch('experiments.rejudge.create_judge')
    def test_rejudge_all_processes_all_results(self, mock_create_judge, sample_results):
        """Test rejudge_all processes all results."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"
        mock_judge.judge_id = "correctness_v1"
        mock_judge.model_name = "test-model"
        mock_judge.evaluate.return_value = {
            'score': 0.8,
            'reasoning': 'Good',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        new_results = rejudger.rejudge_all()

        assert len(new_results['results']) == 2
        assert mock_judge.evaluate.call_count == 2
        assert 'rejudged_at' in new_results['metadata']
        assert 'original_judge' in new_results['metadata']

    @patch('experiments.rejudge.create_judge')
    def test_rejudge_all_generates_summary(self, mock_create_judge, sample_results):
        """Test rejudge_all generates new summary."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"
        mock_judge.judge_id = "correctness_v1"
        mock_judge.model_name = "test-model"
        mock_judge.evaluate.return_value = {
            'score': 0.8,
            'reasoning': 'Good',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        new_results = rejudger.rejudge_all()

        assert 'summary' in new_results
        assert 'keyword' in new_results['summary']

    @patch('experiments.rejudge.create_judge')
    def test_save_results_creates_file(self, mock_create_judge, sample_results, tmp_path):
        """Test save_results creates output file."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"
        mock_judge.judge_id = "correctness_v1"
        mock_judge.model_name = "test-model"
        mock_judge.evaluate.return_value = {
            'score': 0.8,
            'reasoning': 'Good',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        new_results = rejudger.rejudge_all()
        output_path = tmp_path / "rejudged_results.json"
        rejudger.save_results(new_results, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)

        assert len(saved['results']) == 2
        assert saved['metadata']['judge']['identifier'] == "test_judge"

    @patch('experiments.rejudge.create_judge')
    def test_generate_summary_by_complexity(self, mock_create_judge, sample_results):
        """Test summary includes breakdown by complexity."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"
        mock_judge.judge_id = "correctness_v1"
        mock_judge.model_name = "test-model"
        mock_judge.evaluate.return_value = {
            'score': 0.8,
            'reasoning': 'Good',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        new_results = rejudger.rejudge_all()

        # Should have breakdown by complexity
        agent_summary = new_results['summary']['keyword']
        assert 'by_complexity' in agent_summary
        assert 'simple' in agent_summary['by_complexity']

    @patch('experiments.rejudge.create_judge')
    def test_generate_summary_by_category(self, mock_create_judge, sample_results):
        """Test summary includes breakdown by category."""
        mock_judge = Mock()
        mock_judge.identifier = "test_judge"
        mock_judge.judge_id = "correctness_v1"
        mock_judge.model_name = "test-model"
        mock_judge.evaluate.return_value = {
            'score': 0.8,
            'reasoning': 'Good',
            'issues': []
        }

        rejudger = Rejudger(
            results_path=sample_results,
            judge=mock_judge
        )

        new_results = rejudger.rejudge_all()

        # Should have breakdown by category
        agent_summary = new_results['summary']['keyword']
        assert 'by_category' in agent_summary
        assert 'authentication' in agent_summary['by_category']
