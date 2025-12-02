"""Unit tests for run_experiments module."""

import pytest
from unittest.mock import Mock, patch

from experiments.run_experiments import (
    JUDGE_REGISTRY,
    create_judge,
    generate_output_filename,
)
from experiments.judges import CorrectnessJudge, CategoricalJudge, IntegrityJudge


class TestJudgeRegistry:
    """Tests for the judge registry."""

    def test_registry_has_correctness(self):
        """Registry includes correctness judge."""
        assert "correctness" in JUDGE_REGISTRY
        assert JUDGE_REGISTRY["correctness"] == CorrectnessJudge

    def test_registry_has_categorical(self):
        """Registry includes categorical judge."""
        assert "categorical" in JUDGE_REGISTRY
        assert JUDGE_REGISTRY["categorical"] == CategoricalJudge

    def test_registry_has_integrity(self):
        """Registry includes integrity judge."""
        assert "integrity" in JUDGE_REGISTRY
        assert JUDGE_REGISTRY["integrity"] == IntegrityJudge


class TestCreateJudge:
    """Tests for create_judge function."""

    @patch('experiments.judges.correctness_judge.Agent')
    def test_create_correctness_judge(self, mock_agent):
        """Test creating correctness judge."""
        judge = create_judge("correctness", "test-model")

        assert isinstance(judge, CorrectnessJudge)
        assert judge.model_name == "test-model"
        assert judge.judge_id == "correctness_v1"

    @patch('experiments.judges.categorical_judge.Agent')
    def test_create_categorical_judge(self, mock_agent):
        """Test creating categorical judge."""
        judge = create_judge("categorical", "test-model")

        assert isinstance(judge, CategoricalJudge)
        assert judge.model_name == "test-model"

    @patch('experiments.judges.integrity_judge.Agent')
    def test_create_integrity_judge(self, mock_agent):
        """Test creating integrity judge."""
        judge = create_judge("integrity", "test-model")

        assert isinstance(judge, IntegrityJudge)
        assert judge.model_name == "test-model"

    def test_create_unknown_judge_raises(self):
        """Test creating unknown judge raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_judge("unknown_judge", "test-model")

        assert "unknown_judge" in str(exc_info.value)
        assert "Valid types" in str(exc_info.value)


class TestGenerateOutputFilename:
    """Tests for generate_output_filename function."""

    def test_basic_filename(self):
        """Test basic filename generation."""
        filename = generate_output_filename(
            model="claude-sonnet-4-5",
            agents=["keyword"],
            judge_identifier="claude-sonnet-4-5_correctness_v1",
            experiment_type="main"
        )

        assert filename.endswith(".json")
        assert "sonnet_4_5" in filename
        assert "keyword" in filename
        assert "correctness" in filename

    def test_multiple_agents_sorted(self):
        """Test multiple agents are sorted in filename."""
        filename = generate_output_filename(
            model="claude-sonnet-4-5",
            agents=["semantic", "keyword"],  # unsorted
            judge_identifier="test_correctness_v1",
            experiment_type="main"
        )

        # Agents should be sorted: keyword-semantic
        assert "keyword-semantic" in filename

    def test_model_sanitization(self):
        """Test model name is sanitized."""
        filename = generate_output_filename(
            model="claude-sonnet-4-5",
            agents=["keyword"],
            judge_identifier="test_correctness_v1",
            experiment_type="main"
        )

        # claude- prefix and hyphens should be handled
        assert "sonnet_4_5" in filename
        assert "claude-" not in filename

    def test_timestamp_format(self):
        """Test filename includes timestamp."""
        filename = generate_output_filename(
            model="test-model",
            agents=["keyword"],
            judge_identifier="test_correctness_v1",
            experiment_type="main"
        )

        # Timestamp format: YYYYMMDDTHHMMSS
        import re
        assert re.search(r'\d{8}T\d{6}', filename)


class TestExperimentRunnerInit:
    """Tests for ExperimentRunner initialization."""

    @patch('experiments.run_experiments.create_judge')
    def test_runner_with_provided_judge(self, mock_create_judge):
        """Test runner uses provided judge instead of creating one."""
        from experiments.run_experiments import ExperimentRunner
        from experiments.judges.base import BaseJudge

        # Create a mock judge
        mock_judge = Mock(spec=BaseJudge)
        mock_judge.identifier = "test_judge"

        # Patch _load_test_cases to avoid file I/O
        with patch.object(ExperimentRunner, '_load_test_cases', return_value=[]):
            runner = ExperimentRunner(
                test_cases_path="test.json",
                agents={},
                judge=mock_judge
            )

            # Should use provided judge, not create one
            mock_create_judge.assert_not_called()
            assert runner.judge == mock_judge

    @patch('experiments.run_experiments.create_judge')
    def test_runner_creates_judge_when_not_provided(self, mock_create_judge):
        """Test runner creates judge when not provided."""
        from experiments.run_experiments import ExperimentRunner

        mock_judge = Mock()
        mock_judge.identifier = "test_identifier"
        mock_create_judge.return_value = mock_judge

        with patch.object(ExperimentRunner, '_load_test_cases', return_value=[]):
            runner = ExperimentRunner(
                test_cases_path="test.json",
                agents={},
                judge_type="correctness",
                judge_model="test-model"
            )

            mock_create_judge.assert_called_once_with("correctness", "test-model")

    @patch('experiments.run_experiments.create_judge')
    def test_runner_applies_limit(self, mock_create_judge):
        """Test runner applies limit to test cases."""
        from experiments.run_experiments import ExperimentRunner

        mock_judge = Mock()
        mock_judge.identifier = "test"
        mock_create_judge.return_value = mock_judge

        test_cases = [{"id": i} for i in range(10)]

        with patch.object(ExperimentRunner, '_load_test_cases', return_value=test_cases):
            runner = ExperimentRunner(
                test_cases_path="test.json",
                agents={},
                limit=3
            )

            assert len(runner.test_cases) == 3

    @patch('experiments.run_experiments.create_judge')
    def test_runner_stores_experiment_type(self, mock_create_judge):
        """Test runner stores experiment type."""
        from experiments.run_experiments import ExperimentRunner

        mock_judge = Mock()
        mock_judge.identifier = "test"
        mock_create_judge.return_value = mock_judge

        with patch.object(ExperimentRunner, '_load_test_cases', return_value=[]):
            runner = ExperimentRunner(
                test_cases_path="test.json",
                agents={},
                experiment_type="integrity"
            )

            assert runner.experiment_type == "integrity"
