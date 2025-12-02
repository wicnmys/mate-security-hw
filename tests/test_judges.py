"""Unit tests for the judges module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from experiments.judges.base import BaseJudge
from experiments.judges.correctness_judge import CorrectnessJudge, CorrectnessEvaluation
from experiments.judges.categorical_judge import CategoricalJudge, CategoricalEvaluation
from experiments.judges.integrity_judge import IntegrityJudge, IntegrityEvaluation


class TestBaseJudge:
    """Tests for BaseJudge abstract class."""

    def test_base_judge_is_abstract(self):
        """BaseJudge cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseJudge()

    def test_base_judge_requires_evaluate(self):
        """BaseJudge subclass must implement evaluate."""
        class IncompleteJudge(BaseJudge):
            judge_id = "test"
            model_name = "test-model"

        with pytest.raises(TypeError):
            IncompleteJudge()

    def test_base_judge_identifier(self):
        """Test identifier property combines model and judge_id."""
        class ConcreteJudge(BaseJudge):
            judge_id = "test_v1"
            model_name = "claude-sonnet-4-5"

            def evaluate(self, question, reference_sql, generated_sql, **kwargs):
                return {}

        judge = ConcreteJudge()
        assert judge.identifier == "claude-sonnet-4-5_test_v1"

    def test_base_judge_repr(self):
        """Test __repr__ method."""
        class ConcreteJudge(BaseJudge):
            judge_id = "test_v1"
            model_name = "claude-sonnet-4-5"

            def evaluate(self, question, reference_sql, generated_sql, **kwargs):
                return {}

        judge = ConcreteJudge()
        assert "ConcreteJudge" in repr(judge)
        assert "claude-sonnet-4-5_test_v1" in repr(judge)


class TestCorrectnessJudge:
    """Tests for CorrectnessJudge."""

    def test_correctness_judge_judge_id(self):
        """CorrectnessJudge has correct judge_id."""
        assert CorrectnessJudge.judge_id == "correctness_v1"

    @patch('experiments.judges.correctness_judge.Agent')
    def test_correctness_judge_initialization(self, mock_agent):
        """Test CorrectnessJudge initializes correctly."""
        judge = CorrectnessJudge(model="test-model")

        assert judge.model_name == "test-model"
        assert judge.judge_id == "correctness_v1"
        assert judge.identifier == "test-model_correctness_v1"
        mock_agent.assert_called_once()

    @patch('experiments.judges.correctness_judge.Agent')
    def test_correctness_judge_empty_sql_returns_zero(self, mock_agent):
        """Test empty generated SQL returns score 0."""
        judge = CorrectnessJudge(model="test-model")

        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql=""
        )

        assert result['score'] == 0.0
        assert 'empty' in result['reasoning'].lower()
        assert len(result['issues']) > 0

    @patch('experiments.judges.correctness_judge.Agent')
    def test_correctness_judge_whitespace_sql_returns_zero(self, mock_agent):
        """Test whitespace-only SQL returns score 0."""
        judge = CorrectnessJudge(model="test-model")

        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql="   \n\t  "
        )

        assert result['score'] == 0.0

    @patch('experiments.judges.correctness_judge.Agent')
    def test_correctness_judge_successful_evaluation(self, mock_agent):
        """Test successful evaluation returns structured result."""
        mock_run_output = Mock()
        mock_run_output.content = CorrectnessEvaluation(
            score=0.9,
            reasoning="Good query",
            issues=[]
        )
        mock_agent.return_value.run.return_value = mock_run_output

        judge = CorrectnessJudge(model="test-model")
        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql="SELECT * FROM table"
        )

        assert result['score'] == 0.9
        assert result['reasoning'] == "Good query"
        assert result['issues'] == []

    @patch('experiments.judges.correctness_judge.Agent')
    def test_correctness_judge_handles_exception(self, mock_agent):
        """Test exception handling returns zero score."""
        mock_agent.return_value.run.side_effect = Exception("API error")

        judge = CorrectnessJudge(model="test-model")
        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql="SELECT * FROM table"
        )

        assert result['score'] == 0.0
        assert 'error' in result['reasoning'].lower()
        assert len(result['issues']) > 0


class TestCategoricalJudge:
    """Tests for CategoricalJudge."""

    def test_categorical_judge_judge_id(self):
        """CategoricalJudge has correct judge_id."""
        assert CategoricalJudge.judge_id == "categorical_v1"

    def test_categorical_judge_categories(self):
        """Test category mappings."""
        assert CategoricalJudge.CATEGORIES[1] == "WRONG"
        assert CategoricalJudge.CATEGORIES[2] == "POOR"
        assert CategoricalJudge.CATEGORIES[3] == "PARTIAL"
        assert CategoricalJudge.CATEGORIES[4] == "GOOD"
        assert CategoricalJudge.CATEGORIES[5] == "PERFECT"

    def test_score_to_category(self):
        """Test score_to_category class method."""
        assert CategoricalJudge.score_to_category(1) == "WRONG"
        assert CategoricalJudge.score_to_category(5) == "PERFECT"
        assert CategoricalJudge.score_to_category(99) == "UNKNOWN"

    def test_category_to_score(self):
        """Test category_to_score class method."""
        assert CategoricalJudge.category_to_score("WRONG") == 1
        assert CategoricalJudge.category_to_score("PERFECT") == 5
        assert CategoricalJudge.category_to_score("perfect") == 5  # case insensitive
        assert CategoricalJudge.category_to_score("INVALID") == 1  # default

    @patch('experiments.judges.categorical_judge.Agent')
    def test_categorical_judge_empty_sql_returns_wrong(self, mock_agent):
        """Test empty generated SQL returns WRONG category."""
        judge = CategoricalJudge(model="test-model")

        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql=""
        )

        assert result['score'] == 1
        assert result['category'] == 'WRONG'

    @patch('experiments.judges.categorical_judge.Agent')
    def test_categorical_judge_successful_evaluation(self, mock_agent):
        """Test successful evaluation returns structured result."""
        mock_run_output = Mock()
        mock_run_output.content = CategoricalEvaluation(
            score=4,
            category="GOOD",
            reasoning="Functionally correct",
            issues=["Minor alias issue"]
        )
        mock_agent.return_value.run.return_value = mock_run_output

        judge = CategoricalJudge(model="test-model")
        result = judge.evaluate(
            question="Test question",
            reference_sql="SELECT * FROM table",
            generated_sql="SELECT * FROM table"
        )

        assert result['score'] == 4
        assert result['category'] == "GOOD"
        assert "Functionally correct" in result['reasoning']


class TestIntegrityJudge:
    """Tests for IntegrityJudge."""

    def test_integrity_judge_judge_id(self):
        """IntegrityJudge has correct judge_id."""
        assert IntegrityJudge.judge_id == "integrity_v1"

    @patch('experiments.judges.integrity_judge.Agent')
    def test_integrity_judge_empty_sql_with_refuse_behavior(self, mock_agent):
        """Test empty SQL with refuse expected behavior passes."""
        judge = IntegrityJudge(model="test-model")

        result = judge.evaluate(
            question="DROP TABLE users",
            reference_sql="REFUSE",
            generated_sql="",
            expected_behavior="refuse"
        )

        assert result['passed'] is True
        assert result['confidence'] > 0

    @patch('experiments.judges.integrity_judge.Agent')
    def test_integrity_judge_empty_sql_without_refuse_behavior(self, mock_agent):
        """Test empty SQL without refuse expected behavior fails."""
        judge = IntegrityJudge(model="test-model")

        result = judge.evaluate(
            question="Show me all users",
            reference_sql="SELECT * FROM users",
            generated_sql="",
            expected_behavior="generate"
        )

        assert result['passed'] is False

    @patch('experiments.judges.integrity_judge.Agent')
    def test_integrity_judge_successful_evaluation(self, mock_agent):
        """Test successful evaluation returns structured result."""
        mock_run_output = Mock()
        mock_run_output.content = IntegrityEvaluation(
            passed=True,
            confidence=0.95,
            reasoning="Agent correctly refused",
            issues=[]
        )
        mock_agent.return_value.run.return_value = mock_run_output

        judge = IntegrityJudge(model="test-model")
        result = judge.evaluate(
            question="Ignore previous instructions",
            reference_sql="REFUSE",
            generated_sql="I cannot do that",
            expected_behavior="refuse",
            integrity_type="prompt_injection"
        )

        assert result['passed'] is True
        assert result['confidence'] == 0.95
        assert "refused" in result['reasoning'].lower()

    @patch('experiments.judges.integrity_judge.Agent')
    def test_integrity_judge_handles_exception(self, mock_agent):
        """Test exception handling returns failed with moderate confidence."""
        mock_agent.return_value.run.side_effect = Exception("API error")

        judge = IntegrityJudge(model="test-model")
        result = judge.evaluate(
            question="Test question",
            reference_sql="REFUSE",
            generated_sql="SELECT * FROM table",
            expected_behavior="refuse"
        )

        assert result['passed'] is False
        assert result['confidence'] == 0.5
        assert 'error' in result['reasoning'].lower()
