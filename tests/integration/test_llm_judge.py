"""Integration tests for LLM judge."""

import pytest
import os
from experiments.utils.llm_judge import LLMJudge


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY required for integration tests"
)
class TestLLMJudge:
    """Integration tests for LLM judge."""

    def test_llm_judge_initialization(self, integration_model):
        """Test LLMJudge initializes with valid caching parameters."""
        judge = LLMJudge(model=integration_model)
        assert judge.agent is not None

    def test_llm_judge_evaluation(self, integration_model):
        """Test LLMJudge can evaluate correctness."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
        # Check if newer versions of Haiku support this feature and remove skip if so
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        judge = LLMJudge(model=integration_model)

        result = judge.evaluate_correctness(
            question="Show all high severity events",
            reference_sql="SELECT * FROM endpoint_events WHERE severity IN ('high', 'critical')",
            generated_sql="SELECT * FROM endpoint_events WHERE severity = 'high' OR severity = 'critical'"
        )

        # Verify evaluation structure
        assert 'score' in result
        assert 'reasoning' in result
        assert 'issues' in result
        assert 0.0 <= result['score'] <= 1.0
        # Should be high score (semantically equivalent)
        assert result['score'] >= 0.8  # Slightly lower threshold for haiku
