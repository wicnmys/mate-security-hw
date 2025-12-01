"""Integration tests for agent initialization with real API."""

import pytest
import os
from src.agents.semantic_agent import SemanticAgent
from src.agents.keyword_agent import KeywordAgent


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY required for integration tests"
)
class TestAgentInitialization:
    """Integration tests for agent initialization with real API."""

    def test_semantic_agent_initialization(self, integration_model):
        """Test SemanticAgent initializes with valid caching parameters."""
        agent = SemanticAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        # Verify agent is properly initialized
        assert agent.agent is not None
        assert agent.retriever is not None
        assert agent.validator is not None

    def test_keyword_agent_initialization(self, integration_model):
        """Test KeywordAgent initializes with valid caching parameters."""
        agent = KeywordAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        assert agent.agent is not None
        assert agent.retriever is not None
        assert agent.validator is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY required for integration tests"
)
class TestAgentQueryGeneration:
    """Integration tests for end-to-end query generation."""

    def test_semantic_agent_generates_query(self, integration_model):
        """Test SemanticAgent can generate a query end-to-end."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
        # Check if newer versions of Haiku support this feature and remove skip if so
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = SemanticAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        result = agent.run("Show me all high severity endpoint events")

        # Verify response structure
        assert 'query' in result
        assert 'explanation' in result
        assert 'tables_used' in result
        assert 'confidence' in result
        assert result['query'] is not None
        assert len(result['query']) > 0

    def test_keyword_agent_generates_query(self, integration_model):
        """Test KeywordAgent can generate a query end-to-end."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
        # Check if newer versions of Haiku support this feature and remove skip if so
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = KeywordAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5
        )

        result = agent.run("Show me failed login attempts")

        assert 'query' in result
        assert 'explanation' in result
        assert result['query'] is not None
