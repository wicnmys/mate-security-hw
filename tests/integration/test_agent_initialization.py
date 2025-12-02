"""Integration tests for agent initialization with real API."""

import pytest
import os
from src.agents.semantic_agent import SemanticAgent
from src.agents.keyword_agent import KeywordAgent
from src.agents.react_agent import ReActAgent
from src.agents.react_agent_v2 import ReActAgentV2


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

    def test_react_agent_initialization(self, integration_model):
        """Test ReActAgent initializes with valid caching parameters."""
        agent = ReActAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5,
            retrieval_type="semantic"
        )

        assert agent.agent is not None
        assert agent.retriever is not None
        assert agent.validator is not None
        assert len(agent._tools) == 3  # retrieve_tables, validate_sql, submit_answer

    def test_react_agent_v2_initialization(self, integration_model):
        """Test ReActAgentV2 initializes with valid caching parameters and judge."""
        agent = ReActAgentV2(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5,
            retrieval_type="semantic",
            judge_model=integration_model
        )

        assert agent.agent is not None
        assert agent.retriever is not None
        assert agent.validator is not None
        assert agent._judge_agent is not None
        assert len(agent._tools) == 4  # retrieve_tables, validate_sql, llm_judge_evaluate, submit_answer


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

    def test_react_agent_generates_query(self, integration_model):
        """Test ReActAgent can generate a query end-to-end using tools."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
        # Check if newer versions of Haiku support this feature and remove skip if so
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = ReActAgent(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5,
            retrieval_type="semantic"
        )

        result = agent.run("Show me all critical endpoint events")

        assert 'query' in result
        assert 'explanation' in result
        assert 'tables_used' in result
        assert 'confidence' in result
        assert 'reasoning_steps' in result

    def test_react_agent_v2_generates_query(self, integration_model):
        """Test ReActAgentV2 can generate a query end-to-end using dual validation."""
        # TODO: Revisit this limitation - as of 2025-12-01, Haiku doesn't support structured outputs (output_format)
        # Check if newer versions of Haiku support this feature and remove skip if so
        if 'haiku' in integration_model.lower():
            pytest.skip(f"Skipping: {integration_model} doesn't support structured outputs")

        agent = ReActAgentV2(
            schema_path="schemas/dataset.json",
            model=integration_model,
            top_k_tables=5,
            retrieval_type="semantic",
            judge_model=integration_model
        )

        result = agent.run("Show me all high severity endpoint events")

        assert 'query' in result
        assert 'explanation' in result
        assert 'tables_used' in result
        assert 'confidence' in result
        assert 'reasoning_steps' in result
        # V2-specific: should have judge scores
        assert 'judge_scores' in result
