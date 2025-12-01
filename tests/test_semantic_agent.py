"""Unit tests for semantic agent."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.agents.semantic_agent import SemanticAgent
from src.agents.base import SQLQueryResponse


@pytest.fixture
def sample_schema_file():
    """Create a temporary schema file for testing."""
    schema = {
        "users": {
            "description": "User data",
            "category": "security",
            "fields": [
                {"name": "user_id", "type": "integer"},
                {"name": "username", "type": "string"}
            ]
        },
        "events": {
            "description": "Event logs",
            "category": "logging",
            "fields": [
                {"name": "event_id", "type": "integer"},
                {"name": "severity", "type": "string"}
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(schema, f)
        return f.name


@pytest.fixture
def mock_retrieval_results():
    """Mock retrieval results."""
    return [
        {
            'table_name': 'users',
            'schema': {
                'description': 'User data',
                'category': 'security',
                'fields': [
                    {'name': 'user_id', 'type': 'integer'},
                    {'name': 'username', 'type': 'string'}
                ]
            },
            'score': 0.95,
            'match_type': 'semantic'
        },
        {
            'table_name': 'events',
            'schema': {
                'description': 'Event logs',
                'category': 'logging',
                'fields': [
                    {'name': 'event_id', 'type': 'integer'},
                    {'name': 'severity', 'type': 'string'}
                ]
            },
            'score': 0.85,
            'match_type': 'semantic'
        }
    ]


@pytest.fixture
def mock_sql_response():
    """Mock SQL query response."""
    return SQLQueryResponse(
        query="SELECT username FROM users WHERE user_id = 1",
        explanation="Retrieves username for user with ID 1",
        tables_used=["users"],
        confidence=0.9,
        reasoning_steps=["Identified users table", "Selected username field"]
    )


class TestSemanticAgentInitialization:
    """Tests for semantic agent initialization."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_initialization(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that agent initializes correctly."""
        agent = SemanticAgent(
            schema_path=sample_schema_file,
            model="test-model",
            top_k_tables=3
        )

        # Should load schemas
        assert agent.schemas is not None
        assert 'users' in agent.schemas
        assert 'events' in agent.schemas

        # Should set top_k_tables
        assert agent.top_k_tables == 3

        # Should initialize retriever
        mock_retrieval_class.assert_called_once()

        # Should initialize validator
        assert agent.validator is not None
        assert agent.validator.schemas == agent.schemas

        # Should initialize Agno agent
        mock_agent_class.assert_called_once()

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_initialization_with_defaults(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test initialization with default parameters."""
        agent = SemanticAgent(schema_path=sample_schema_file)

        # Should use default top_k_tables
        assert agent.top_k_tables == 5


class TestGetInstructions:
    """Tests for _get_instructions method."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_get_instructions_returns_string(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions are returned as string."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_instructions_content(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions contain key guidance."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        # Should mention key concepts
        assert 'SQL' in instructions
        assert 'query' in instructions.lower()
        assert 'confidence' in instructions.lower()


class TestBuildSchemaContext:
    """Tests for _build_schema_context method."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_build_schema_context(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test schema context building."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        context = agent._build_schema_context(mock_retrieval_results)

        assert isinstance(context, str)
        # Should include table names
        assert 'users' in context
        assert 'events' in context
        # Should include relevance scores
        assert 'Relevance' in context

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_build_schema_context_formatting(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test that schema context is properly formatted."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        context = agent._build_schema_context(mock_retrieval_results)

        # Should be numbered
        assert '1.' in context
        assert '2.' in context

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_build_schema_context_empty(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test schema context with empty retrieval results."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        context = agent._build_schema_context([])

        assert context == ""


class TestExplainRetrieval:
    """Tests for explain_retrieval method."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_explain_retrieval(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test retrieval explanation."""
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent.explain_retrieval("test question", k=3)

        # Should call retriever with correct parameters
        mock_retriever.get_top_k.assert_called_once_with(question="test question", k=3)

        # Should return structured explanation
        assert 'question' in result
        assert result['question'] == "test question"
        assert 'tables_retrieved' in result
        assert len(result['tables_retrieved']) == 2

        # Check table info structure
        table_info = result['tables_retrieved'][0]
        assert 'table' in table_info
        assert 'category' in table_info
        assert 'score' in table_info
        assert 'description' in table_info


class TestRun:
    """Tests for run method."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_run_success(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results, mock_sql_response):
        """Test successful query generation."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = mock_sql_response

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        # Create agent and run
        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent.run("Get username for user 1")

        # Should retrieve relevant tables
        mock_retriever.get_top_k.assert_called_once()

        # Should call Agno agent
        mock_agent.run.assert_called_once()

        # Should return formatted response
        assert 'query' in result
        assert 'explanation' in result
        assert 'tables_used' in result
        assert 'confidence' in result
        assert result['query'] == mock_sql_response.query

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_run_with_validation_warnings(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test run with validation warnings."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        # Response with query that will generate warnings
        mock_response = SQLQueryResponse(
            query="SELECT nonexistent_field FROM users",
            explanation="Query with invalid field",
            tables_used=["users"],
            confidence=0.9,
            reasoning_steps=[]
        )

        mock_run_output = Mock()
        mock_run_output.content = mock_response

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        # Create agent and run
        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent.run("Get user data")

        # Should have reasoning steps with validation warnings
        assert 'reasoning_steps' in result
        # The validator should have flagged the nonexistent field
        assert len(result['reasoning_steps']) > 0

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_run_with_validation_errors(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test run with validation errors."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        # Response with invalid query (references nonexistent table)
        mock_response = SQLQueryResponse(
            query="SELECT * FROM nonexistent_table",
            explanation="Query with nonexistent table",
            tables_used=["nonexistent_table"],
            confidence=0.9,
            reasoning_steps=[]
        )

        mock_run_output = Mock()
        mock_run_output.content = mock_response

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        # Create agent and run
        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent.run("Get data")

        # Confidence should be reduced
        assert result['confidence'] <= 0.5

        # Should have error in reasoning steps
        assert any('Validation errors' in str(step) for step in result['reasoning_steps'])

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_run_handles_exception(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that run handles exceptions gracefully."""
        # Setup mocks to raise exception
        mock_retriever = Mock()
        mock_retriever.get_top_k.side_effect = Exception("Test error")
        mock_retrieval_class.return_value = mock_retriever

        mock_agent_class.return_value = Mock()

        # Create agent and run
        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent.run("Test question")

        # Should return error response
        assert 'error' in result
        assert result['query'] is None
        assert result['confidence'] == 0.0
        assert "Test error" in result['explanation']

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_run_uses_top_k_parameter(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results, mock_sql_response):
        """Test that run uses the configured top_k_tables parameter."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = mock_sql_response

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        # Create agent with custom top_k
        agent = SemanticAgent(schema_path=sample_schema_file, top_k_tables=3)
        agent.run("Test question")

        # Should use the configured top_k
        call_kwargs = mock_retriever.get_top_k.call_args.kwargs
        assert call_kwargs['k'] == 3


class TestFormatResponse:
    """Tests for _format_response method (inherited from BaseAgent)."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_format_response(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_sql_response):
        """Test response formatting."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        result = agent._format_response(mock_sql_response)

        assert isinstance(result, dict)
        assert result['query'] == mock_sql_response.query
        assert result['explanation'] == mock_sql_response.explanation
        assert result['tables_used'] == mock_sql_response.tables_used
        assert result['confidence'] == mock_sql_response.confidence


class TestHandleError:
    """Tests for _handle_error method (inherited from BaseAgent)."""

    @patch('src.agents.semantic_agent.SemanticRetrieval')
    @patch('src.agents.sql_agent.Agent')
    def test_handle_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test error handling."""
        agent = SemanticAgent(schema_path=sample_schema_file)
        error = Exception("Test error message")
        result = agent._handle_error("test question", error)

        assert isinstance(result, dict)
        assert result['query'] is None
        assert result['confidence'] == 0.0
        assert 'error' in result
        assert "Test error message" in result['error']
        assert "Test error message" in result['explanation']
