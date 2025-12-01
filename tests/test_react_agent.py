"""Unit tests for ReAct agent."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.agents.react_agent import ReActAgent, AgentState


@pytest.fixture
def sample_schema_file():
    """Create a temporary schema file for testing."""
    schema = {
        "endpoint_events": {
            "description": "Endpoint security events",
            "category": "endpoint",
            "fields": [
                {"name": "id", "type": "integer", "description": "Event ID"},
                {"name": "timestamp", "type": "timestamp", "description": "Event timestamp"},
                {"name": "severity", "type": "string", "description": "Event severity level"},
                {"name": "event_type", "type": "string", "description": "Type of event"},
                {"name": "hostname", "type": "string", "description": "Host name"}
            ]
        },
        "authentication_events": {
            "description": "Authentication and login events",
            "category": "authentication",
            "fields": [
                {"name": "id", "type": "integer", "description": "Event ID"},
                {"name": "timestamp", "type": "timestamp", "description": "Event timestamp"},
                {"name": "user_id", "type": "string", "description": "User ID"},
                {"name": "status", "type": "string", "description": "Login status"},
                {"name": "ip_address", "type": "string", "description": "Source IP"}
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
            'table_name': 'endpoint_events',
            'schema': {
                'description': 'Endpoint security events',
                'category': 'endpoint',
                'fields': [
                    {'name': 'id', 'type': 'integer'},
                    {'name': 'timestamp', 'type': 'timestamp'},
                    {'name': 'severity', 'type': 'string'}
                ]
            },
            'score': 0.95,
            'match_type': 'semantic'
        }
    ]


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentState(question="test question")

        assert state.question == "test question"
        assert state.iteration == 0
        assert state.retrieved_tables == []
        assert state.retrieved_table_schemas == []
        assert state.generated_queries == []
        assert state.validation_results == []
        assert state.reasoning_trace == []
        assert state.has_submitted_answer is False
        assert state.final_answer is None
        assert state.retrieval_calls == 0
        assert state.validation_attempts == 0

    def test_state_modification(self):
        """Test state can be modified."""
        state = AgentState(question="test")
        state.iteration = 5
        state.retrieved_tables.append("table1")
        state.has_submitted_answer = True

        assert state.iteration == 5
        assert "table1" in state.retrieved_tables
        assert state.has_submitted_answer is True


class TestReActAgentInitialization:
    """Tests for ReAct agent initialization."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_initialization_semantic(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test agent initializes correctly with semantic retrieval."""
        agent = ReActAgent(
            schema_path=sample_schema_file,
            model="test-model",
            top_k_tables=3,
            retrieval_type="semantic"
        )

        # Should load schemas
        assert agent.schemas is not None
        assert 'endpoint_events' in agent.schemas
        assert 'authentication_events' in agent.schemas

        # Should set configuration
        assert agent.top_k_tables == 3
        assert agent.model_name == "test-model"
        assert agent.retrieval_type == "semantic"

        # Should initialize retriever
        mock_retrieval_class.assert_called_once()

        # Should initialize validator
        assert agent.validator is not None
        assert agent.validator.schemas == agent.schemas

        # Should initialize Agno agent with tools
        mock_agent_class.assert_called_once()

    @patch('src.agents.react_agent.KeywordRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_initialization_keyword(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test agent initializes correctly with keyword retrieval."""
        agent = ReActAgent(
            schema_path=sample_schema_file,
            retrieval_type="keyword"
        )

        # Should use keyword retrieval
        assert agent.retrieval_type == "keyword"
        mock_retrieval_class.assert_called_once()

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_initialization_with_defaults(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test initialization with default parameters."""
        agent = ReActAgent(schema_path=sample_schema_file)

        # Should use default values
        assert agent.top_k_tables == 5
        assert agent.max_iterations == 10
        assert agent.max_retrieval_calls == 3
        assert agent.max_validation_attempts == 4

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_tools_created(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that tools are created during initialization."""
        agent = ReActAgent(schema_path=sample_schema_file)

        # Should have created tools
        assert agent._tools is not None
        assert len(agent._tools) == 3  # retrieve_tables, validate_sql, submit_answer


class TestGetInstructions:
    """Tests for _get_instructions method."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_get_instructions_returns_string(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions are returned as string."""
        agent = ReActAgent(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_instructions_content(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions contain key guidance."""
        agent = ReActAgent(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        # Should mention key concepts
        assert 'retrieve_tables' in instructions
        assert 'validate_sql' in instructions
        assert 'submit_answer' in instructions
        assert 'confidence' in instructions.lower()
        assert 'ReAct' in instructions


class TestGetSuggestion:
    """Tests for _get_suggestion method."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_suggestion_for_unknown_tables(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown tables error."""
        agent = ReActAgent(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Unknown tables: foo, bar")

        assert suggestion is not None
        assert 'retrieve_tables' in suggestion.lower()

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_suggestion_for_unknown_fields(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown fields error."""
        agent = ReActAgent(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Unknown fields: xyz")

        assert suggestion is not None
        assert 'field' in suggestion.lower()

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_suggestion_for_syntax_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for syntax error."""
        agent = ReActAgent(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Invalid SQL syntax")

        assert suggestion is not None
        assert 'syntax' in suggestion.lower()

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_suggestion_for_unknown_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown error type returns None."""
        agent = ReActAgent(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Some random error")

        assert suggestion is None


class TestExplainRetrieval:
    """Tests for explain_retrieval method."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_explain_retrieval(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test retrieval explanation."""
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgent(schema_path=sample_schema_file)
        result = agent.explain_retrieval("test question", k=3)

        # Should call retriever with correct parameters
        mock_retriever.get_top_k.assert_called_once_with(question="test question", k=3)

        # Should return structured explanation
        assert 'question' in result
        assert result['question'] == "test question"
        assert 'tables_retrieved' in result
        assert len(result['tables_retrieved']) == 1

        # Check table info structure
        table_info = result['tables_retrieved'][0]
        assert 'table' in table_info
        assert 'category' in table_info
        assert 'score' in table_info
        assert 'description' in table_info


class TestRun:
    """Tests for run method."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_run_with_submitted_answer(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test successful run with submitted answer."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = None

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = ReActAgent(schema_path=sample_schema_file)

        # Mock run to set state and simulate submit_answer tool being called
        def side_effect(*args, **kwargs):
            # Simulate what happens when tools are called during run
            agent._state.has_submitted_answer = True
            agent._state.final_answer = {
                'query': 'SELECT * FROM endpoint_events',
                'explanation': 'Get all endpoint events',
                'tables_used': ['endpoint_events'],
                'confidence': 0.9,
                'reasoning_steps': ['Retrieved tables', 'Validated query']
            }
            return mock_run_output

        mock_agent.run.side_effect = side_effect

        result = agent.run("Get all endpoint events")

        # Should return the submitted answer
        assert result['query'] == 'SELECT * FROM endpoint_events'
        assert result['confidence'] == 0.9
        assert 'tables_used' in result

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_run_without_submit_returns_low_confidence(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test run when agent doesn't use submit_answer tool."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = "Here is my answer without using the tool"

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        agent = ReActAgent(schema_path=sample_schema_file)
        result = agent.run("Test question")

        # Should return low confidence response
        assert result['confidence'] <= 0.3
        assert 'submit_answer tool' in str(result['reasoning_steps'])

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_run_handles_exception(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that run handles exceptions gracefully."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent

        agent = ReActAgent(schema_path=sample_schema_file)
        result = agent.run("Test question")

        # Should return error response
        assert 'error' in result
        assert result['query'] is None
        assert result['confidence'] == 0.0
        assert "Test error" in result['explanation']

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_run_initializes_state(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that run initializes state correctly."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = "Response"

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        agent = ReActAgent(schema_path=sample_schema_file)

        # State should be None before run
        assert agent._state is None

        agent.run("Test question")

        # State should be initialized after run
        assert agent._state is not None
        assert agent._state.question == "Test question"


class TestToolFunctions:
    """Tests for tool function behavior."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_retrieve_tables_tool(self, mock_agent_class, mock_retrieval_class, sample_schema_file, mock_retrieval_results):
        """Test retrieve_tables tool function."""
        mock_retriever = Mock()
        mock_retriever.get_top_k.return_value = mock_retrieval_results
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgent(schema_path=sample_schema_file)
        agent._state = AgentState(question="test")

        # Get the retrieve_tables tool
        retrieve_tables = agent._tools[0]

        # Call the tool (simulating Agno calling it)
        # Note: We can't directly call decorated tools, but we can test the agent behavior

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_validate_sql_tool_valid_query(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test validate_sql tool with valid query."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgent(schema_path=sample_schema_file)
        agent._state = AgentState(question="test")

        # Test validation through the validator directly
        result = agent.validator.validate("SELECT * FROM endpoint_events", strict=False)
        assert result['valid'] is True

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_validate_sql_tool_invalid_query(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test validate_sql tool with invalid query."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgent(schema_path=sample_schema_file)
        agent._state = AgentState(question="test")

        # Test validation through the validator directly
        result = agent.validator.validate("SELECT * FROM nonexistent_table", strict=False)
        assert result['valid'] is False
        assert 'Unknown tables' in str(result['errors'])


class TestFormatResponse:
    """Tests for _format_response method (inherited from BaseAgent)."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_format_response(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test response formatting."""
        from src.agents.base import SQLQueryResponse

        agent = ReActAgent(schema_path=sample_schema_file)

        response = SQLQueryResponse(
            query="SELECT * FROM endpoint_events",
            explanation="Get all events",
            tables_used=["endpoint_events"],
            confidence=0.9,
            reasoning_steps=["Step 1", "Step 2"]
        )

        result = agent._format_response(response)

        assert isinstance(result, dict)
        assert result['query'] == response.query
        assert result['explanation'] == response.explanation
        assert result['tables_used'] == response.tables_used
        assert result['confidence'] == response.confidence


class TestHandleError:
    """Tests for _handle_error method (inherited from BaseAgent)."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_handle_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test error handling."""
        agent = ReActAgent(schema_path=sample_schema_file)
        error = Exception("Test error message")
        result = agent._handle_error("test question", error)

        assert isinstance(result, dict)
        assert result['query'] is None
        assert result['confidence'] == 0.0
        assert 'error' in result
        assert "Test error message" in result['error']
        assert "Test error message" in result['explanation']


class TestLoopControl:
    """Tests for loop control and stopping conditions."""

    @patch('src.agents.react_agent.SemanticRetrieval')
    @patch('src.agents.react_agent.Agent')
    def test_max_iterations_config(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test max iterations configuration."""
        agent = ReActAgent(
            schema_path=sample_schema_file,
            max_iterations=5,
            max_retrieval_calls=2,
            max_validation_attempts=3
        )

        assert agent.max_iterations == 5
        assert agent.max_retrieval_calls == 2
        assert agent.max_validation_attempts == 3
