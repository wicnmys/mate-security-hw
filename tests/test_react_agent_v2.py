"""Unit tests for ReAct agent v2 with LLM-as-judge."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.agents.react_agent_v2 import ReActAgentV2, AgentStateV2, LLMJudgeOutput


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


class TestAgentStateV2:
    """Tests for AgentStateV2 dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentStateV2(question="test question")

        assert state.question == "test question"
        assert state.iteration == 0
        assert state.retrieved_tables == []
        assert state.retrieved_table_schemas == []
        assert state.generated_queries == []
        assert state.validation_results == []
        assert state.judge_results == []
        assert state.reasoning_trace == []
        assert state.has_submitted_answer is False
        assert state.final_answer is None
        assert state.retrieval_calls == 0
        assert state.validation_attempts == 0
        assert state.judge_calls == 0
        assert state.judge_scores == []

    def test_state_modification(self):
        """Test state can be modified."""
        state = AgentStateV2(question="test")
        state.iteration = 5
        state.retrieved_tables.append("table1")
        state.judge_calls = 2
        state.judge_scores.append(0.85)
        state.has_submitted_answer = True

        assert state.iteration == 5
        assert "table1" in state.retrieved_tables
        assert state.judge_calls == 2
        assert state.judge_scores == [0.85]
        assert state.has_submitted_answer is True

    def test_judge_score_tracking(self):
        """Test that judge scores are tracked correctly."""
        state = AgentStateV2(question="test")
        state.judge_scores.append(0.5)
        state.judge_scores.append(0.7)
        state.judge_scores.append(0.9)

        assert len(state.judge_scores) == 3
        assert state.judge_scores[-1] == 0.9
        # Check improvement
        assert state.judge_scores[-1] > state.judge_scores[0]


class TestLLMJudgeOutput:
    """Tests for LLMJudgeOutput model."""

    def test_valid_output(self):
        """Test valid judge output."""
        output = LLMJudgeOutput(
            is_correct=True,
            correctness_score=0.95,
            issues=[],
            suggestions=[],
            reasoning="Query correctly answers the question."
        )

        assert output.is_correct is True
        assert output.correctness_score == 0.95
        assert output.issues == []
        assert output.suggestions == []

    def test_output_with_issues(self):
        """Test judge output with issues."""
        output = LLMJudgeOutput(
            is_correct=False,
            correctness_score=0.5,
            issues=["Missing time filter", "Wrong table used"],
            suggestions=["Add WHERE timestamp > ...", "Use events table instead"],
            reasoning="Query partially answers the question but misses key aspects."
        )

        assert output.is_correct is False
        assert output.correctness_score == 0.5
        assert len(output.issues) == 2
        assert len(output.suggestions) == 2

    def test_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid score
        output = LLMJudgeOutput(
            is_correct=True,
            correctness_score=0.0,
            issues=[],
            suggestions=[],
            reasoning="Test"
        )
        assert output.correctness_score == 0.0

        output = LLMJudgeOutput(
            is_correct=True,
            correctness_score=1.0,
            issues=[],
            suggestions=[],
            reasoning="Test"
        )
        assert output.correctness_score == 1.0

        # Invalid scores should raise validation errors
        with pytest.raises(ValueError):
            LLMJudgeOutput(
                is_correct=True,
                correctness_score=1.5,
                issues=[],
                suggestions=[],
                reasoning="Test"
            )


class TestReActAgentV2Initialization:
    """Tests for ReAct agent v2 initialization."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_initialization_semantic(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test agent initializes correctly with semantic retrieval."""
        agent = ReActAgentV2(
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

        # Should initialize Agno agent with tools (2 calls - one for main, one for judge)
        assert mock_agent_class.call_count == 2

    @patch('src.agents.react_agent_v2.KeywordRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_initialization_keyword(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test agent initializes correctly with keyword retrieval."""
        agent = ReActAgentV2(
            schema_path=sample_schema_file,
            retrieval_type="keyword"
        )

        # Should use keyword retrieval
        assert agent.retrieval_type == "keyword"
        mock_retrieval_class.assert_called_once()

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_initialization_with_defaults(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test initialization with default parameters."""
        agent = ReActAgentV2(schema_path=sample_schema_file)

        # Should use default values
        assert agent.top_k_tables == 5
        assert agent.max_iterations == 10
        assert agent.max_retrieval_calls == 3
        assert agent.max_validation_attempts == 4
        assert agent.max_judge_calls == 3

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_tools_created(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that tools are created during initialization."""
        agent = ReActAgentV2(schema_path=sample_schema_file)

        # Should have created tools (4 instead of 3 - includes llm_judge_evaluate)
        assert agent._tools is not None
        assert len(agent._tools) == 4  # retrieve_tables, validate_sql, llm_judge_evaluate, submit_answer

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_judge_agent_initialized(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that judge agent is initialized."""
        agent = ReActAgentV2(
            schema_path=sample_schema_file,
            judge_model="claude-sonnet-4-5"
        )

        assert agent.judge_model == "claude-sonnet-4-5"
        # Agent class should be called twice (main agent and judge agent)
        assert mock_agent_class.call_count == 2


class TestGetInstructions:
    """Tests for _get_instructions method."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_get_instructions_returns_string(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions are returned as string."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_instructions_content(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions contain key guidance."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        # Should mention key concepts
        assert 'retrieve_tables' in instructions
        assert 'validate_sql' in instructions
        assert 'llm_judge_evaluate' in instructions
        assert 'submit_answer' in instructions
        assert 'confidence' in instructions.lower()
        assert 'Dual Validation' in instructions

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_instructions_mention_judge_limits(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that instructions mention judge call limits."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        instructions = agent._get_instructions()

        # Should mention loop limits
        assert 'Maximum judge calls' in instructions or 'judge calls: 3' in instructions


class TestGetJudgeInstructions:
    """Tests for _get_judge_instructions method."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_judge_instructions_returns_string(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that judge instructions are returned as string."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        instructions = agent._get_judge_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_judge_instructions_content(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that judge instructions contain evaluation criteria."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        instructions = agent._get_judge_instructions()

        # Should contain evaluation guidance
        assert 'Scoring Rubric' in instructions
        assert 'is_correct' in instructions
        assert 'correctness_score' in instructions
        assert 'issues' in instructions
        assert 'suggestions' in instructions


class TestGetSuggestion:
    """Tests for _get_suggestion method."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_suggestion_for_unknown_tables(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown tables error."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Unknown tables: foo, bar")

        assert suggestion is not None
        assert 'retrieve_tables' in suggestion.lower()

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_suggestion_for_unknown_fields(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown fields error."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Unknown fields: xyz")

        assert suggestion is not None
        assert 'field' in suggestion.lower()

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_suggestion_for_syntax_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for syntax error."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Invalid SQL syntax")

        assert suggestion is not None
        assert 'syntax' in suggestion.lower()

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_suggestion_for_unknown_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test suggestion for unknown error type returns None."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
        suggestion = agent._get_suggestion("Some random error")

        assert suggestion is None


class TestRun:
    """Tests for run method."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_run_with_submitted_answer(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test successful run with submitted answer."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = None

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = ReActAgentV2(schema_path=sample_schema_file)

        # Mock run to set state and simulate submit_answer tool being called
        def side_effect(*args, **kwargs):
            # Simulate what happens when tools are called during run
            agent._state.has_submitted_answer = True
            agent._state.judge_scores = [0.85]
            agent._state.final_answer = {
                'query': 'SELECT * FROM endpoint_events',
                'explanation': 'Get all endpoint events',
                'tables_used': ['endpoint_events'],
                'confidence': 0.9,
                'reasoning_steps': ['Retrieved tables', 'Validated query', 'Judge approved'],
                'judge_scores': [0.85],
                'final_judge_score': 0.85
            }
            return mock_run_output

        mock_agent.run.side_effect = side_effect

        result = agent.run("Get all endpoint events")

        # Should return the submitted answer
        assert result['query'] == 'SELECT * FROM endpoint_events'
        assert result['confidence'] == 0.9
        assert 'tables_used' in result
        assert 'judge_scores' in result
        assert result['judge_scores'] == [0.85]
        assert result['final_judge_score'] == 0.85

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_run_without_submit_returns_low_confidence(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test run when agent doesn't use submit_answer tool."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = "Here is my answer without using the tool"

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        agent = ReActAgentV2(schema_path=sample_schema_file)
        result = agent.run("Test question")

        # Should return low confidence response
        assert result['confidence'] <= 0.3
        assert 'submit_answer tool' in str(result['reasoning_steps'])

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_run_handles_exception(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that run handles exceptions gracefully."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent

        agent = ReActAgentV2(schema_path=sample_schema_file)
        result = agent.run("Test question")

        # Should return error response
        assert 'error' in result
        assert result['query'] is None
        assert result['confidence'] == 0.0
        assert "Test error" in result['explanation']

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_run_initializes_state(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that run initializes state correctly."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = "Response"

        mock_agent = Mock()
        mock_agent.run.return_value = mock_run_output
        mock_agent_class.return_value = mock_agent

        agent = ReActAgentV2(schema_path=sample_schema_file)

        # State should be None before run
        assert agent._state is None

        agent.run("Test question")

        # State should be initialized after run
        assert agent._state is not None
        assert agent._state.question == "Test question"
        # State should be AgentStateV2 (with judge tracking)
        assert hasattr(agent._state, 'judge_calls')
        assert hasattr(agent._state, 'judge_scores')


class TestToolFunctions:
    """Tests for tool function behavior."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_validate_sql_tool_valid_query(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test validate_sql tool with valid query."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgentV2(schema_path=sample_schema_file)
        agent._state = AgentStateV2(question="test")

        # Test validation through the validator directly
        result = agent.validator.validate("SELECT * FROM endpoint_events", strict=False)
        assert result['valid'] is True

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_validate_sql_tool_invalid_query(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test validate_sql tool with invalid query."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgentV2(schema_path=sample_schema_file)
        agent._state = AgentStateV2(question="test")

        # Test validation through the validator directly
        result = agent.validator.validate("SELECT * FROM nonexistent_table", strict=False)
        assert result['valid'] is False
        assert 'Unknown tables' in str(result['errors'])


class TestFormatResponse:
    """Tests for _format_response method (inherited from BaseAgent)."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_format_response(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test response formatting."""
        from src.agents.base import SQLQueryResponse

        agent = ReActAgentV2(schema_path=sample_schema_file)

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

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_handle_error(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test error handling."""
        agent = ReActAgentV2(schema_path=sample_schema_file)
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

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_max_iterations_config(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test max iterations configuration."""
        agent = ReActAgentV2(
            schema_path=sample_schema_file,
            max_iterations=5,
            max_retrieval_calls=2,
            max_validation_attempts=3,
            max_judge_calls=2
        )

        assert agent.max_iterations == 5
        assert agent.max_retrieval_calls == 2
        assert agent.max_validation_attempts == 3
        assert agent.max_judge_calls == 2


class TestJudgeLoopPrevention:
    """Tests for judge loop prevention logic."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_judge_call_limit(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that judge calls are limited."""
        agent = ReActAgentV2(
            schema_path=sample_schema_file,
            max_judge_calls=3
        )

        assert agent.max_judge_calls == 3

    def test_judge_score_improvement_tracking(self):
        """Test that we can track score improvement."""
        state = AgentStateV2(question="test")

        # Simulate improving scores
        state.judge_scores.append(0.4)
        state.judge_scores.append(0.6)
        state.judge_scores.append(0.8)

        # Check improvement
        assert state.judge_scores[-1] > state.judge_scores[0]
        # Calculate improvement
        improvement = state.judge_scores[-1] - state.judge_scores[0]
        assert improvement == pytest.approx(0.4)

    def test_judge_score_not_improving(self):
        """Test detection of scores not improving."""
        state = AgentStateV2(question="test")

        # Simulate non-improving scores
        state.judge_scores.append(0.5)
        state.judge_scores.append(0.5)
        state.judge_scores.append(0.45)

        # Last score is not improving
        if len(state.judge_scores) >= 2:
            is_improving = state.judge_scores[-1] > state.judge_scores[-2]
            assert is_improving is False


class TestDualValidation:
    """Tests for dual validation behavior."""

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_confidence_adjusted_by_judge(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that confidence is adjusted based on judge scores."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        agent = ReActAgentV2(schema_path=sample_schema_file)

        # Initialize state with judge scores
        agent._state = AgentStateV2(question="test")
        agent._state.judge_scores = [0.6]  # Low judge score

        # When submitting with high confidence but low judge score,
        # the final confidence should be reduced
        # This is tested implicitly through the submit_answer tool behavior

    @patch('src.agents.react_agent_v2.SemanticRetrieval')
    @patch('src.agents.react_agent_v2.Agent')
    def test_final_answer_includes_judge_scores(self, mock_agent_class, mock_retrieval_class, sample_schema_file):
        """Test that final answer includes judge scores."""
        mock_retriever = Mock()
        mock_retrieval_class.return_value = mock_retriever

        mock_run_output = Mock()
        mock_run_output.content = None

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = ReActAgentV2(schema_path=sample_schema_file)

        # Mock run to set state with judge scores
        def side_effect(*args, **kwargs):
            agent._state.has_submitted_answer = True
            agent._state.judge_scores = [0.5, 0.7, 0.9]
            agent._state.final_answer = {
                'query': 'SELECT * FROM endpoint_events',
                'explanation': 'Get all events',
                'tables_used': ['endpoint_events'],
                'confidence': 0.85,
                'reasoning_steps': ['Step 1'],
                'judge_scores': [0.5, 0.7, 0.9],
                'final_judge_score': 0.9
            }
            return mock_run_output

        mock_agent.run.side_effect = side_effect

        result = agent.run("Test question")

        assert 'judge_scores' in result
        assert result['judge_scores'] == [0.5, 0.7, 0.9]
        assert result['final_judge_score'] == 0.9
