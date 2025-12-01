"""ReAct agent with retrieval and validation tools for SQL generation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool
from dotenv import load_dotenv

from src.agents.base import BaseAgent, SQLQueryResponse
from src.retrieval.semantic_retrieval import SemanticRetrieval
from src.retrieval.keyword_retrieval import KeywordRetrieval
from src.utils.schema_loader import load_schemas, format_schema_for_llm
from src.utils.validator import SQLValidator
from src.constants import (
    DEFAULT_TOP_K_TABLES,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Track agent state across iterations."""
    question: str
    iteration: int = 0
    retrieved_tables: List[str] = field(default_factory=list)
    retrieved_table_schemas: List[Dict[str, Any]] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    has_submitted_answer: bool = False
    final_answer: Optional[Dict[str, Any]] = None
    retrieval_calls: int = 0
    validation_attempts: int = 0


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent for SQL query generation.

    Uses an iterative tool-use loop to:
    1. Retrieve relevant tables based on the question
    2. Generate and validate SQL queries
    3. Self-correct based on validation feedback
    4. Submit final answer with confidence score
    """

    def __init__(
        self,
        schema_path: str,
        model: str = DEFAULT_LLM_MODEL,
        top_k_tables: int = DEFAULT_TOP_K_TABLES,
        retrieval_type: str = "semantic",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cache_path: Optional[str] = "embeddings_cache/table_embeddings.npz",
        max_iterations: int = 10,
        max_retrieval_calls: int = 3,
        max_validation_attempts: int = 4
    ):
        """
        Initialize ReAct agent.

        Args:
            schema_path: Path to schema JSON file or directory
            model: LLM model to use (default: claude-sonnet-4-5)
            top_k_tables: Number of tables to retrieve for context
            retrieval_type: Type of retrieval: 'semantic' or 'keyword'
            embedding_model: Embedding model for semantic retrieval
            cache_path: Path to cache embeddings
            max_iterations: Maximum number of tool calls
            max_retrieval_calls: Maximum times to retrieve tables
            max_validation_attempts: Maximum times to validate same query
        """
        # Load schemas
        self.schemas = load_schemas(schema_path)
        self.top_k_tables = top_k_tables
        self.model_name = model
        self.retrieval_type = retrieval_type

        # Configuration for loop control
        self.max_iterations = max_iterations
        self.max_retrieval_calls = max_retrieval_calls
        self.max_validation_attempts = max_validation_attempts

        # Initialize retriever based on type
        if retrieval_type == "semantic":
            self.retriever = SemanticRetrieval(
                schemas=self.schemas,
                embedding_model=embedding_model,
                cache_path=cache_path
            )
        else:
            self.retriever = KeywordRetrieval(schemas=self.schemas)

        # Initialize validator
        self.validator = SQLValidator(schemas=self.schemas)

        # State will be initialized per-run
        self._state: Optional[AgentState] = None

        # Create tools that reference self
        self._tools = self._create_tools()

        # Initialize Agno agent with tools
        self.agent = Agent(
            name="sql_react_agent",
            model=Claude(
                id=model,
                cache_system_prompt=True,
                cache_ttl=DEFAULT_CACHE_TTL_SECONDS
            ),
            instructions=self._get_instructions(),
            tools=self._tools,
            markdown=False
        )

    def _create_tools(self) -> List:
        """Create tool functions that reference the agent's state."""
        agent_self = self

        @tool
        def retrieve_tables(query: str, top_k: int = 5, retrieval_type: str = "semantic") -> dict:
            """
            Retrieve relevant database tables based on search query.

            Args:
                query: Search query to find relevant tables
                top_k: Number of tables to retrieve (default: 5)
                retrieval_type: Type of retrieval - 'semantic' or 'keyword' (default: semantic)

            Returns:
                Dictionary with tables, scores, and total available
            """
            if agent_self._state is None:
                return {"error": "Agent state not initialized"}

            # Increment retrieval call counter
            agent_self._state.retrieval_calls += 1
            agent_self._state.iteration += 1

            # Check retrieval limit
            if agent_self._state.retrieval_calls > agent_self.max_retrieval_calls:
                return {
                    "error": f"Maximum retrieval calls ({agent_self.max_retrieval_calls}) exceeded. Use the tables you have.",
                    "tables": [],
                    "scores": [],
                    "total_available": len(agent_self.schemas)
                }

            try:
                # Use the configured retriever (semantic or keyword based on initialization)
                results = agent_self.retriever.get_top_k(query, k=top_k)

                # Store retrieved tables in state
                for result in results:
                    table_name = result['table_name']
                    if table_name not in agent_self._state.retrieved_tables:
                        agent_self._state.retrieved_tables.append(table_name)
                        agent_self._state.retrieved_table_schemas.append(result)

                # Format tables for response
                tables_info = []
                scores = []
                for result in results:
                    table_schema = result['schema']
                    formatted = format_schema_for_llm(result['table_name'], table_schema, max_fields=30)
                    tables_info.append({
                        'table_name': result['table_name'],
                        'description': table_schema.get('description', 'No description'),
                        'category': table_schema.get('category', 'unknown'),
                        'fields': [f['name'] for f in table_schema.get('fields', [])[:15]],
                        'formatted_schema': formatted
                    })
                    scores.append(result['score'])

                agent_self._state.reasoning_trace.append(
                    f"Retrieved {len(results)} tables for query: '{query}'"
                )

                return {
                    "tables": tables_info,
                    "scores": scores,
                    "total_available": len(agent_self.schemas)
                }

            except Exception as e:
                logger.exception("Error retrieving tables: %s", e)
                return {"error": str(e), "tables": [], "scores": [], "total_available": 0}

        @tool
        def validate_sql(sql_query: str, strict: bool = False) -> dict:
            """
            Validate a SQL query for syntax and schema correctness.

            Args:
                sql_query: The SQL query to validate
                strict: Use strict validation mode (fail on warnings)

            Returns:
                Dictionary with validation results including valid status, issues, tables used, and fields used
            """
            if agent_self._state is None:
                return {"error": "Agent state not initialized"}

            # Increment validation counter
            agent_self._state.validation_attempts += 1
            agent_self._state.iteration += 1

            # Store generated query
            agent_self._state.generated_queries.append(sql_query)

            try:
                # Perform validation
                result = agent_self.validator.validate(sql_query, strict=strict)

                # Extract additional info
                tables_used = agent_self.validator._extract_table_names(sql_query)
                has_dangerous, dangerous_ops = agent_self.validator.check_dangerous_operations(sql_query)

                # Build issues list
                issues = []
                for error in result.get('errors', []):
                    issues.append({
                        'type': 'error',
                        'message': error,
                        'suggestion': agent_self._get_suggestion(error)
                    })
                for warning in result.get('warnings', []):
                    issues.append({
                        'type': 'warning',
                        'message': warning,
                        'suggestion': agent_self._get_suggestion(warning)
                    })

                validation_result = {
                    'valid': result['valid'],
                    'issues': issues,
                    'tables_used': tables_used,
                    'is_dangerous': has_dangerous,
                    'dangerous_operations': dangerous_ops if has_dangerous else []
                }

                # Store validation result
                agent_self._state.validation_results.append(validation_result)

                status = "✅ Valid" if result['valid'] else "❌ Invalid"
                agent_self._state.reasoning_trace.append(
                    f"Validated query: {status} - {len(issues)} issues found"
                )

                return validation_result

            except Exception as e:
                logger.exception("Error validating SQL: %s", e)
                return {
                    "valid": False,
                    "issues": [{"type": "error", "message": str(e), "suggestion": None}],
                    "tables_used": [],
                    "is_dangerous": False
                }

        @tool
        def submit_answer(sql_query: str, explanation: str, confidence: float, reasoning_steps: list) -> str:
            """
            Submit the final SQL query answer. This ends the agent loop.

            Args:
                sql_query: The final SQL query
                explanation: Explanation of what the query does
                confidence: Confidence score between 0.0 and 1.0
                reasoning_steps: List of steps taken to arrive at the answer

            Returns:
                Confirmation message
            """
            if agent_self._state is None:
                return "Error: Agent state not initialized"

            agent_self._state.iteration += 1
            agent_self._state.has_submitted_answer = True

            # Extract tables used from the query
            tables_used = agent_self.validator._extract_table_names(sql_query)

            # Perform final validation
            final_validation = agent_self.validator.validate(sql_query, strict=False)

            # Adjust confidence based on validation
            adjusted_confidence = confidence
            if not final_validation['valid']:
                adjusted_confidence = min(confidence, 0.5)
                reasoning_steps = list(reasoning_steps) + [
                    f"Note: Final query has validation errors: {'; '.join(final_validation['errors'])}"
                ]
            elif final_validation['warnings']:
                adjusted_confidence = min(confidence, 0.8)
                reasoning_steps = list(reasoning_steps) + [
                    f"Note: Final query has warnings: {'; '.join(final_validation['warnings'])}"
                ]

            # Store final answer
            agent_self._state.final_answer = {
                'query': sql_query,
                'explanation': explanation,
                'tables_used': tables_used,
                'confidence': adjusted_confidence,
                'reasoning_steps': reasoning_steps + agent_self._state.reasoning_trace
            }

            agent_self._state.reasoning_trace.append(
                f"Submitted answer with confidence {adjusted_confidence:.2f}"
            )

            return f"Answer submitted successfully with confidence {adjusted_confidence:.2f}"

        return [retrieve_tables, validate_sql, submit_answer]

    def _get_suggestion(self, error_message: str) -> Optional[str]:
        """Get a suggestion for fixing a validation error."""
        error_lower = error_message.lower()

        if 'unknown tables' in error_lower:
            return "Check the table names against the retrieved schemas. Use retrieve_tables to find the correct table names."
        elif 'unknown fields' in error_lower:
            return "Verify field names exist in the table schema. Check the retrieved table schemas for valid field names."
        elif 'syntax' in error_lower:
            return "Review SQL syntax. Ensure proper SELECT, FROM, WHERE structure and balanced parentheses."
        elif 'dangerous' in error_lower:
            return "Avoid using DROP, DELETE, TRUNCATE, or other destructive operations."
        else:
            return None

    def _get_instructions(self) -> str:
        """Get instructions for the ReAct SQL generation agent."""
        return """You are an expert SQL query generation agent. Your task is to generate accurate SQL queries from natural language questions.

## Your Tools

You have access to three tools:

1. **retrieve_tables**: Search for relevant database tables
   - Use this to find tables that might be relevant to the question
   - You can adjust top_k to retrieve more or fewer tables
   - You can search again with different queries to refine results

2. **validate_sql**: Validate a SQL query
   - Use this to check if your query is syntactically correct
   - It will tell you about missing tables, invalid fields, and syntax errors
   - Use the suggestions to fix issues

3. **submit_answer**: Submit your final answer
   - Only use this when you are confident in your query
   - Include your reasoning steps and confidence score

## ReAct Process

Follow this reasoning pattern:

**Thought**: What do I need to do next? What have I learned?
**Action**: Choose a tool and provide inputs
**Observation**: Analyze the tool's output
**Thought**: What does this tell me? Do I need to adjust?
... (repeat as needed)
**Action**: submit_answer with final query

## Guidelines

1. **Start with retrieval**: Always retrieve tables first to understand the schema
2. **Generate iteratively**: Create a query, validate it, fix issues
3. **Refine search if needed**: If validation shows missing tables, retrieve more
4. **Be thorough**: Check that all fields exist in the schema
5. **Know when to stop**: Don't loop forever - if you can't find a solution after 3-4 attempts, submit your best effort with low confidence

## Confidence Scoring

- **0.9-1.0**: Query validated successfully, all tables/fields confirmed
- **0.7-0.9**: Query validated with minor warnings, made reasonable assumptions
- **0.5-0.7**: Some validation issues remain, multiple interpretations possible
- **0.3-0.5**: Significant uncertainty, key tables/fields may be wrong
- **0.0-0.3**: Unable to generate valid query, question may be unanswerable

## Example Reasoning Trace

Question: "Show me all high-severity endpoint events from today"

Thought: I need to find tables related to endpoint events. Let me search for relevant tables.
Action: retrieve_tables(query="endpoint events severity", top_k=3)
Observation: Found endpoint_events table with fields: id, timestamp, severity, event_type...

Thought: Good, I found the endpoint_events table. It has severity and timestamp fields. Let me construct a query.
Action: validate_sql(sql_query="SELECT * FROM endpoint_events WHERE severity = 'high' AND timestamp >= CURRENT_DATE")
Observation: Valid query. No issues found. Tables used: [endpoint_events].

Thought: The query validated successfully. I'm confident this is correct.
Action: submit_answer(sql_query="...", confidence=0.95, ...)

## Important

- Always use the tools to retrieve tables and validate queries before submitting
- If validation fails, try to fix the issues based on the suggestions
- Include all reasoning steps in your final answer"""

    def run(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question using ReAct loop.

        Args:
            question: Natural language question

        Returns:
            Dictionary with query, explanation, tables_used, confidence, etc.
        """
        try:
            # Initialize state for this run
            self._state = AgentState(question=question)

            # Run the agent with the question
            run_output = self.agent.run(
                f"""Generate a SQL query to answer this question:

Question: {question}

Remember to:
1. First use retrieve_tables to find relevant tables
2. Then construct and validate your SQL query
3. Finally submit your answer with submit_answer

Start by retrieving relevant tables for this question."""
            )

            # Check if we have a submitted answer
            if self._state.has_submitted_answer and self._state.final_answer:
                return self._state.final_answer

            # If agent didn't submit answer via tool, try to parse from response
            if hasattr(run_output, 'content') and run_output.content:
                content = run_output.content
                if isinstance(content, str):
                    # Agent may have generated a response without using submit_answer
                    # Create a low-confidence response
                    return {
                        'query': None,
                        'explanation': content[:500] if len(content) > 500 else content,
                        'tables_used': self._state.retrieved_tables,
                        'confidence': 0.3,
                        'reasoning_steps': self._state.reasoning_trace + [
                            "Warning: Agent did not use submit_answer tool"
                        ]
                    }

            # Fallback: return error response
            return {
                'query': None,
                'explanation': "Agent did not produce a valid answer",
                'tables_used': self._state.retrieved_tables,
                'confidence': 0.0,
                'reasoning_steps': self._state.reasoning_trace
            }

        except Exception as e:
            logger.exception("Error in ReAct agent: %s", e)
            return self._handle_error(question, e)

    def explain_retrieval(self, question: str, k: int = DEFAULT_TOP_K_TABLES) -> Dict[str, Any]:
        """
        Explain which tables would be retrieved for a question.

        Args:
            question: Natural language question
            k: Number of tables to retrieve

        Returns:
            Dictionary with retrieval information
        """
        relevant_tables = self.retriever.get_top_k(question=question, k=k)

        return {
            'question': question,
            'tables_retrieved': [
                {
                    'table': t['table_name'],
                    'category': t['schema'].get('category'),
                    'score': t['score'],
                    'description': t['schema'].get('description')
                }
                for t in relevant_tables
            ]
        }
