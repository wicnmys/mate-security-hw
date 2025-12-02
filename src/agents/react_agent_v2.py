"""ReAct agent with retrieval, validation, and LLM-as-judge tools for SQL generation."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class LLMJudgeOutput(BaseModel):
    """Output from LLM-as-judge evaluation."""
    is_correct: bool = Field(description="Whether query correctly answers question")
    correctness_score: float = Field(description="Score from 0.0 to 1.0", ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list, description="List of semantic issues found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    reasoning: str = Field(description="Judge's reasoning")


@dataclass
class AgentStateV2:
    """Track agent state across iterations with judge tracking."""
    question: str
    iteration: int = 0
    retrieved_tables: List[str] = field(default_factory=list)
    retrieved_table_schemas: List[Dict[str, Any]] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    judge_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    has_submitted_answer: bool = False
    final_answer: Optional[Dict[str, Any]] = None
    retrieval_calls: int = 0
    validation_attempts: int = 0
    judge_calls: int = 0
    judge_scores: List[float] = field(default_factory=list)


class ReActAgentV2(BaseAgent):
    """
    ReAct (Reasoning + Acting) agent with dual validation for SQL query generation.

    Extends ReActAgent with an LLM-as-judge tool that provides semantic validation
    in addition to structural validation. This creates a two-tier validation system:
    1. Structural Validation (Tool): Syntax, schema, field existence
    2. Semantic Validation (LLM Judge): Does the query actually answer the question?
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
        max_validation_attempts: int = 4,
        max_judge_calls: int = 3,
        judge_model: str = "claude-sonnet-4-5"
    ):
        """
        Initialize ReAct agent with dual validation.

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
            max_judge_calls: Maximum times to call LLM judge (expensive)
            judge_model: Model to use for LLM judge evaluations
        """
        # Load schemas
        self.schemas = load_schemas(schema_path)
        self.top_k_tables = top_k_tables
        self.model_name = model
        self.retrieval_type = retrieval_type
        self.judge_model = judge_model

        # Configuration for loop control
        self.max_iterations = max_iterations
        self.max_retrieval_calls = max_retrieval_calls
        self.max_validation_attempts = max_validation_attempts
        self.max_judge_calls = max_judge_calls

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

        # Initialize LLM judge agent
        self._judge_agent = Agent(
            name="sql_semantic_judge",
            model=Claude(
                id=judge_model,
                cache_system_prompt=True,
                cache_ttl=DEFAULT_CACHE_TTL_SECONDS
            ),
            instructions=self._get_judge_instructions(),
            output_schema=LLMJudgeOutput,
            markdown=False
        )

        # State will be initialized per-run
        self._state: Optional[AgentStateV2] = None

        # Create tools that reference self
        self._tools = self._create_tools()

        # Initialize Agno agent with tools
        self.agent = Agent(
            name="sql_react_agent_v2",
            model=Claude(
                id=model,
                cache_system_prompt=True,
                cache_ttl=DEFAULT_CACHE_TTL_SECONDS
            ),
            instructions=self._get_instructions(),
            tools=self._tools,
            markdown=False
        )

    def _get_judge_instructions(self) -> str:
        """Get instructions for the LLM judge."""
        return """You are an expert SQL evaluator. Your task is to assess whether a generated SQL query correctly answers the original question.

**Evaluation Criteria:**
1. **Question Understanding**: Does the query address what the user actually asked?
2. **Table Selection**: Are the right tables being queried?
3. **Filtering/Conditions**: Are the WHERE clauses appropriate for the question?
4. **Columns**: Are the right columns selected?
5. **Aggregations**: Are GROUP BY, HAVING, COUNT, etc. used correctly when needed?
6. **Completeness**: Does the query capture all aspects of the question?

**Scoring Rubric:**
- **1.0**: Perfect, fully answers the question
- **0.9**: Correct with minor cosmetic differences
- **0.8**: Correct approach, minor issues (missing ORDER BY, suboptimal but valid)
- **0.7**: Mostly correct, one significant issue (wrong filter value, missing condition)
- **0.5-0.6**: Partially correct (right tables, wrong aggregation or missing key filter)
- **0.3-0.4**: Wrong approach but related tables
- **0.0-0.2**: Completely wrong (wrong tables, doesn't answer question)

Return a structured evaluation with:
- **is_correct**: True if score >= 0.7
- **correctness_score**: Float between 0 and 1
- **issues**: List of specific problems found
- **suggestions**: Actionable improvements
- **reasoning**: Clear explanation of your assessment

Be strict but fair. Focus on whether the query would return the data the user actually wants."""

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
                # Use the configured retriever
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

                status = "VALID" if result['valid'] else "INVALID"
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
        def llm_judge_evaluate(
            original_question: str,
            sql_query: str,
            table_schemas: list,
            agent_explanation: str
        ) -> dict:
            """
            Evaluate if the SQL query correctly answers the original question using LLM-as-judge.

            This provides semantic validation beyond syntax checking.
            Use AFTER structural validation passes.

            Args:
                original_question: The user's original question
                sql_query: The generated SQL query to evaluate
                table_schemas: List of table schema dictionaries used in the query
                agent_explanation: Your explanation of what the query does

            Returns:
                Dictionary with:
                - is_correct: Boolean indicating if query answers the question
                - correctness_score: Float 0.0-1.0
                - issues: List of semantic issues found
                - suggestions: List of improvements
                - reasoning: Judge's explanation
            """
            if agent_self._state is None:
                return {"error": "Agent state not initialized"}

            # Increment judge call counter
            agent_self._state.judge_calls += 1
            agent_self._state.iteration += 1

            # Check judge limit
            if agent_self._state.judge_calls > agent_self.max_judge_calls:
                # Check if we have any previous scores
                if agent_self._state.judge_scores:
                    last_score = agent_self._state.judge_scores[-1]
                    return {
                        "error": f"Maximum judge calls ({agent_self.max_judge_calls}) exceeded.",
                        "is_correct": last_score >= 0.7,
                        "correctness_score": last_score,
                        "issues": ["Judge call limit reached - using last score"],
                        "suggestions": ["Submit your best effort with submit_answer"],
                        "reasoning": f"Last score was {last_score:.2f}. Please submit your answer now."
                    }
                return {
                    "error": f"Maximum judge calls ({agent_self.max_judge_calls}) exceeded.",
                    "is_correct": False,
                    "correctness_score": 0.5,
                    "issues": ["Judge call limit reached"],
                    "suggestions": ["Submit your best effort with submit_answer"],
                    "reasoning": "Unable to evaluate further. Please submit your answer."
                }

            try:
                # Format schemas for the judge
                schemas_formatted = json.dumps(table_schemas, indent=2) if table_schemas else "No schemas provided"

                judge_prompt = f"""Evaluate this SQL query:

## Original Question
{original_question}

## Generated SQL Query
```sql
{sql_query}
```

## Agent's Explanation
{agent_explanation}

## Table Schemas Used
{schemas_formatted}

Assess whether this SQL query correctly answers the original question. Be thorough but fair."""

                # Call the judge agent
                run_output = agent_self._judge_agent.run(judge_prompt)
                evaluation = run_output.content

                # Record the score
                agent_self._state.judge_scores.append(evaluation.correctness_score)

                # Store judge result
                judge_result = {
                    'is_correct': evaluation.is_correct,
                    'correctness_score': evaluation.correctness_score,
                    'issues': evaluation.issues,
                    'suggestions': evaluation.suggestions,
                    'reasoning': evaluation.reasoning
                }
                agent_self._state.judge_results.append(judge_result)

                # Format observation for agent
                status = "CORRECT" if evaluation.is_correct else "NEEDS IMPROVEMENT"
                agent_self._state.reasoning_trace.append(
                    f"Judge evaluation: {status} (score: {evaluation.correctness_score:.2f})"
                )

                return judge_result

            except Exception as e:
                logger.exception("Error in LLM judge evaluation: %s", e)
                return {
                    "error": str(e),
                    "is_correct": False,
                    "correctness_score": 0.0,
                    "issues": [f"Judge evaluation failed: {str(e)}"],
                    "suggestions": ["Try structural validation first"],
                    "reasoning": f"Evaluation error: {str(e)}"
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

            # Adjust confidence based on validation and judge results
            adjusted_confidence = confidence

            # If we have judge scores, factor them in
            if agent_self._state.judge_scores:
                last_judge_score = agent_self._state.judge_scores[-1]
                # Weight confidence with judge score
                adjusted_confidence = min(confidence, (confidence + last_judge_score) / 2)

            if not final_validation['valid']:
                adjusted_confidence = min(adjusted_confidence, 0.5)
                reasoning_steps = list(reasoning_steps) + [
                    f"Note: Final query has validation errors: {'; '.join(final_validation['errors'])}"
                ]
            elif final_validation['warnings']:
                adjusted_confidence = min(adjusted_confidence, 0.8)
                reasoning_steps = list(reasoning_steps) + [
                    f"Note: Final query has warnings: {'; '.join(final_validation['warnings'])}"
                ]

            # Add judge information to reasoning
            if agent_self._state.judge_scores:
                reasoning_steps = list(reasoning_steps) + [
                    f"Judge scores: {agent_self._state.judge_scores}"
                ]

            # Store final answer
            agent_self._state.final_answer = {
                'query': sql_query,
                'explanation': explanation,
                'tables_used': tables_used,
                'confidence': adjusted_confidence,
                'reasoning_steps': reasoning_steps + agent_self._state.reasoning_trace,
                'judge_scores': agent_self._state.judge_scores.copy() if agent_self._state.judge_scores else [],
                'final_judge_score': agent_self._state.judge_scores[-1] if agent_self._state.judge_scores else None
            }

            agent_self._state.reasoning_trace.append(
                f"Submitted answer with confidence {adjusted_confidence:.2f}"
            )

            return f"Answer submitted successfully with confidence {adjusted_confidence:.2f}"

        return [retrieve_tables, validate_sql, llm_judge_evaluate, submit_answer]

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
        """Get instructions for the ReAct SQL generation agent with dual validation."""
        return """You are an expert SQL query generation agent with self-evaluation capabilities.

## Your Tools

You have access to four tools:

1. **retrieve_tables**: Search for relevant database tables
   - Use this to find tables relevant to the question
   - Can refine search with different queries

2. **validate_sql**: Validate SQL syntax and schema
   - Checks if query is syntactically correct
   - Verifies tables and fields exist
   - Fast, rule-based validation

3. **llm_judge_evaluate**: Semantic correctness evaluation
   - Evaluates if your query actually answers the question
   - Catches logical errors and misunderstandings
   - Provides suggestions for improvement
   - Use this AFTER structural validation passes

4. **submit_answer**: Submit your final answer
   - Only use when confident in your query
   - Ideally after both validations pass

## Dual Validation Strategy

Always validate in two stages:

```
Stage 1: Structural Validation (validate_sql)
- Syntax correct?
- Tables exist?
- Fields exist?
- No dangerous operations?

Stage 2: Semantic Validation (llm_judge_evaluate)
- Answers the question?
- Correct filters/conditions?
- Right interpretation?
- Complete solution?
```

## When to Use Each Validator

| Situation | Use validate_sql | Use llm_judge_evaluate |
|-----------|------------------|------------------------|
| First query attempt | Always first | If structural passes |
| After fixing syntax | Yes | Only if it passes |
| Unsure about logic | Not needed | Yes |
| Final check before submit | Quick sanity check | For confidence |

## ReAct Process with Dual Validation

**Thought**: Analyze the question, plan approach
**Action**: retrieve_tables(...)
**Observation**: Found relevant tables

**Thought**: Construct initial query
**Action**: validate_sql(sql_query="...")
**Observation**: Syntax valid

**Thought**: Syntax is good, but does it answer the question correctly?
**Action**: llm_judge_evaluate(question="...", sql_query="...", ...)
**Observation**: Score 0.6 - Missing time filter

**Thought**: Need to add time filter. Let me fix and re-validate.
**Action**: validate_sql(sql_query="...updated...")
**Observation**: Valid

**Action**: llm_judge_evaluate(...)
**Observation**: Score 0.95 - Correct!

**Thought**: Both validations pass with high score. Ready to submit.
**Action**: submit_answer(...)

## Confidence Scoring with Dual Validation

Your confidence should reflect BOTH validation results:

| Structural | Semantic (Judge) | Confidence |
|------------|------------------|------------|
| Pass | Score >= 0.9 | 0.9 - 1.0 |
| Pass | Score 0.7-0.9 | 0.7 - 0.9 |
| Pass | Score 0.5-0.7 | 0.5 - 0.7 |
| Pass | Score < 0.5 | 0.3 - 0.5 |
| Fail | Any | 0.0 - 0.3 |

## Loop Limits

- Maximum iterations: 10
- Maximum judge calls: 3 (expensive)
- If you can't get judge score > 0.7 after 3 attempts, submit best effort

## Guidelines

1. **Start with retrieval**: Always retrieve tables first to understand the schema
2. **Generate iteratively**: Create a query, validate structurally, then semantically
3. **Use judge feedback**: Fix issues the judge identifies before resubmitting
4. **Be thorough**: Check that all fields exist in the schema
5. **Know when to stop**: After 3 judge calls or if you can't improve, submit your best

## Important

- Always use structural validation before semantic validation
- The judge is expensive - don't waste calls on syntactically invalid queries
- Include all reasoning steps in your final answer
- If validation fails, try to fix the issues based on the suggestions"""

    def run(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question using ReAct loop with dual validation.

        Args:
            question: Natural language question

        Returns:
            Dictionary with query, explanation, tables_used, confidence, judge_scores, etc.
        """
        try:
            # Initialize state for this run
            self._state = AgentStateV2(question=question)

            # Run the agent with the question
            run_output = self.agent.run(
                f"""Generate a SQL query to answer this question:

Question: {question}

Remember to:
1. First use retrieve_tables to find relevant tables
2. Construct and validate your SQL query with validate_sql
3. Use llm_judge_evaluate to check if it answers the question correctly
4. Finally submit your answer with submit_answer

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
                    return {
                        'query': None,
                        'explanation': content[:500] if len(content) > 500 else content,
                        'tables_used': self._state.retrieved_tables,
                        'confidence': 0.3,
                        'reasoning_steps': self._state.reasoning_trace + [
                            "Warning: Agent did not use submit_answer tool"
                        ],
                        'judge_scores': self._state.judge_scores,
                        'final_judge_score': self._state.judge_scores[-1] if self._state.judge_scores else None
                    }

            # Fallback: return error response
            return {
                'query': None,
                'explanation': "Agent did not produce a valid answer",
                'tables_used': self._state.retrieved_tables,
                'confidence': 0.0,
                'reasoning_steps': self._state.reasoning_trace,
                'judge_scores': self._state.judge_scores,
                'final_judge_score': None
            }

        except Exception as e:
            logger.exception("Error in ReAct agent v2: %s", e)
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
