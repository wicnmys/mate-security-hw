"""Unified SQL query agent with pluggable retrieval strategies."""

import logging
from typing import Dict, Any, Optional
from agno.agent import Agent
from agno.models.anthropic import Claude
from dotenv import load_dotenv

from src.agents.base import BaseAgent, SQLQueryResponse, Retriever
from src.utils.schema_loader import load_schemas, format_schema_for_llm
from src.utils.validator import SQLValidator
from src.constants import DEFAULT_TOP_K_TABLES, DEFAULT_CACHE_TTL_SECONDS, DEFAULT_LLM_MODEL

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SQLQueryAgent(BaseAgent):
    """
    Unified SQL query generation agent with pluggable retrieval.

    Uses dependency injection for the retrieval strategy, allowing
    seamless switching between semantic, keyword, or custom retrievers.
    """

    def __init__(
        self,
        schema_path: str,
        retriever: Retriever,
        model: str = DEFAULT_LLM_MODEL,
        top_k_tables: int = DEFAULT_TOP_K_TABLES,
        validator: Optional[SQLValidator] = None
    ):
        """
        Initialize SQL query agent.

        Args:
            schema_path: Path to schema JSON file or directory
            retriever: Retrieval strategy implementing the Retriever protocol
            model: LLM model to use (default: claude-sonnet-4-5)
            top_k_tables: Number of tables to retrieve for context
            validator: Optional custom validator (created from schemas if not provided)
        """
        # Load schemas
        self.schemas = load_schemas(schema_path)
        self.top_k_tables = top_k_tables

        # Use injected retriever
        self.retriever = retriever

        # Initialize validator (use provided or create new)
        self.validator = validator if validator else SQLValidator(schemas=self.schemas)

        # Initialize Agno agent with prompt caching enabled
        self.agent = Agent(
            name="sql_query_generator",
            model=Claude(
                id=model,
                cache_system_prompt=True,
                cache_ttl=DEFAULT_CACHE_TTL_SECONDS
            ),
            instructions=self._get_instructions(),
            output_schema=SQLQueryResponse,
            markdown=True
        )

    def _get_instructions(self) -> str:
        """Get instructions for the SQL generation agent."""
        return """You are an expert SQL query generator for a security events database.

Your task is to convert natural language questions into accurate, optimized SQL queries.

**Process:**
1. Analyze the user's question to understand what data they need
2. Review the provided table schemas to identify relevant tables and fields
3. Determine if joins are needed between multiple tables
4. Construct a syntactically correct SQL query
5. Provide a clear explanation of what the query does
6. Assign a confidence score based on query complexity and ambiguity

**Guidelines:**
- Use PostgreSQL/MySQL syntax
- Always qualify fields with table names when using JOINs
- Use appropriate WHERE clauses for filtering
- Include ORDER BY and LIMIT when relevant (e.g., "recent", "top N")
- For time-based queries, use timestamp fields appropriately
- Choose the most efficient query structure
- Handle ambiguous questions by making reasonable assumptions and noting them

**Confidence Scoring:**
- 0.9-1.0: Clear question, obvious table/field selection, simple query
- 0.7-0.9: Some ambiguity resolved with reasonable assumptions
- 0.5-0.7: Multiple interpretations possible, chose most likely
- Below 0.5: High uncertainty, may need clarification

**Common Security Query Patterns:**
- "high severity" → WHERE severity IN ('high', 'critical')
- "last 24 hours" → WHERE timestamp >= NOW() - INTERVAL '24 hours'
- "failed logins" → authentication_events WHERE status = 'failure'
- "top N" → ORDER BY ... LIMIT N
- "suspicious" → Look for threat-related fields

Return your response in the structured format with query, explanation, tables_used, confidence, and reasoning_steps."""

    def run(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question

        Returns:
            Dictionary with query, explanation, tables_used, confidence, etc.
        """
        try:
            # Step 1: Retrieve relevant tables
            relevant_tables = self.retriever.get_top_k(
                question=question,
                k=self.top_k_tables
            )

            # Step 2: Format schemas for LLM context
            schema_context = self._build_schema_context(relevant_tables)

            # Step 3: Generate SQL using Agno agent
            run_output = self.agent.run(
                f"""Question: {question}

Available Tables:
{schema_context}

Generate a SQL query to answer this question."""
            )

            # Extract the structured response from RunOutput
            response = run_output.content

            # Step 4: Validate the generated query
            validation = self.validator.validate(response.query, strict=False)

            # Add validation warnings to response if any
            if validation['warnings']:
                if not response.reasoning_steps:
                    response.reasoning_steps = []
                response.reasoning_steps.append(
                    f"Validation warnings: {'; '.join(validation['warnings'])}"
                )

            # Adjust confidence if validation failed
            if not validation['valid']:
                response.confidence = min(response.confidence, 0.5)
                response.reasoning_steps.append(
                    f"Validation errors: {'; '.join(validation['errors'])}"
                )

            return self._format_response(response)

        except Exception as e:
            logger.debug("Error generating SQL query for question: %s", question, exc_info=True)
            return self._handle_error(question, e)

    def _build_schema_context(self, relevant_tables: list[Dict[str, Any]]) -> str:
        """
        Build formatted schema context for LLM.

        Args:
            relevant_tables: List of retrieved table information

        Returns:
            Formatted string with table schemas
        """
        context_parts = []

        for i, table_info in enumerate(relevant_tables, 1):
            table_name = table_info['table_name']
            schema = table_info['schema']
            score = table_info['score']

            formatted = format_schema_for_llm(table_name, schema, max_fields=30)
            context_parts.append(f"{i}. {formatted}\n   (Relevance: {score:.3f})\n")

        return "\n".join(context_parts)

    def explain_retrieval(self, question: str, k: int = DEFAULT_TOP_K_TABLES) -> Dict[str, Any]:
        """
        Explain which tables were retrieved and why (useful for debugging).

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
