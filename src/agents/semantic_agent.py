"""Semantic agent using embedding-based retrieval and Agno framework."""

import os
from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from dotenv import load_dotenv

from src.agents.base import BaseAgent, SQLQueryResponse
from src.retrieval.semantic_retrieval import SemanticRetrieval
from src.utils.schema_loader import load_schemas, format_schema_for_llm
from src.utils.validator import SQLValidator


# Load environment variables
load_dotenv()


class SemanticAgent(BaseAgent):
    """
    SQL query generation agent using semantic retrieval.

    Uses embedding-based similarity to find relevant tables,
    then generates SQL using Agno framework with Claude/GPT-4.
    """

    def __init__(
        self,
        schema_path: str,
        model: str = "claude-sonnet-4-5",
        embedding_cache_path: str = "embeddings_cache/table_embeddings.pkl",
        top_k_tables: int = 5
    ):
        """
        Initialize semantic agent.

        Args:
            schema_path: Path to schema JSON file or directory
            model: LLM model to use (default: Claude 3.5 Sonnet)
            embedding_cache_path: Path to cache embeddings
            top_k_tables: Number of tables to retrieve for context
        """
        # Load schemas
        self.schemas = load_schemas(schema_path)
        self.top_k_tables = top_k_tables

        # Initialize retrieval
        self.retriever = SemanticRetrieval(
            schemas=self.schemas,
            cache_path=embedding_cache_path
        )

        # Initialize validator
        self.validator = SQLValidator(schemas=self.schemas)

        # Initialize Agno agent with prompt caching enabled
        self.agent = Agent(
            name="sql_query_generator",
            model=Claude(
                id=model,
                cache_system_prompt=True,  # Cache instructions (never change)
                cache_tool_definitions=True,  # Cache output schema (never changes)
                cache_ttl="1h"  # Cache for 1 hour
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
            # Step 1: Retrieve relevant tables using semantic search
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

    def explain_retrieval(self, question: str, k: int = 5) -> Dict[str, Any]:
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
