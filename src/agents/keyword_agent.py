"""Keyword agent using keyword-based retrieval and Agno framework."""

from typing import Dict, Any

from src.agents.sql_agent import SQLQueryAgent
from src.retrieval.keyword_retrieval import KeywordRetrieval
from src.utils.schema_loader import load_schemas
from src.constants import DEFAULT_TOP_K_TABLES, DEFAULT_LLM_MODEL


class KeywordAgent(SQLQueryAgent):
    """
    SQL query generation agent using keyword retrieval.

    Uses keyword-based matching to find relevant tables,
    then generates SQL using Agno framework with Claude/GPT-4.

    This is a convenience class that creates a SQLQueryAgent
    with a KeywordRetrieval strategy.
    """

    def __init__(
        self,
        schema_path: str,
        model: str = DEFAULT_LLM_MODEL,
        top_k_tables: int = DEFAULT_TOP_K_TABLES
    ):
        """
        Initialize keyword agent.

        Args:
            schema_path: Path to schema JSON file or directory
            model: LLM model to use (default: claude-sonnet-4-5)
            top_k_tables: Number of tables to retrieve for context
        """
        # Load schemas first to create retriever
        schemas = load_schemas(schema_path)

        # Initialize keyword retrieval
        retriever = KeywordRetrieval(schemas=schemas)

        # Initialize parent with the keyword retriever
        super().__init__(
            schema_path=schema_path,
            retriever=retriever,
            model=model,
            top_k_tables=top_k_tables
        )
