"""Semantic agent using embedding-based retrieval and Agno framework."""

from typing import Dict, Any

from src.agents.sql_agent import SQLQueryAgent
from src.retrieval.semantic_retrieval import SemanticRetrieval
from src.utils.schema_loader import load_schemas
from src.constants import DEFAULT_TOP_K_TABLES, DEFAULT_LLM_MODEL


class SemanticAgent(SQLQueryAgent):
    """
    SQL query generation agent using semantic retrieval.

    Uses embedding-based similarity to find relevant tables,
    then generates SQL using Agno framework with Claude/GPT-4.

    This is a convenience class that creates a SQLQueryAgent
    with a SemanticRetrieval strategy.
    """

    def __init__(
        self,
        schema_path: str,
        model: str = DEFAULT_LLM_MODEL,
        embedding_cache_path: str = "embeddings_cache/table_embeddings.npz",
        top_k_tables: int = DEFAULT_TOP_K_TABLES
    ):
        """
        Initialize semantic agent.

        Args:
            schema_path: Path to schema JSON file or directory
            model: LLM model to use (default: claude-sonnet-4-5)
            embedding_cache_path: Path to cache embeddings
            top_k_tables: Number of tables to retrieve for context
        """
        # Load schemas first to create retriever
        schemas = load_schemas(schema_path)

        # Initialize semantic retrieval
        retriever = SemanticRetrieval(
            schemas=schemas,
            cache_path=embedding_cache_path
        )

        # Initialize parent with the semantic retriever
        super().__init__(
            schema_path=schema_path,
            retriever=retriever,
            model=model,
            top_k_tables=top_k_tables
        )
