"""Base agent class for SQL query generation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Protocol, runtime_checkable
from pydantic import BaseModel, Field


@runtime_checkable
class Retriever(Protocol):
    """Protocol for table retrieval strategies."""

    def get_top_k(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant tables for a question.

        Args:
            question: Natural language question
            k: Number of tables to retrieve

        Returns:
            List of table schemas with metadata
        """
        ...


class SQLQueryResponse(BaseModel):
    """Structured response format for SQL query generation."""

    query: str = Field(description="Generated SQL query")
    explanation: str = Field(description="Natural language explanation of the query")
    tables_used: list[str] = Field(description="List of tables referenced in the query")
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning process (optional)"
    )


class BaseAgent(ABC):
    """
    Abstract base class for SQL query generation agents.

    All agent variants should inherit from this class and implement
    the run() method.
    """

    @abstractmethod
    def run(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question

        Returns:
            Dictionary containing query, explanation, tables_used, confidence
        """
        pass

    def _format_response(self, response: SQLQueryResponse) -> Dict[str, Any]:
        """
        Format agent response as dictionary.

        Args:
            response: Pydantic model response

        Returns:
            Dictionary representation
        """
        return response.model_dump()

    def _handle_error(self, question: str, error: Exception) -> Dict[str, Any]:
        """
        Handle errors and return formatted error response.

        Args:
            question: Original question
            error: Exception that occurred

        Returns:
            Error response dictionary
        """
        return {
            'query': None,
            'explanation': f"Error processing question: {str(error)}",
            'tables_used': [],
            'confidence': 0.0,
            'reasoning_steps': [],
            'error': str(error)
        }
