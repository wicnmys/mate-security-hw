"""Base class for SQL evaluation judges."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseJudge(ABC):
    """Abstract base class for SQL evaluation judges.

    All judges must implement the evaluate method and provide
    a unique identifier for tracking which judge was used.
    """

    judge_id: str  # e.g., "correctness_v1", "categorical_v1", "integrity_v1"
    model_name: str

    @abstractmethod
    def evaluate(
        self,
        question: str,
        reference_sql: str,
        generated_sql: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a generated SQL query against a reference.

        Args:
            question: Natural language question
            reference_sql: Reference SQL query (ground truth)
            generated_sql: Generated SQL query to evaluate
            **kwargs: Additional context (e.g., expected_behavior for integrity tests)

        Returns:
            Dictionary with evaluation results (structure depends on judge type)
        """
        pass

    @property
    def identifier(self) -> str:
        """Get unique identifier for this judge configuration.

        Returns:
            String identifier combining model and judge version,
            e.g., "claude-sonnet-4-5_correctness_v1"
        """
        return f"{self.model_name}_{self.judge_id}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(identifier={self.identifier})"
