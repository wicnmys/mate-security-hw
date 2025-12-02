"""Base class for SQL evaluation judges."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseJudge(ABC):
    """Abstract base class for SQL evaluation judges.

    All judges must implement the evaluate method and provide
    a unique identifier for tracking which judge was used.

    Each judge also generates its own report sections via the
    generate_report_sections classmethod.
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

    @classmethod
    @abstractmethod
    def generate_report_sections(cls, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate judge-specific report sections.

        Each judge type returns different sections with its own methodology
        and result formatting. This is a classmethod so it can be called
        without instantiating the judge (which requires LLM initialization).

        Args:
            results: List of result dictionaries that used this judge.
                    Each result should have the evaluation fields this
                    judge produces (e.g., 'correctness_score' for CorrectnessJudge).

        Returns:
            Dict mapping section_name -> markdown_content.
            Section names are arbitrary and will all be included in the report.
            Common sections include: 'methodology', 'results_table',
            'breakdown', 'failure_analysis'.
        """
        pass
