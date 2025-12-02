"""Experiment configuration definitions."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for an experiment type.

    Defines the test cases, default judge, and output location
    for a specific type of experiment.
    """
    name: str  # e.g., "main", "integrity", "consistency"
    description: str
    test_cases_path: str  # Path to test cases JSON file
    default_judge: str  # Default judge type to use
    output_subfolder: str  # Subfolder under experiments/results/
    supported_judges: List[str] = field(default_factory=list)  # List of supported judge types

    def get_output_dir(self, base_dir: str = "experiments/results") -> str:
        """Get the full output directory path."""
        return f"{base_dir}/{self.output_subfolder}"


# Pre-defined experiment configurations
MAIN_EXPERIMENT = ExperimentConfig(
    name="main",
    description="Main SQL generation experiment with standard queries",
    test_cases_path="experiments/test_cases/generated_test_cases.json",
    default_judge="correctness",
    output_subfolder="main",
    supported_judges=["correctness", "categorical"]
)

INTEGRITY_EXPERIMENT = ExperimentConfig(
    name="integrity",
    description="Security/integrity testing with adversarial inputs",
    test_cases_path="experiments/test_cases/integrity_test_cases.json",
    default_judge="integrity",
    output_subfolder="integrity",
    supported_judges=["integrity", "correctness"]  # Can use correctness for comparison
)

CONSISTENCY_EXPERIMENT = ExperimentConfig(
    name="consistency",
    description="Consistency testing across multiple runs (future)",
    test_cases_path="experiments/test_cases/consistency_test_cases.json",
    default_judge="correctness",
    output_subfolder="consistency",
    supported_judges=["correctness", "categorical"]
)


# Registry of all experiment types
_EXPERIMENT_REGISTRY = {
    "main": MAIN_EXPERIMENT,
    "integrity": INTEGRITY_EXPERIMENT,
    "consistency": CONSISTENCY_EXPERIMENT,
}


def get_experiment_config(experiment_type: str) -> ExperimentConfig:
    """Get configuration for an experiment type.

    Args:
        experiment_type: Name of the experiment type (main, integrity, consistency)

    Returns:
        ExperimentConfig for the requested type

    Raises:
        ValueError: If experiment_type is not recognized
    """
    if experiment_type not in _EXPERIMENT_REGISTRY:
        valid_types = ", ".join(_EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            f"Valid types are: {valid_types}"
        )
    return _EXPERIMENT_REGISTRY[experiment_type]


def list_experiment_types() -> List[str]:
    """Get list of available experiment types."""
    return list(_EXPERIMENT_REGISTRY.keys())


def register_experiment(config: ExperimentConfig) -> None:
    """Register a new experiment configuration.

    Args:
        config: ExperimentConfig to register

    Raises:
        ValueError: If an experiment with the same name already exists
    """
    if config.name in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment type '{config.name}' already registered")
    _EXPERIMENT_REGISTRY[config.name] = config
