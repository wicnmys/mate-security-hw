"""Unit tests for the experiment configs module."""

import pytest

from experiments.configs.experiment_config import (
    ExperimentConfig,
    MAIN_EXPERIMENT,
    INTEGRITY_EXPERIMENT,
    CONSISTENCY_EXPERIMENT,
    get_experiment_config,
    list_experiment_types,
    register_experiment,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_config_creation(self):
        """Test creating an ExperimentConfig."""
        config = ExperimentConfig(
            name="test",
            description="Test experiment",
            test_cases_path="test/cases.json",
            default_judge="correctness",
            output_subfolder="test",
            supported_judges=["correctness", "categorical"]
        )

        assert config.name == "test"
        assert config.description == "Test experiment"
        assert config.test_cases_path == "test/cases.json"
        assert config.default_judge == "correctness"
        assert config.output_subfolder == "test"
        assert "correctness" in config.supported_judges

    def test_get_output_dir_default(self):
        """Test get_output_dir with default base."""
        config = ExperimentConfig(
            name="test",
            description="Test",
            test_cases_path="test.json",
            default_judge="correctness",
            output_subfolder="test_folder"
        )

        assert config.get_output_dir() == "experiments/results/test_folder"

    def test_get_output_dir_custom_base(self):
        """Test get_output_dir with custom base."""
        config = ExperimentConfig(
            name="test",
            description="Test",
            test_cases_path="test.json",
            default_judge="correctness",
            output_subfolder="test_folder"
        )

        assert config.get_output_dir("/custom/base") == "/custom/base/test_folder"


class TestPredefinedConfigs:
    """Tests for predefined experiment configurations."""

    def test_main_experiment_config(self):
        """Test MAIN_EXPERIMENT configuration."""
        assert MAIN_EXPERIMENT.name == "main"
        assert "generated_test_cases.json" in MAIN_EXPERIMENT.test_cases_path
        assert MAIN_EXPERIMENT.default_judge == "correctness"
        assert MAIN_EXPERIMENT.output_subfolder == "main"
        assert "correctness" in MAIN_EXPERIMENT.supported_judges

    def test_integrity_experiment_config(self):
        """Test INTEGRITY_EXPERIMENT configuration."""
        assert INTEGRITY_EXPERIMENT.name == "integrity"
        assert "integrity_test_cases.json" in INTEGRITY_EXPERIMENT.test_cases_path
        assert INTEGRITY_EXPERIMENT.default_judge == "integrity"
        assert INTEGRITY_EXPERIMENT.output_subfolder == "integrity"
        assert "integrity" in INTEGRITY_EXPERIMENT.supported_judges

    def test_consistency_experiment_config(self):
        """Test CONSISTENCY_EXPERIMENT configuration."""
        assert CONSISTENCY_EXPERIMENT.name == "consistency"
        assert CONSISTENCY_EXPERIMENT.default_judge == "correctness"
        assert CONSISTENCY_EXPERIMENT.output_subfolder == "consistency"


class TestGetExperimentConfig:
    """Tests for get_experiment_config function."""

    def test_get_main_config(self):
        """Test getting main experiment config."""
        config = get_experiment_config("main")
        assert config.name == "main"
        assert config == MAIN_EXPERIMENT

    def test_get_integrity_config(self):
        """Test getting integrity experiment config."""
        config = get_experiment_config("integrity")
        assert config.name == "integrity"
        assert config == INTEGRITY_EXPERIMENT

    def test_get_consistency_config(self):
        """Test getting consistency experiment config."""
        config = get_experiment_config("consistency")
        assert config.name == "consistency"
        assert config == CONSISTENCY_EXPERIMENT

    def test_get_unknown_config_raises(self):
        """Test getting unknown config raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_experiment_config("unknown_type")

        assert "unknown_type" in str(exc_info.value)
        assert "Valid types" in str(exc_info.value)


class TestListExperimentTypes:
    """Tests for list_experiment_types function."""

    def test_list_includes_main(self):
        """Test list includes main experiment type."""
        types = list_experiment_types()
        assert "main" in types

    def test_list_includes_integrity(self):
        """Test list includes integrity experiment type."""
        types = list_experiment_types()
        assert "integrity" in types

    def test_list_includes_consistency(self):
        """Test list includes consistency experiment type."""
        types = list_experiment_types()
        assert "consistency" in types

    def test_list_returns_list(self):
        """Test list_experiment_types returns a list."""
        types = list_experiment_types()
        assert isinstance(types, list)
        assert len(types) >= 3


class TestRegisterExperiment:
    """Tests for register_experiment function."""

    def test_register_new_experiment(self):
        """Test registering a new experiment type."""
        new_config = ExperimentConfig(
            name="custom_test_12345",
            description="Custom test experiment",
            test_cases_path="custom/test.json",
            default_judge="correctness",
            output_subfolder="custom"
        )

        # Register it
        register_experiment(new_config)

        # Should be retrievable
        retrieved = get_experiment_config("custom_test_12345")
        assert retrieved == new_config

        # Should be in list
        assert "custom_test_12345" in list_experiment_types()

    def test_register_duplicate_raises(self):
        """Test registering duplicate experiment type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_experiment(MAIN_EXPERIMENT)

        assert "already registered" in str(exc_info.value)
