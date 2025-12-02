"""Tests for the refactored generate_report.py module."""

import pytest
from experiments.generate_report import (
    get_judge_type,
    get_judge_class,
    ReportGenerator,
)
from experiments.judges.correctness_judge import CorrectnessJudge
from experiments.judges.categorical_judge import CategoricalJudge
from experiments.judges.integrity_judge import IntegrityJudge


class TestGetJudgeType:
    """Tests for get_judge_type function."""

    def test_new_format_correctness(self):
        """Should return judge type from metadata.judge.type."""
        data = {
            "metadata": {
                "judge": {
                    "type": "correctness_v1"
                }
            }
        }
        assert get_judge_type(data, "/any/path.json") == "correctness_v1"

    def test_new_format_integrity(self):
        """Should return integrity type from metadata."""
        data = {
            "metadata": {
                "judge": {
                    "type": "integrity_v1"
                }
            }
        }
        assert get_judge_type(data, "/any/path.json") == "integrity_v1"

    def test_legacy_format_infer_integrity_from_path(self):
        """Should infer integrity from file path when metadata missing."""
        data = {"metadata": {"judge_model": "claude-sonnet-4-5"}}
        assert get_judge_type(data, "/results/integrity/file.json") == "integrity_v1"

    def test_legacy_format_infer_correctness_from_path(self):
        """Should default to correctness when path is not integrity."""
        data = {"metadata": {"judge_model": "claude-sonnet-4-5"}}
        assert get_judge_type(data, "/results/main/file.json") == "correctness_v1"

    def test_empty_metadata(self):
        """Should handle empty metadata gracefully."""
        data = {"metadata": {}}
        assert get_judge_type(data, "/results/main/file.json") == "correctness_v1"

    def test_missing_metadata(self):
        """Should handle missing metadata gracefully."""
        data = {}
        assert get_judge_type(data, "/results/main/file.json") == "correctness_v1"


class TestGetJudgeClass:
    """Tests for get_judge_class function."""

    def test_correctness_v1(self):
        """Should return CorrectnessJudge for correctness_v1."""
        assert get_judge_class("correctness_v1") == CorrectnessJudge

    def test_categorical_v1(self):
        """Should return CategoricalJudge for categorical_v1."""
        assert get_judge_class("categorical_v1") == CategoricalJudge

    def test_integrity_v1(self):
        """Should return IntegrityJudge for integrity_v1."""
        assert get_judge_class("integrity_v1") == IntegrityJudge

    def test_unknown_defaults_to_correctness(self):
        """Should default to CorrectnessJudge for unknown types."""
        assert get_judge_class("unknown_v1") == CorrectnessJudge

    def test_strips_version_suffix(self):
        """Should strip _v1 and _v2 suffixes."""
        assert get_judge_class("correctness_v2") == CorrectnessJudge
        assert get_judge_class("integrity") == IntegrityJudge


class TestJudgeGenerateReportSections:
    """Tests for generate_report_sections classmethods."""

    def test_correctness_returns_expected_sections(self):
        """CorrectnessJudge should return all expected sections."""
        results = [
            {"agent": "react", "correctness_score": 0.8, "complexity": "simple"},
            {"agent": "react", "correctness_score": 0.6, "complexity": "medium"},
        ]
        sections = CorrectnessJudge.generate_report_sections(results)

        assert "methodology" in sections
        assert "results_table" in sections
        assert "complexity_breakdown" in sections
        assert "category_breakdown" in sections
        assert "failure_analysis" in sections

    def test_correctness_methodology_contains_rubric(self):
        """Correctness methodology should contain scoring rubric."""
        sections = CorrectnessJudge.generate_report_sections([])
        assert "0.0 - 1.0" in sections["methodology"]
        assert "Perfectly correct" in sections["methodology"]

    def test_categorical_returns_expected_sections(self):
        """CategoricalJudge should return all expected sections."""
        results = [
            {"agent": "react", "score": 4, "category": "GOOD"},
            {"agent": "react", "score": 3, "category": "PARTIAL"},
        ]
        sections = CategoricalJudge.generate_report_sections(results)

        assert "methodology" in sections
        assert "results_table" in sections
        assert "distribution" in sections
        assert "failure_analysis" in sections

    def test_categorical_methodology_contains_categories(self):
        """Categorical methodology should contain category definitions."""
        sections = CategoricalJudge.generate_report_sections([])
        assert "PERFECT" in sections["methodology"]
        assert "GOOD" in sections["methodology"]
        assert "PARTIAL" in sections["methodology"]
        assert "POOR" in sections["methodology"]
        assert "WRONG" in sections["methodology"]

    def test_integrity_returns_expected_sections(self):
        """IntegrityJudge should return all expected sections."""
        results = [
            {"agent": "react", "passed": True, "confidence": 0.9, "integrity_type": "prompt_injection"},
            {"agent": "react", "passed": False, "confidence": 0.6, "integrity_type": "off_topic"},
        ]
        sections = IntegrityJudge.generate_report_sections(results)

        assert "methodology" in sections
        assert "results_table" in sections
        assert "category_breakdown" in sections
        assert "failure_analysis" in sections

    def test_integrity_methodology_contains_criteria(self):
        """Integrity methodology should contain pass/fail criteria."""
        sections = IntegrityJudge.generate_report_sections([])
        assert "PASS if" in sections["methodology"]
        assert "FAIL if" in sections["methodology"]
        assert "Prompt Injection" in sections["methodology"]


class TestReportIteratesAllSections:
    """Tests that report generation iterates all returned sections."""

    def test_no_required_keys(self):
        """Report should work with any section keys returned by judge."""
        results = [
            {"agent": "react", "correctness_score": 0.8},
        ]
        sections = CorrectnessJudge.generate_report_sections(results)

        # Just verify we can iterate over all sections
        all_content = []
        for section_name, content in sections.items():
            assert isinstance(section_name, str)
            assert isinstance(content, str)
            all_content.append(content)

        assert len(all_content) > 0

    def test_all_sections_have_content(self):
        """All returned sections should have non-empty content."""
        results = [
            {"agent": "react", "correctness_score": 0.8, "complexity": "simple"},
        ]
        sections = CorrectnessJudge.generate_report_sections(results)

        for section_name, content in sections.items():
            assert len(content) > 0, f"Section '{section_name}' should have content"


class TestMixedJudgesSeparateSections:
    """Tests for handling multiple judge types."""

    def test_different_judges_return_different_section_keys(self):
        """Different judge types should return different section structures."""
        correctness_sections = CorrectnessJudge.generate_report_sections([])
        categorical_sections = CategoricalJudge.generate_report_sections([])
        integrity_sections = IntegrityJudge.generate_report_sections([])

        # Each should have methodology
        assert "methodology" in correctness_sections
        assert "methodology" in categorical_sections
        assert "methodology" in integrity_sections

        # But methodology content differs
        assert "0.0 - 1.0" in correctness_sections["methodology"]
        assert "PERFECT" in categorical_sections["methodology"]
        assert "PASS if" in integrity_sections["methodology"]

    def test_correctness_has_complexity_breakdown(self):
        """Only correctness should have complexity_breakdown."""
        results = [{"agent": "react", "correctness_score": 0.8, "complexity": "simple"}]
        correctness_sections = CorrectnessJudge.generate_report_sections(results)

        assert "complexity_breakdown" in correctness_sections

    def test_categorical_has_distribution(self):
        """Only categorical should have distribution."""
        results = [{"agent": "react", "score": 4, "category": "GOOD"}]
        categorical_sections = CategoricalJudge.generate_report_sections(results)

        assert "distribution" in categorical_sections

    def test_integrity_has_category_breakdown_by_attack_type(self):
        """Integrity should break down by attack type."""
        results = [
            {"agent": "react", "passed": True, "confidence": 0.9, "integrity_type": "prompt_injection"},
        ]
        integrity_sections = IntegrityJudge.generate_report_sections(results)

        assert "category_breakdown" in integrity_sections
        assert "Attack Type" in integrity_sections["category_breakdown"]


class TestFailureAnalysis:
    """Tests for failure analysis sections."""

    def test_correctness_failure_threshold(self):
        """Correctness failures should use score < 0.5 threshold."""
        results = [
            {"agent": "react", "correctness_score": 0.4, "question": "test question"},
            {"agent": "react", "correctness_score": 0.6, "question": "passing question"},
        ]
        sections = CorrectnessJudge.generate_report_sections(results)

        assert "1 failures" in sections["failure_analysis"]

    def test_categorical_failure_threshold(self):
        """Categorical failures should use score <= 2 threshold."""
        results = [
            {"agent": "react", "score": 2, "category": "POOR", "question": "test"},
            {"agent": "react", "score": 1, "category": "WRONG", "question": "test2"},
            {"agent": "react", "score": 3, "category": "PARTIAL", "question": "test3"},
        ]
        sections = CategoricalJudge.generate_report_sections(results)

        assert "2 failures" in sections["failure_analysis"]

    def test_integrity_failure_counts_not_passed(self):
        """Integrity failures should count passed=False."""
        results = [
            {"agent": "react", "passed": False, "confidence": 0.5, "integrity_type": "test"},
            {"agent": "react", "passed": False, "confidence": 0.6, "integrity_type": "test"},
            {"agent": "react", "passed": True, "confidence": 0.9, "integrity_type": "test"},
        ]
        sections = IntegrityJudge.generate_report_sections(results)

        assert "2 failures" in sections["failure_analysis"]
