"""Unit tests for schema loader."""

import json
import pytest
import tempfile
from pathlib import Path
from src.utils.schema_loader import (
    load_schemas,
    _validate_schemas,
    get_table_description,
    format_schema_for_llm
)


@pytest.fixture
def sample_schema():
    """Sample schema for testing."""
    return {
        "test_table": {
            "description": "A test table for unit testing",
            "category": "test",
            "fields": [
                {"name": "id", "type": "integer", "description": "Primary key"},
                {"name": "name", "type": "string", "description": "User name"},
                {"name": "email", "type": "string", "description": "Email address"}
            ]
        }
    }


@pytest.fixture
def sample_schema_minimal():
    """Minimal valid schema without optional fields."""
    return {
        "minimal_table": {
            "fields": [
                {"name": "id", "type": "integer"},
                {"name": "value", "type": "string"}
            ]
        }
    }


class TestLoadSchemas:
    """Tests for load_schemas function."""

    def test_load_single_json_file(self, sample_schema, tmp_path):
        """Test loading schema from a single JSON file."""
        schema_file = tmp_path / "schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)

        result = load_schemas(str(schema_file))
        assert result == sample_schema

    def test_load_directory_of_json_files(self, sample_schema, sample_schema_minimal, tmp_path):
        """Test loading schemas from a directory of JSON files."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        # Create two schema files
        with open(schema_dir / "schema1.json", 'w') as f:
            json.dump(sample_schema, f)
        with open(schema_dir / "schema2.json", 'w') as f:
            json.dump(sample_schema_minimal, f)

        result = load_schemas(str(schema_dir))

        # Should contain both schemas
        assert "test_table" in result
        assert "minimal_table" in result
        assert result["test_table"] == sample_schema["test_table"]
        assert result["minimal_table"] == sample_schema_minimal["minimal_table"]

    def test_nonexistent_path_raises_error(self):
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Schema path does not exist"):
            load_schemas("/nonexistent/path")

    def test_non_json_file_raises_error(self, tmp_path):
        """Test that non-JSON file raises ValueError."""
        text_file = tmp_path / "schema.txt"
        text_file.write_text("not json")

        with pytest.raises(ValueError, match="Schema file must be JSON"):
            load_schemas(str(text_file))

    def test_empty_directory_raises_error(self, tmp_path):
        """Test that empty directory raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No JSON files found in directory"):
            load_schemas(str(empty_dir))

    def test_malformed_json_raises_error(self, tmp_path):
        """Test that malformed JSON raises error."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_schemas(str(malformed_file))


class TestValidateSchemas:
    """Tests for _validate_schemas function."""

    def test_valid_schema_passes(self, sample_schema):
        """Test that valid schema passes validation."""
        _validate_schemas(sample_schema)  # Should not raise

    def test_non_dict_schema_raises_error(self):
        """Test that non-dict schema raises ValueError."""
        with pytest.raises(ValueError, match="Schemas must be a dictionary"):
            _validate_schemas([])

    def test_schema_with_non_dict_value_raises_error(self):
        """Test that schema with non-dict value raises ValueError."""
        invalid_schema = {"table": "not a dict"}
        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_schemas(invalid_schema)

    def test_schema_missing_fields_raises_error(self):
        """Test that schema missing 'fields' raises ValueError."""
        invalid_schema = {
            "table": {
                "description": "Missing fields"
            }
        }
        with pytest.raises(ValueError, match="missing 'fields'"):
            _validate_schemas(invalid_schema)

    def test_schema_with_non_list_fields_raises_error(self):
        """Test that schema with non-list fields raises ValueError."""
        invalid_schema = {
            "table": {
                "fields": "not a list"
            }
        }
        with pytest.raises(ValueError, match="must be a list"):
            _validate_schemas(invalid_schema)

    def test_field_missing_name_or_type_raises_error(self):
        """Test that field missing name or type raises ValueError."""
        invalid_schema = {
            "table": {
                "fields": [
                    {"name": "id"}  # Missing type
                ]
            }
        }
        with pytest.raises(ValueError, match="missing 'name' or 'type'"):
            _validate_schemas(invalid_schema)

    def test_non_dict_field_raises_error(self):
        """Test that non-dict field raises ValueError."""
        invalid_schema = {
            "table": {
                "fields": ["not a dict"]
            }
        }
        with pytest.raises(ValueError, match="Field in table .* must be a dictionary"):
            _validate_schemas(invalid_schema)


class TestGetTableDescription:
    """Tests for get_table_description function."""

    def test_basic_description_with_descriptions(self, sample_schema):
        """Test basic description generation with field descriptions."""
        result = get_table_description("test_table", sample_schema["test_table"])

        assert "Table: test_table" in result
        assert "Category: test" in result
        assert "Description: A test table for unit testing" in result
        assert "Fields:" in result
        assert "id: Primary key" in result
        assert "name: User name" in result
        assert "email: Email address" in result

    def test_description_without_field_descriptions(self, sample_schema):
        """Test description without field descriptions."""
        result = get_table_description(
            "test_table",
            sample_schema["test_table"],
            include_descriptions=False
        )

        assert "Table: test_table" in result
        assert "Fields: id, name, email" in result
        assert "Primary key" not in result
        assert "User name" not in result

    def test_description_with_missing_optional_fields(self, sample_schema_minimal):
        """Test description with schema missing optional fields."""
        result = get_table_description("minimal_table", sample_schema_minimal["minimal_table"])

        assert "Table: minimal_table" in result
        assert "Fields:" in result
        assert "id" in result
        assert "value" in result
        # Should not have category or description lines
        assert "Category:" not in result
        assert "Description:" not in result

    def test_context_limit_without_model_raises_error(self, sample_schema):
        """Test that context_limit without embedding_model raises ValueError."""
        with pytest.raises(ValueError, match="embedding_model must be specified when context_limit is set"):
            get_table_description(
                "test_table",
                sample_schema["test_table"],
                context_limit=100
            )

    def test_context_limit_with_model_truncates(self):
        """Test that context_limit with model truncates description."""
        # Create a schema with many fields to force truncation
        large_schema = {
            "description": "A table with many fields",
            "category": "test",
            "fields": [
                {"name": f"field_{i}", "type": "string", "description": f"This is field number {i} with a long description"}
                for i in range(50)
            ]
        }

        # Use a small context limit to force truncation
        result = get_table_description(
            "large_table",
            large_schema,
            context_limit=100,
            embedding_model="multi-qa-mpnet-base-dot-v1"
        )

        assert "Table: large_table" in result
        # Should be truncated and show the truncation message
        assert "more fields" in result
        # Should not include all 50 fields
        assert "field_49" not in result

    def test_field_description_fallback_to_type(self):
        """Test that field description falls back to type when description missing."""
        schema = {
            "fields": [
                {"name": "id", "type": "integer"}  # No description
            ]
        }

        result = get_table_description("table", schema, include_descriptions=True)
        assert "id: integer" in result


class TestFormatSchemaForLLM:
    """Tests for format_schema_for_llm function."""

    def test_basic_formatting(self, sample_schema):
        """Test basic schema formatting for LLM."""
        result = format_schema_for_llm("test_table", sample_schema["test_table"])

        assert "Table: test_table" in result
        assert "Category: test" in result
        assert "Description: A test table for unit testing" in result
        assert "Fields:" in result
        assert "- id (integer): Primary key" in result
        assert "- name (string): User name" in result

    def test_max_fields_limit(self):
        """Test that max_fields limits the number of fields shown."""
        schema = {
            "description": "Test table",
            "fields": [{"name": f"field_{i}", "type": "string"} for i in range(100)]
        }

        result = format_schema_for_llm("table", schema, max_fields=10)

        # Should show first 10 fields
        assert "field_0" in result
        assert "field_9" in result
        # Should indicate more fields exist
        assert "90 more fields" in result
        # Should not show field_10 or later
        assert "field_10" not in result

    def test_formatting_without_optional_fields(self, sample_schema_minimal):
        """Test formatting with minimal schema."""
        result = format_schema_for_llm("minimal_table", sample_schema_minimal["minimal_table"])

        assert "Table: minimal_table" in result
        assert "Fields:" in result
        assert "- id (integer)" in result
        assert "- value (string)" in result
        # Should not have category or description
        assert "Category:" not in result or "Category: \n" in result

    def test_field_without_description(self):
        """Test field formatting without description."""
        schema = {
            "fields": [
                {"name": "id", "type": "integer"}  # No description
            ]
        }

        result = format_schema_for_llm("table", schema)
        assert "- id (integer)" in result
        # Should not have extra colon or description
        assert "- id (integer):" not in result or "- id (integer): " not in result.replace("- id (integer):\n", "")
