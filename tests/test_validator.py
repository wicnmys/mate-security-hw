"""Unit tests for SQL validator."""

import pytest
from src.utils.validator import SQLValidator


@pytest.fixture
def sample_schemas():
    """Sample schemas for testing."""
    return {
        "users": {
            "fields": [
                {"name": "user_id", "type": "integer"},
                {"name": "username", "type": "string"},
                {"name": "email", "type": "string"}
            ]
        },
        "events": {
            "fields": [
                {"name": "event_id", "type": "integer"},
                {"name": "user_id", "type": "integer"},
                {"name": "timestamp", "type": "datetime"},
                {"name": "severity", "type": "string"}
            ]
        }
    }


@pytest.fixture
def validator(sample_schemas):
    """Initialize validator with sample schemas."""
    return SQLValidator(sample_schemas)


@pytest.fixture
def validator_no_schemas():
    """Initialize validator without schemas."""
    return SQLValidator()


class TestIsValid:
    """Tests for is_valid method."""

    def test_valid_basic_select(self, validator):
        """Test validation of basic SELECT query."""
        query = "SELECT * FROM users"
        assert validator.is_valid(query) is True

    def test_valid_select_with_where(self, validator):
        """Test validation of SELECT with WHERE clause."""
        query = "SELECT username FROM users WHERE user_id = 1"
        assert validator.is_valid(query) is True

    def test_valid_select_with_join(self, validator):
        """Test validation of SELECT with JOIN."""
        query = "SELECT u.username, e.severity FROM users u JOIN events e ON u.user_id = e.user_id"
        assert validator.is_valid(query) is True

    def test_invalid_empty_query(self, validator):
        """Test that empty query is invalid."""
        assert validator.is_valid("") is False
        assert validator.is_valid("   ") is False

    def test_invalid_no_select(self, validator):
        """Test that query without SELECT is invalid."""
        query = "FROM users"
        assert validator.is_valid(query) is False

    def test_invalid_select_without_from(self, validator):
        """Test that SELECT without FROM is invalid."""
        query = "SELECT username"
        assert validator.is_valid(query) is False

    def test_invalid_unclosed_quote(self, validator):
        """Test that unclosed quotes make query invalid."""
        query = "SELECT * FROM users WHERE username = 'test"
        assert validator.is_valid(query) is False

    def test_invalid_unclosed_parenthesis(self, validator):
        """Test that unclosed parentheses make query invalid."""
        query = "SELECT * FROM users WHERE user_id IN (1, 2, 3"
        assert validator.is_valid(query) is False

    def test_valid_with_quotes(self, validator):
        """Test validation of query with properly closed quotes."""
        query = "SELECT * FROM users WHERE username = 'john'"
        assert validator.is_valid(query) is True

    def test_valid_with_parentheses(self, validator):
        """Test validation of query with properly balanced parentheses."""
        query = "SELECT * FROM users WHERE user_id IN (1, 2, 3)"
        assert validator.is_valid(query) is True


class TestCheckBalancedDelimiters:
    """Tests for _check_balanced_delimiters method."""

    def test_balanced_parentheses(self, validator):
        """Test balanced parentheses."""
        assert validator._check_balanced_delimiters("SELECT COUNT(*) FROM users") is True
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE id IN (1,2,3)") is True

    def test_unbalanced_parentheses(self, validator):
        """Test unbalanced parentheses."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE id IN (1,2,3") is False
        assert validator._check_balanced_delimiters("SELECT * FROM users)") is False

    def test_balanced_quotes(self, validator):
        """Test balanced quotes."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = 'test'") is True
        assert validator._check_balanced_delimiters('SELECT * FROM users WHERE name = "test"') is True

    def test_unbalanced_quotes(self, validator):
        """Test unbalanced quotes."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = 'test") is False
        assert validator._check_balanced_delimiters('SELECT * FROM users WHERE name = "test') is False

    def test_mixed_quotes(self, validator):
        """Test mixed single and double quotes."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = \"it's\"") is True
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = 'say \"hi\"'") is True

    def test_escaped_quotes(self, validator):
        """Test escaped quotes."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = 'it\\'s'") is True

    def test_parentheses_in_quotes(self, validator):
        """Test that parentheses inside quotes don't affect balance."""
        assert validator._check_balanced_delimiters("SELECT * FROM users WHERE name = '(test)'") is True


class TestTableExists:
    """Tests for table_exists method."""

    def test_existing_table(self, validator):
        """Test that existing table is recognized."""
        query = "SELECT * FROM users"
        assert validator.table_exists(query) is True

    def test_multiple_existing_tables(self, validator):
        """Test that multiple existing tables are recognized."""
        query = "SELECT * FROM users JOIN events ON users.user_id = events.user_id"
        assert validator.table_exists(query) is True

    def test_nonexistent_table(self, validator):
        """Test that nonexistent table is detected."""
        query = "SELECT * FROM nonexistent_table"
        assert validator.table_exists(query) is False

    def test_mixed_existing_nonexistent(self, validator):
        """Test query with both existing and nonexistent tables."""
        query = "SELECT * FROM users JOIN nonexistent ON users.id = nonexistent.id"
        assert validator.table_exists(query) is False

    def test_no_schemas_loaded(self, validator_no_schemas):
        """Test that validation passes when no schemas are loaded."""
        query = "SELECT * FROM anything"
        # Should return True when no schemas to validate against
        assert validator_no_schemas.table_exists(query) is True


class TestExtractTableNames:
    """Tests for _extract_table_names method."""

    def test_extract_from_simple_query(self, validator):
        """Test extracting table from simple SELECT."""
        query = "SELECT * FROM users"
        tables = validator._extract_table_names(query)
        assert 'users' in tables

    def test_extract_from_join(self, validator):
        """Test extracting tables from JOIN."""
        query = "SELECT * FROM users JOIN events ON users.user_id = events.user_id"
        tables = validator._extract_table_names(query)
        assert 'users' in tables
        assert 'events' in tables

    def test_extract_case_insensitive(self, validator):
        """Test that extraction is case-insensitive."""
        query = "SELECT * FROM USERS"
        tables = validator._extract_table_names(query)
        assert 'users' in tables

    def test_extract_no_duplicates(self, validator):
        """Test that duplicate table names are not included."""
        query = "SELECT * FROM users u1 JOIN users u2 ON u1.id = u2.parent_id"
        tables = validator._extract_table_names(query)
        assert tables.count('users') == 1

    def test_extract_from_multiple_joins(self, validator):
        """Test extracting from query with multiple JOINs."""
        query = """
            SELECT * FROM users
            JOIN events ON users.user_id = events.user_id
            JOIN sessions ON users.user_id = sessions.user_id
        """
        tables = validator._extract_table_names(query)
        assert len(tables) == 3
        assert 'users' in tables
        assert 'events' in tables
        assert 'sessions' in tables


class TestCheckDangerousOperations:
    """Tests for check_dangerous_operations method."""

    def test_safe_select(self, validator):
        """Test that SELECT is not flagged as dangerous."""
        query = "SELECT * FROM users"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is False
        assert len(ops) == 0

    def test_detect_drop(self, validator):
        """Test detection of DROP operation."""
        query = "DROP TABLE users"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'DROP' in ops

    def test_detect_delete(self, validator):
        """Test detection of DELETE operation."""
        query = "DELETE FROM users WHERE user_id = 1"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'DELETE' in ops

    def test_detect_update(self, validator):
        """Test detection of UPDATE operation."""
        query = "UPDATE users SET username = 'new' WHERE user_id = 1"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'UPDATE' in ops

    def test_detect_insert(self, validator):
        """Test detection of INSERT operation."""
        query = "INSERT INTO users (username) VALUES ('test')"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'INSERT' in ops

    def test_detect_multiple_dangerous_ops(self, validator):
        """Test detection of multiple dangerous operations."""
        query = "DROP TABLE users; DELETE FROM events"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'DROP' in ops
        assert 'DELETE' in ops

    def test_case_insensitive_detection(self, validator):
        """Test that detection is case-insensitive."""
        query = "drop table users"
        has_dangerous, ops = validator.check_dangerous_operations(query)
        assert has_dangerous is True
        assert 'DROP' in ops


class TestValidateFields:
    """Tests for validate_fields method."""

    def test_validate_star_select(self, validator):
        """Test that SELECT * is always valid."""
        query = "SELECT * FROM users"
        valid, warnings = validator.validate_fields(query)
        assert valid is True
        assert len(warnings) == 0

    def test_validate_existing_fields(self, validator):
        """Test validation of existing fields."""
        query = "SELECT username, email FROM users"
        valid, warnings = validator.validate_fields(query)
        assert valid is True
        assert len(warnings) == 0

    def test_validate_nonexistent_field(self, validator):
        """Test detection of nonexistent field."""
        query = "SELECT nonexistent_field FROM users"
        valid, warnings = validator.validate_fields(query)
        # In non-strict mode, should be valid with warning
        assert valid is True
        assert len(warnings) > 0

    def test_validate_nonexistent_field_strict(self, validator):
        """Test detection of nonexistent field in strict mode."""
        query = "SELECT nonexistent_field FROM users"
        valid, warnings = validator.validate_fields(query, strict=True)
        # In strict mode, should be invalid
        assert valid is False
        assert len(warnings) > 0

    def test_validate_qualified_field_names(self, validator):
        """Test validation of table.field format."""
        query = "SELECT users.username FROM users"
        valid, warnings = validator.validate_fields(query)
        assert valid is True

    def test_validate_aggregate_functions(self, validator):
        """Test that aggregate functions are not validated as fields."""
        query = "SELECT COUNT(*), MAX(user_id) FROM users"
        valid, warnings = validator.validate_fields(query)
        # Should not fail on COUNT(*) or MAX()
        assert valid is True

    def test_validate_field_alias(self, validator):
        """Test validation of fields with aliases."""
        query = "SELECT username AS name FROM users"
        valid, warnings = validator.validate_fields(query)
        assert valid is True

    def test_validate_no_schemas(self, validator_no_schemas):
        """Test field validation without schemas."""
        query = "SELECT anything FROM anywhere"
        valid, warnings = validator_no_schemas.validate_fields(query)
        assert valid is True
        assert "Cannot validate fields without schemas" in warnings[0]


class TestExtractSelectFields:
    """Tests for _extract_select_fields method."""

    def test_extract_star(self, validator):
        """Test extracting SELECT *."""
        query = "SELECT * FROM users"
        fields = validator._extract_select_fields(query)
        assert fields == ['*']

    def test_extract_single_field(self, validator):
        """Test extracting single field."""
        query = "SELECT username FROM users"
        fields = validator._extract_select_fields(query)
        assert 'username' in fields

    def test_extract_multiple_fields(self, validator):
        """Test extracting multiple fields."""
        query = "SELECT username, email, user_id FROM users"
        fields = validator._extract_select_fields(query)
        assert len(fields) == 3
        assert 'username' in fields
        assert 'email' in fields
        assert 'user_id' in fields

    def test_extract_with_aliases(self, validator):
        """Test extracting fields with aliases."""
        query = "SELECT username AS name, email AS contact FROM users"
        fields = validator._extract_select_fields(query)
        assert any('username' in f.lower() for f in fields)
        assert any('email' in f.lower() for f in fields)

    def test_extract_qualified_fields(self, validator):
        """Test extracting qualified field names."""
        query = "SELECT users.username, events.severity FROM users JOIN events"
        fields = validator._extract_select_fields(query)
        assert any('username' in f.lower() for f in fields)
        assert any('severity' in f.lower() for f in fields)


class TestValidate:
    """Tests for comprehensive validate method."""

    def test_validate_valid_query(self, validator):
        """Test validation of completely valid query."""
        query = "SELECT username FROM users WHERE user_id = 1"
        result = validator.validate(query)

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_invalid_syntax(self, validator):
        """Test validation of query with invalid syntax."""
        query = "INVALID SQL"
        result = validator.validate(query)

        assert result['valid'] is False
        assert "Invalid SQL syntax" in result['errors'][0]

    def test_validate_nonexistent_table(self, validator):
        """Test validation of query with nonexistent table."""
        query = "SELECT * FROM nonexistent_table"
        result = validator.validate(query)

        assert result['valid'] is False
        assert any('Unknown tables' in err for err in result['errors'])

    def test_validate_dangerous_operation(self, validator):
        """Test validation of query with dangerous operation."""
        query = "DROP TABLE users"
        result = validator.validate(query)

        # Should have warning about dangerous operation
        # But also error about invalid syntax (no SELECT)
        assert any('dangerous' in str(w).lower() for w in result['warnings']) or result['valid'] is False

    def test_validate_with_warnings(self, validator):
        """Test validation that produces warnings."""
        query = "SELECT nonexistent_field FROM users"
        result = validator.validate(query, strict=False)

        # Should be valid but with warnings
        assert result['valid'] is True
        assert len(result['warnings']) > 0

    def test_validate_strict_mode(self, validator):
        """Test validation in strict mode."""
        query = "SELECT nonexistent_field FROM users"
        result = validator.validate(query, strict=True)

        # Should be invalid in strict mode
        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_validate_includes_query(self, validator):
        """Test that validation result includes original query."""
        query = "SELECT * FROM users"
        result = validator.validate(query)

        assert result['query'] == query

    def test_validate_no_schemas(self, validator_no_schemas):
        """Test validation without schemas."""
        query = "SELECT * FROM users"
        result = validator_no_schemas.validate(query)

        # Should be valid (can't check table/field without schemas)
        assert result['valid'] is True
