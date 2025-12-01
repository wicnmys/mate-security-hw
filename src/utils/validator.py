"""SQL query validation utilities using sqlglot for proper AST parsing."""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from src.constants import DANGEROUS_SQL_OPERATIONS

logger = logging.getLogger(__name__)


class SQLValidator:
    """
    Validator for SQL queries using sqlglot AST parsing.

    Performs syntax checking and schema validation without executing queries.
    Uses PostgreSQL dialect for parsing.
    """

    def __init__(self, schemas: Optional[Dict[str, Any]] = None):
        """
        Initialize SQL validator.

        Args:
            schemas: Optional dictionary of table schemas for validation
        """
        self.schemas = schemas or {}
        self.dialect = "postgres"

    def is_valid(self, query: str) -> bool:
        """
        Check if SQL query has valid syntax using sqlglot parsing.

        Args:
            query: SQL query string

        Returns:
            True if query is syntactically valid
        """
        if not query or not query.strip():
            return False

        # Quick check: require SELECT keyword to be present
        # (sqlglot is lenient and may parse "FROM x" as "SELECT * FROM x")
        query_upper = query.upper().strip()
        if not query_upper.startswith('SELECT') and 'SELECT' not in query_upper:
            return False

        try:
            # Try to parse the query
            parsed = sqlglot.parse(query, dialect=self.dialect)

            # Must have at least one statement
            if not parsed or len(parsed) == 0:
                return False

            # Check if any statement is a SELECT with FROM clause
            for stmt in parsed:
                if stmt is None:
                    return False
                # Check for SELECT statement with FROM clause
                if isinstance(stmt, exp.Select):
                    # Require a FROM clause for data retrieval queries
                    if stmt.find(exp.From):
                        return True
                    # SELECT without FROM is not valid for our use case
                    return False
                # Allow other statement types that have a FROM clause
                if stmt.find(exp.From):
                    return True

            # If we get here with no errors, it's syntactically valid
            # but may not be a data retrieval query
            return False

        except ParseError as e:
            logger.debug("Parse error: %s", e)
            return False
        except Exception as e:
            logger.debug("Unexpected error parsing SQL: %s", e)
            return False

    def _check_balanced_delimiters(self, query: str) -> bool:
        """
        Check if quotes and parentheses are balanced.
        Kept for backward compatibility with tests.

        Args:
            query: SQL query string

        Returns:
            True if delimiters are balanced
        """
        paren_count = 0
        in_single_quote = False
        in_double_quote = False

        i = 0
        while i < len(query):
            char = query[i]

            if char == '\\' and i + 1 < len(query):
                i += 2
                continue

            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                if paren_count < 0:
                    return False

            i += 1

        return paren_count == 0 and not in_single_quote and not in_double_quote

    def table_exists(self, query: str) -> bool:
        """
        Check if all referenced tables exist in schemas.

        Args:
            query: SQL query string

        Returns:
            True if all tables exist or no schemas loaded
        """
        if not self.schemas:
            return True

        table_names = self._extract_table_names(query)
        return all(table in self.schemas for table in table_names)

    def _extract_table_names(self, query: str) -> List[str]:
        """
        Extract table names from SQL query using sqlglot AST.

        Args:
            query: SQL query string

        Returns:
            List of table names found in query
        """
        tables = []

        try:
            parsed = sqlglot.parse(query, dialect=self.dialect)

            for stmt in parsed:
                if stmt is None:
                    continue

                # Find all Table expressions in the AST
                for table in stmt.find_all(exp.Table):
                    table_name = table.name.lower()
                    if table_name and table_name not in tables:
                        tables.append(table_name)

        except (ParseError, Exception) as e:
            logger.debug("Error extracting tables, falling back to regex: %s", e)
            tables = self._extract_table_names_regex(query)

        return tables

    def _extract_table_names_regex(self, query: str) -> List[str]:
        """
        Fallback regex-based table extraction.

        Args:
            query: SQL query string

        Returns:
            List of table names found in query
        """
        tables = []
        patterns = [
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]

        query_upper = query.upper()

        for pattern in patterns:
            matches = re.finditer(pattern, query_upper)
            for match in matches:
                table_name = match.group(1).lower()
                if table_name not in tables:
                    tables.append(table_name)

        return tables

    def check_dangerous_operations(self, query: str) -> Tuple[bool, List[str]]:
        """
        Check if query contains dangerous operations using AST analysis.

        Args:
            query: SQL query string

        Returns:
            Tuple of (has_dangerous_ops, list of dangerous operations found)
        """
        dangerous_found = []

        try:
            parsed = sqlglot.parse(query, dialect=self.dialect)

            for stmt in parsed:
                if stmt is None:
                    continue

                # Check statement type against dangerous operations
                stmt_type = type(stmt).__name__.upper()

                for op in DANGEROUS_SQL_OPERATIONS:
                    if op in stmt_type:
                        if op not in dangerous_found:
                            dangerous_found.append(op)

        except (ParseError, Exception):
            # Fall back to keyword detection on parse error
            dangerous_found = self._check_dangerous_regex(query)

        return len(dangerous_found) > 0, dangerous_found

    def _check_dangerous_regex(self, query: str) -> List[str]:
        """
        Fallback regex-based dangerous operation detection.

        Args:
            query: SQL query string

        Returns:
            List of dangerous operations found
        """
        query_upper = query.upper()
        dangerous_found = []

        for operation in DANGEROUS_SQL_OPERATIONS:
            if re.search(rf'\b{operation}\b', query_upper):
                dangerous_found.append(operation)

        return dangerous_found

    def validate_fields(self, query: str, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate that fields in query exist in referenced tables.

        Args:
            query: SQL query string
            strict: If True, fail on any unknown field. If False, warn only.

        Returns:
            Tuple of (all_valid, list of warnings/errors)
        """
        if not self.schemas:
            return True, ["Cannot validate fields without schemas"]

        warnings = []
        table_names = self._extract_table_names(query)

        if not table_names:
            return True, ["No tables found in query"]

        select_fields = self._extract_select_fields(query)

        if '*' in select_fields or not select_fields:
            return True, []

        # Build available fields from referenced tables
        available_fields = set()
        for table_name in table_names:
            if table_name in self.schemas:
                schema = self.schemas[table_name]
                for field in schema.get('fields', []):
                    field_name = field['name'].lower()
                    available_fields.add(field_name)
                    available_fields.add(f"{table_name}.{field_name}")

        # Check each field
        unknown_fields = []
        for field in select_fields:
            field_lower = field.lower().strip()

            # Skip if it looks like an aggregate or expression
            if self._is_aggregate_or_expression(field_lower):
                continue

            # Extract base field name (handle aliases and table qualifiers)
            base_field = self._extract_base_field_name(field_lower)

            if base_field and base_field not in available_fields:
                # Check without table prefix
                field_only = base_field.split('.')[-1] if '.' in base_field else base_field
                if field_only not in available_fields:
                    unknown_fields.append(field)

        if unknown_fields:
            warnings.append(f"Unknown fields: {', '.join(unknown_fields)}")
            if strict:
                return False, warnings

        return True, warnings

    def _is_aggregate_or_expression(self, field: str) -> bool:
        """Check if field is an aggregate function or expression."""
        aggregate_patterns = [
            'count(', 'sum(', 'avg(', 'max(', 'min(',
            'distinct', 'case ', 'when ', 'coalesce(',
            'array_agg(', 'string_agg(', 'json_agg(',
            'now()', 'current_', 'date_trunc(',
        ]
        return any(pattern in field for pattern in aggregate_patterns)

    def _extract_base_field_name(self, field: str) -> str:
        """Extract the base field name from a field expression."""
        # Remove alias (field AS alias)
        field_parts = re.split(r'\s+as\s+', field, flags=re.IGNORECASE)
        base = field_parts[0].strip()

        # If it contains functions or expressions, skip
        if '(' in base or ')' in base:
            return ''

        return base

    def _extract_select_fields(self, query: str) -> List[str]:
        """
        Extract field names from SELECT clause using sqlglot AST.

        Args:
            query: SQL query string

        Returns:
            List of field names/expressions
        """
        fields = []

        try:
            parsed = sqlglot.parse(query, dialect=self.dialect)

            for stmt in parsed:
                if stmt is None:
                    continue

                # Find SELECT expressions
                if isinstance(stmt, exp.Select):
                    for expression in stmt.expressions:
                        # Check for star
                        if isinstance(expression, exp.Star):
                            return ['*']

                        # Get the SQL representation of the expression
                        field_sql = expression.sql(dialect=self.dialect)
                        if field_sql:
                            fields.append(field_sql)

        except (ParseError, Exception) as e:
            logger.debug("Error extracting fields, falling back to regex: %s", e)
            fields = self._extract_select_fields_regex(query)

        return fields

    def _extract_select_fields_regex(self, query: str) -> List[str]:
        """
        Fallback regex-based field extraction.

        Args:
            query: SQL query string

        Returns:
            List of field names
        """
        select_match = re.search(
            r'\bSELECT\s+(.*?)\s+FROM\b',
            query,
            re.IGNORECASE | re.DOTALL
        )

        if not select_match:
            return []

        select_clause = select_match.group(1)

        if '*' in select_clause:
            return ['*']

        fields = [f.strip() for f in select_clause.split(',')]
        return fields

    def validate(self, query: str, strict: bool = False) -> Dict[str, Any]:
        """
        Comprehensive validation of SQL query.

        Args:
            query: SQL query string
            strict: If True, fail on warnings

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'query': query
        }

        # Check basic syntax using sqlglot
        if not self.is_valid(query):
            results['valid'] = False
            results['errors'].append("Invalid SQL syntax")
            return results

        # Check for dangerous operations
        has_dangerous, dangerous_ops = self.check_dangerous_operations(query)
        if has_dangerous:
            results['warnings'].append(
                f"Query contains potentially dangerous operations: {', '.join(dangerous_ops)}"
            )

        # Check table existence
        if not self.table_exists(query):
            table_names = self._extract_table_names(query)
            unknown_tables = [t for t in table_names if t not in self.schemas]
            results['valid'] = False
            results['errors'].append(f"Unknown tables: {', '.join(unknown_tables)}")

        # Validate fields
        if self.schemas:
            fields_valid, field_warnings = self.validate_fields(query, strict=strict)
            if not fields_valid:
                results['valid'] = False
                results['errors'].extend(field_warnings)
            elif field_warnings:
                results['warnings'].extend(field_warnings)

        return results

    def get_parse_errors(self, query: str) -> List[str]:
        """
        Get detailed parse errors from sqlglot.

        Args:
            query: SQL query string

        Returns:
            List of error messages
        """
        errors = []

        try:
            sqlglot.parse(query, dialect=self.dialect)
        except ParseError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return errors
