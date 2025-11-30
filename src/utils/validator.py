"""SQL query validation utilities."""

import re
from typing import Dict, List, Any, Optional, Tuple


class SQLValidator:
    """
    Validator for SQL queries.

    Performs syntax checking and schema validation without executing queries.
    """

    # SQL keywords that should be present in valid queries
    REQUIRED_KEYWORDS = {'SELECT'}

    # Dangerous SQL operations to flag
    DANGEROUS_OPERATIONS = {
        'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE'
    }

    def __init__(self, schemas: Optional[Dict[str, Any]] = None):
        """
        Initialize SQL validator.

        Args:
            schemas: Optional dictionary of table schemas for validation
        """
        self.schemas = schemas or {}

    def is_valid(self, query: str) -> bool:
        """
        Check if SQL query has valid syntax (basic check).

        Args:
            query: SQL query string

        Returns:
            True if query appears syntactically valid
        """
        if not query or not query.strip():
            return False

        # Remove extra whitespace and convert to uppercase for checking
        query_upper = ' '.join(query.split()).upper()

        # Check for required keywords
        if not any(keyword in query_upper for keyword in self.REQUIRED_KEYWORDS):
            return False

        # Check for basic SQL structure patterns
        # At minimum, should have SELECT ... FROM pattern
        if 'SELECT' in query_upper:
            # Simple pattern check: SELECT ... FROM ...
            if 'FROM' not in query_upper:
                return False

        # Check for unclosed quotes or parentheses
        if not self._check_balanced_delimiters(query):
            return False

        return True

    def _check_balanced_delimiters(self, query: str) -> bool:
        """
        Check if quotes and parentheses are balanced.

        Args:
            query: SQL query string

        Returns:
            True if delimiters are balanced
        """
        # Check parentheses
        paren_count = 0
        in_single_quote = False
        in_double_quote = False

        i = 0
        while i < len(query):
            char = query[i]

            # Handle escape sequences
            if char == '\\' and i + 1 < len(query):
                i += 2
                continue

            # Toggle quote states
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote

            # Count parentheses outside of quotes
            elif not in_single_quote and not in_double_quote:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1

                if paren_count < 0:
                    return False

            i += 1

        # All delimiters should be closed
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
            # Can't validate without schemas
            return True

        # Extract table names from query
        table_names = self._extract_table_names(query)

        # Check if all tables exist
        return all(table in self.schemas for table in table_names)

    def _extract_table_names(self, query: str) -> List[str]:
        """
        Extract table names from SQL query.

        Args:
            query: SQL query string

        Returns:
            List of table names found in query
        """
        tables = []

        # Pattern to match: FROM table_name or JOIN table_name
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
        Check if query contains dangerous operations.

        Args:
            query: SQL query string

        Returns:
            Tuple of (has_dangerous_ops, list of dangerous operations found)
        """
        query_upper = query.upper()
        dangerous_found = []

        for operation in self.DANGEROUS_OPERATIONS:
            if re.search(rf'\b{operation}\b', query_upper):
                dangerous_found.append(operation)

        return len(dangerous_found) > 0, dangerous_found

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

        # Extract table names
        table_names = self._extract_table_names(query)

        if not table_names:
            return True, ["No tables found in query"]

        # Extract field names from SELECT clause
        select_fields = self._extract_select_fields(query)

        if '*' in select_fields:
            # SELECT * is always valid
            return True, []

        # Build available fields from referenced tables
        available_fields = set()
        for table_name in table_names:
            if table_name in self.schemas:
                schema = self.schemas[table_name]
                for field in schema.get('fields', []):
                    available_fields.add(field['name'].lower())
                    # Also add table.field format
                    available_fields.add(f"{table_name}.{field['name'].lower()}")

        # Check each field
        unknown_fields = []
        for field in select_fields:
            field_lower = field.lower().strip()

            # Skip aggregate functions and expressions
            if any(func in field_lower for func in ['count(', 'sum(', 'avg(', 'max(', 'min(', 'distinct']):
                continue

            # Extract field name from potential alias (field AS alias)
            field_parts = re.split(r'\s+as\s+', field_lower, flags=re.IGNORECASE)
            actual_field = field_parts[0].strip()

            # Check if field exists
            if actual_field not in available_fields and '.' not in actual_field:
                # Also check without table prefix
                base_field = actual_field.split('.')[-1] if '.' in actual_field else actual_field
                if base_field not in available_fields:
                    unknown_fields.append(field)

        if unknown_fields:
            warnings.append(f"Unknown fields: {', '.join(unknown_fields)}")
            if strict:
                return False, warnings

        return True, warnings

    def _extract_select_fields(self, query: str) -> List[str]:
        """
        Extract field names from SELECT clause.

        Args:
            query: SQL query string

        Returns:
            List of field names
        """
        # Find SELECT clause
        select_match = re.search(
            r'\bSELECT\s+(.*?)\s+FROM\b',
            query,
            re.IGNORECASE | re.DOTALL
        )

        if not select_match:
            return []

        select_clause = select_match.group(1)

        # Handle SELECT *
        if '*' in select_clause:
            return ['*']

        # Split by commas (basic parsing)
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

        # Check basic syntax
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
