"""Keyword-based table retrieval."""

import re
from typing import Dict, List, Any

from src.constants import MIN_KEYWORD_LENGTH


class KeywordRetrieval:
    """
    Simple keyword-based retrieval for database tables.

    Matches keywords in questions against table names, field names,
    and descriptions to find relevant tables.
    """

    def __init__(self, schemas: Dict[str, Any]):
        """
        Initialize keyword retrieval.

        Args:
            schemas: Dictionary mapping table names to schema definitions
        """
        self.schemas = schemas
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """Build keyword index for all tables."""
        self.keyword_index: Dict[str, List[str]] = {}

        for table_name, schema in self.schemas.items():
            keywords = set()

            # Add table name tokens
            keywords.update(self._tokenize(table_name))

            # Add category
            if 'category' in schema:
                keywords.update(self._tokenize(schema['category']))

            # Add description tokens
            if 'description' in schema:
                keywords.update(self._tokenize(schema['description']))

            # Add field names and descriptions
            for field in schema.get('fields', []):
                keywords.update(self._tokenize(field['name']))
                if 'description' in field:
                    keywords.update(self._tokenize(field['description']))

            # Store keywords for this table
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(table_name)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into keywords.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase keyword tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text)

        # Filter out very short tokens (shorter than MIN_KEYWORD_LENGTH)
        tokens = [t for t in tokens if len(t) >= MIN_KEYWORD_LENGTH]

        return tokens

    def get_top_k(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant tables based on keyword matching.

        Args:
            question: Natural language question
            k: Number of tables to retrieve

        Returns:
            List of table schemas with metadata

        Raises:
            ValueError: If question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Tokenize question
        query_tokens = self._tokenize(question)

        if not query_tokens:
            raise ValueError("No valid keywords found in question")

        # Score each table
        table_scores: Dict[str, int] = {}

        for token in query_tokens:
            if token in self.keyword_index:
                for table_name in self.keyword_index[token]:
                    table_scores[table_name] = table_scores.get(table_name, 0) + 1

        # If no matches, return tables based on common security terms
        if not table_scores:
            # Fallback: return tables with most fields (likely most comprehensive)
            fallback_tables = sorted(
                self.schemas.keys(),
                key=lambda t: len(self.schemas[t].get('fields', [])),
                reverse=True
            )[:k]

            return [
                {
                    'table_name': table_name,
                    'schema': self.schemas[table_name],
                    'score': 0.0,
                    'match_type': 'fallback'
                }
                for table_name in fallback_tables
            ]

        # Sort tables by score
        sorted_tables = sorted(
            table_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Format results
        results = []
        max_score = sorted_tables[0][1] if sorted_tables else 1

        for table_name, score in sorted_tables:
            results.append({
                'table_name': table_name,
                'schema': self.schemas[table_name],
                'score': score / max_score,  # Normalize score
                'match_type': 'keyword'
            })

        return results

    def get_matching_fields(self, table_name: str, keywords: List[str]) -> List[str]:
        """
        Get fields in a table that match given keywords.

        Args:
            table_name: Name of the table
            keywords: List of keywords to match

        Returns:
            List of matching field names
        """
        if table_name not in self.schemas:
            return []

        schema = self.schemas[table_name]
        matching_fields = []

        keyword_set = set(k.lower() for k in keywords)

        for field in schema.get('fields', []):
            field_tokens = self._tokenize(field['name'])

            if 'description' in field:
                field_tokens.extend(self._tokenize(field['description']))

            # Check if any keyword matches
            if any(token in keyword_set for token in field_tokens):
                matching_fields.append(field['name'])

        return matching_fields
