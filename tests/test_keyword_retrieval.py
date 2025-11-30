"""Unit tests for keyword retrieval."""

import pytest
from src.retrieval.keyword_retrieval import KeywordRetrieval


@pytest.fixture
def sample_schemas():
    """Sample schemas for testing."""
    return {
        "users": {
            "description": "User authentication and profile data",
            "category": "security",
            "fields": [
                {"name": "user_id", "type": "integer", "description": "Unique user identifier"},
                {"name": "username", "type": "string", "description": "Login username"},
                {"name": "email", "type": "string", "description": "User email address"},
                {"name": "password_hash", "type": "string", "description": "Hashed password"}
            ]
        },
        "events": {
            "description": "Security event logs",
            "category": "logging",
            "fields": [
                {"name": "event_id", "type": "integer", "description": "Event identifier"},
                {"name": "timestamp", "type": "datetime", "description": "When event occurred"},
                {"name": "severity", "type": "string", "description": "Event severity level"},
                {"name": "message", "type": "string", "description": "Event message"}
            ]
        },
        "sessions": {
            "description": "User session tracking",
            "fields": [
                {"name": "session_id", "type": "string"},
                {"name": "user_id", "type": "integer"},
                {"name": "start_time", "type": "datetime"},
                {"name": "end_time", "type": "datetime"}
            ]
        }
    }


@pytest.fixture
def retrieval(sample_schemas):
    """Initialize keyword retrieval with sample schemas."""
    return KeywordRetrieval(sample_schemas)


class TestKeywordRetrieval:
    """Tests for KeywordRetrieval class."""

    def test_initialization(self, sample_schemas):
        """Test that keyword retrieval initializes correctly."""
        retrieval = KeywordRetrieval(sample_schemas)
        assert retrieval.schemas == sample_schemas
        assert hasattr(retrieval, 'keyword_index')
        assert len(retrieval.keyword_index) > 0

    def test_keyword_index_contains_table_names(self, retrieval):
        """Test that keyword index includes table name tokens."""
        # 'users' table name should create 'users' keyword
        assert 'users' in retrieval.keyword_index
        assert 'users' in retrieval.keyword_index['users']

        # 'events' table name
        assert 'events' in retrieval.keyword_index
        assert 'events' in retrieval.keyword_index['events']

    def test_keyword_index_contains_field_names(self, retrieval):
        """Test that keyword index includes field names."""
        # 'username' is a field in users table
        assert 'username' in retrieval.keyword_index
        assert 'users' in retrieval.keyword_index['username']

        # 'severity' is a field in events table
        assert 'severity' in retrieval.keyword_index
        assert 'events' in retrieval.keyword_index['severity']

    def test_keyword_index_contains_descriptions(self, retrieval):
        """Test that keyword index includes description tokens."""
        # 'authentication' from users description
        assert 'authentication' in retrieval.keyword_index
        assert 'users' in retrieval.keyword_index['authentication']

        # 'security' from users category and events description
        assert 'security' in retrieval.keyword_index


class TestTokenize:
    """Tests for _tokenize method."""

    def test_tokenize_basic(self, retrieval):
        """Test basic tokenization."""
        tokens = retrieval._tokenize("User Authentication")
        assert 'user' in tokens
        assert 'authentication' in tokens

    def test_tokenize_lowercase(self, retrieval):
        """Test that tokenization converts to lowercase."""
        tokens = retrieval._tokenize("USER Authentication")
        assert 'user' in tokens
        assert 'USER' not in tokens

    def test_tokenize_filters_short_tokens(self, retrieval):
        """Test that very short tokens are filtered out."""
        tokens = retrieval._tokenize("a bb ccc dddd")
        # 'a' and 'bb' should be filtered (length <= 2)
        assert 'a' not in tokens
        assert 'bb' not in tokens
        # 'ccc' and 'dddd' should remain
        assert 'ccc' in tokens
        assert 'dddd' in tokens

    def test_tokenize_splits_on_non_alphanumeric(self, retrieval):
        """Test that tokenization splits on non-alphanumeric characters."""
        # Underscores are treated as word characters by \w, so they don't split
        # Hyphens do split
        tokens = retrieval._tokenize("user-profile system")
        assert 'user' in tokens
        assert 'profile' in tokens
        assert 'system' in tokens

    def test_tokenize_empty_string(self, retrieval):
        """Test tokenization of empty string."""
        tokens = retrieval._tokenize("")
        assert tokens == []

    def test_tokenize_only_short_tokens(self, retrieval):
        """Test tokenization with only short tokens."""
        tokens = retrieval._tokenize("a b c")
        assert tokens == []


class TestGetTopK:
    """Tests for get_top_k method."""

    def test_get_top_k_with_exact_match(self, retrieval):
        """Test retrieval with exact keyword match."""
        results = retrieval.get_top_k("Show me user data", k=5)

        assert len(results) > 0
        # 'users' table should be first (contains 'user' keyword)
        assert results[0]['table_name'] == 'users'
        assert results[0]['match_type'] == 'keyword'
        assert results[0]['score'] > 0

    def test_get_top_k_multiple_matches(self, retrieval):
        """Test retrieval with multiple keyword matches."""
        results = retrieval.get_top_k("user authentication and sessions", k=5)

        table_names = [r['table_name'] for r in results]
        # Should include both users and sessions
        assert 'users' in table_names
        assert 'sessions' in table_names

    def test_get_top_k_respects_k_parameter(self, retrieval):
        """Test that k parameter limits results."""
        results = retrieval.get_top_k("security events", k=2)
        assert len(results) <= 2

    def test_get_top_k_empty_question_raises_error(self, retrieval):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            retrieval.get_top_k("")

        with pytest.raises(ValueError, match="Question cannot be empty"):
            retrieval.get_top_k("   ")

    def test_get_top_k_no_valid_keywords_raises_error(self, retrieval):
        """Test that question with no valid keywords raises ValueError."""
        with pytest.raises(ValueError, match="No valid keywords found"):
            retrieval.get_top_k("a b c")  # All tokens too short

    def test_get_top_k_no_matches_returns_fallback(self, retrieval):
        """Test fallback when no keyword matches."""
        results = retrieval.get_top_k("xyzabc nonexistent keywords", k=3)

        assert len(results) > 0
        # Should return fallback results
        assert all(r['match_type'] == 'fallback' for r in results)
        assert all(r['score'] == 0.0 for r in results)

    def test_get_top_k_score_normalization(self, retrieval):
        """Test that scores are normalized."""
        results = retrieval.get_top_k("user authentication", k=5)

        # Top result should have score of 1.0 (normalized)
        assert results[0]['score'] == 1.0

        # Other scores should be <= 1.0
        for result in results:
            assert 0 <= result['score'] <= 1.0

    def test_get_top_k_includes_schema(self, retrieval, sample_schemas):
        """Test that results include full schema."""
        results = retrieval.get_top_k("users", k=1)

        assert len(results) > 0
        assert 'schema' in results[0]
        assert results[0]['schema'] == sample_schemas['users']

    def test_get_top_k_scoring(self, retrieval):
        """Test that tables with more keyword matches score higher."""
        # Users table has 'user' in name, description, and field names
        # Should score higher than sessions which only has 'user' in field name
        results = retrieval.get_top_k("user", k=5)

        table_scores = {r['table_name']: r['score'] for r in results}

        if 'users' in table_scores and 'sessions' in table_scores:
            assert table_scores['users'] >= table_scores['sessions']


class TestGetMatchingFields:
    """Tests for get_matching_fields method."""

    def test_get_matching_fields_exact_match(self, retrieval):
        """Test finding fields with exact keyword match."""
        fields = retrieval.get_matching_fields("users", ["username"])

        assert 'username' in fields

    def test_get_matching_fields_description_match(self, retrieval):
        """Test finding fields by description keywords."""
        fields = retrieval.get_matching_fields("users", ["email"])

        assert 'email' in fields

    def test_get_matching_fields_multiple_keywords(self, retrieval):
        """Test finding fields with multiple keywords."""
        fields = retrieval.get_matching_fields("users", ["user", "email", "password"])

        # Should match user_id, email, password_hash
        assert 'user_id' in fields
        assert 'email' in fields
        assert 'password_hash' in fields

    def test_get_matching_fields_case_insensitive(self, retrieval):
        """Test that field matching is case-insensitive."""
        fields = retrieval.get_matching_fields("users", ["USERNAME"])

        assert 'username' in fields

    def test_get_matching_fields_no_matches(self, retrieval):
        """Test when no fields match keywords."""
        fields = retrieval.get_matching_fields("users", ["nonexistent"])

        assert fields == []

    def test_get_matching_fields_nonexistent_table(self, retrieval):
        """Test with nonexistent table name."""
        fields = retrieval.get_matching_fields("nonexistent_table", ["user"])

        assert fields == []

    def test_get_matching_fields_empty_keywords(self, retrieval):
        """Test with empty keywords list."""
        fields = retrieval.get_matching_fields("users", [])

        assert fields == []

    def test_get_matching_fields_partial_token_match(self, retrieval):
        """Test that partial tokens in descriptions are matched."""
        # 'identifier' is in the description of user_id
        fields = retrieval.get_matching_fields("users", ["identifier"])

        assert 'user_id' in fields
