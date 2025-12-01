"""Unit tests for semantic retrieval."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.semantic_retrieval import SemanticRetrieval


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
                {"name": "email", "type": "string", "description": "User email address"}
            ]
        },
        "events": {
            "description": "Security event logs",
            "category": "logging",
            "fields": [
                {"name": "event_id", "type": "integer", "description": "Event identifier"},
                {"name": "timestamp", "type": "datetime", "description": "When event occurred"},
                {"name": "severity", "type": "string", "description": "Event severity level"}
            ]
        }
    }


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock_model = Mock()

    # Mock encode to return predictable embeddings
    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            # Single text
            return np.random.rand(384)  # Standard embedding size
        else:
            # Batch of texts
            return np.random.rand(len(texts), 384)

    mock_model.encode = Mock(side_effect=mock_encode)
    return mock_model


class TestSemanticRetrieval:
    """Tests for SemanticRetrieval class."""

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_initialization(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that semantic retrieval initializes correctly."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        assert retrieval.schemas == sample_schemas
        assert retrieval.embedding_model_name == "test-model"
        mock_st_class.assert_called_once_with("test-model")

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_precompute_embeddings(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that embeddings are precomputed for all tables."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        # Should have embeddings for all tables
        assert len(retrieval.table_embeddings) == len(sample_schemas)
        assert 'users' in retrieval.table_embeddings
        assert 'events' in retrieval.table_embeddings

        # Should have descriptions for all tables
        assert len(retrieval.table_descriptions) == len(sample_schemas)
        assert 'users' in retrieval.table_descriptions
        assert 'events' in retrieval.table_descriptions

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cache_saving(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that embeddings are cached correctly using numpy format."""
        mock_st_class.return_value = mock_sentence_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "embeddings.npz"

            retrieval = SemanticRetrieval(
                sample_schemas,
                embedding_model="test-model",
                cache_path=str(cache_path)
            )

            # Cache file should exist
            assert cache_path.exists()

            # Cache should contain correct data using numpy format
            cache = np.load(cache_path, allow_pickle=False)

            assert 'model_name' in cache
            assert str(cache['model_name']) == "test-model"
            assert 'table_names' in cache
            assert 'descriptions' in cache

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cache_loading(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that embeddings are loaded from cache."""
        mock_st_class.return_value = mock_sentence_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "embeddings.npz"

            # First instance creates cache
            retrieval1 = SemanticRetrieval(
                sample_schemas,
                embedding_model="test-model",
                cache_path=str(cache_path)
            )

            # Reset the mock to count new calls
            mock_sentence_transformer.encode.reset_mock()

            # Second instance should load from cache
            retrieval2 = SemanticRetrieval(
                sample_schemas,
                embedding_model="test-model",
                cache_path=str(cache_path)
            )

            # Should not have called encode again (loaded from cache)
            mock_sentence_transformer.encode.assert_not_called()

            # Should have same embeddings
            assert set(retrieval2.table_embeddings.keys()) == set(retrieval1.table_embeddings.keys())

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cache_model_mismatch(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that cache is invalidated on model mismatch."""
        mock_st_class.return_value = mock_sentence_transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "embeddings.npz"

            # Create cache with model1
            retrieval1 = SemanticRetrieval(
                sample_schemas,
                embedding_model="model1",
                cache_path=str(cache_path)
            )

            # Reset the mock to count new calls
            mock_sentence_transformer.encode.reset_mock()

            # Load with different model - should recompute
            retrieval2 = SemanticRetrieval(
                sample_schemas,
                embedding_model="model2",
                cache_path=str(cache_path)
            )

            # Should have called encode to recompute embeddings
            assert mock_sentence_transformer.encode.called


class TestGetTopK:
    """Tests for get_top_k method."""

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_returns_results(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that get_top_k returns results."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.get_top_k("Show me users", k=5)

        assert len(results) > 0
        assert len(results) <= 5

        # Check result structure
        for result in results:
            assert 'table_name' in result
            assert 'schema' in result
            assert 'score' in result
            assert 'match_type' in result
            assert result['match_type'] == 'semantic'

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_respects_k_parameter(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that k parameter limits results."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        results = retrieval.get_top_k("security data", k=1)
        assert len(results) == 1

        results = retrieval.get_top_k("security data", k=2)
        assert len(results) == 2

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_empty_question_raises_error(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that empty question raises ValueError."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        with pytest.raises(ValueError, match="Question cannot be empty"):
            retrieval.get_top_k("")

        with pytest.raises(ValueError, match="Question cannot be empty"):
            retrieval.get_top_k("   ")

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_scores_in_range(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that similarity scores are in valid range."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.get_top_k("user authentication", k=5)

        for result in results:
            # Cosine similarity is between -1 and 1, but typically positive for relevant docs
            assert -1 <= result['score'] <= 1

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_includes_schema(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that results include full schema."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.get_top_k("users", k=1)

        assert len(results) > 0
        assert 'schema' in results[0]
        # Schema should be one of our sample schemas
        assert results[0]['schema'] in sample_schemas.values()

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_get_top_k_sorted_by_score(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that results are sorted by similarity score."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.get_top_k("security events", k=5)

        # Scores should be in descending order
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestSearchSimilarTables:
    """Tests for search_similar_tables method."""

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_search_similar_tables_returns_results(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that search_similar_tables returns results."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.search_similar_tables("users", k=3)

        assert len(results) > 0

        # Check result structure
        for result in results:
            assert 'table_name' in result
            assert 'schema' in result
            assert 'score' in result
            assert 'match_type' in result
            assert result['match_type'] == 'semantic_similar'

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_search_similar_tables_excludes_self(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that search_similar_tables excludes reference table by default."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.search_similar_tables("users", k=3, exclude_self=True)

        # Should not include 'users' itself
        table_names = [r['table_name'] for r in results]
        assert 'users' not in table_names

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_search_similar_tables_includes_self_when_requested(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that search_similar_tables can include reference table."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")
        results = retrieval.search_similar_tables("users", k=3, exclude_self=False)

        # Should include 'users' itself (probably as top result)
        table_names = [r['table_name'] for r in results]
        assert 'users' in table_names

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_search_similar_tables_nonexistent_table_raises_error(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that nonexistent table raises ValueError."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        with pytest.raises(ValueError, match="not found in schemas"):
            retrieval.search_similar_tables("nonexistent_table", k=3)


class TestCosineSimilarity:
    """Tests for _cosine_similarity method."""

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cosine_similarity_identical_vectors(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that identical vectors have similarity of 1.0."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        vec = np.array([1.0, 2.0, 3.0])
        similarity = retrieval._cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-6

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cosine_similarity_orthogonal_vectors(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that orthogonal vectors have similarity of 0.0."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = retrieval._cosine_similarity(vec1, vec2)

        assert abs(similarity - 0.0) < 1e-6

    @patch('src.retrieval.semantic_retrieval.SentenceTransformer')
    def test_cosine_similarity_opposite_vectors(self, mock_st_class, sample_schemas, mock_sentence_transformer):
        """Test that opposite vectors have similarity of -1.0."""
        mock_st_class.return_value = mock_sentence_transformer

        retrieval = SemanticRetrieval(sample_schemas, embedding_model="test-model")

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = retrieval._cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-6
