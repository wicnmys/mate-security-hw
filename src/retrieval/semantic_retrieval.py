"""Semantic retrieval using local embeddings for table selection."""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetrieval:
    """
    Semantic retrieval using embedding-based similarity search.

    Uses sentence-transformers (local embeddings) to find tables most
    semantically similar to the user's question. No API calls required.
    """

    def __init__(
        self,
        schemas: Dict[str, Any],
        embedding_model: str = "multi-qa-mpnet-base-dot-v1",
        cache_path: Optional[str] = None
    ):
        """
        Initialize semantic retrieval.

        Args:
            schemas: Dictionary mapping table names to schema definitions
            embedding_model: Sentence-transformers model name (default: multi-qa-mpnet-base-dot-v1)
            cache_path: Optional path to cache embeddings (saves computation time)
        """
        self.schemas = schemas
        self.embedding_model_name = embedding_model
        self.cache_path = cache_path

        # Initialize sentence-transformers model
        print(f"Loading embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)
        print(f"✓ Model loaded")

        # Pre-compute table embeddings
        self.table_embeddings: Dict[str, np.ndarray] = {}
        self.table_descriptions: Dict[str, str] = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        """Pre-compute embeddings for all tables."""
        # Try to load from cache
        if self.cache_path and Path(self.cache_path).exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)

                    # Verify cache is for the same model
                    if cache.get('model_name') == self.embedding_model_name:
                        self.table_embeddings = cache['embeddings']
                        self.table_descriptions = cache['descriptions']
                        print(f"✓ Loaded embeddings from cache: {self.cache_path}")
                        return
                    else:
                        print(f"⚠️  Cache model mismatch, recomputing embeddings...")
            except Exception as e:
                print(f"⚠️  Could not load cache: {e}")

        # Compute embeddings for each table
        from src.utils.schema_loader import get_table_description

        print(f"Computing embeddings for {len(self.schemas)} tables...")

        # Collect all descriptions
        descriptions = []
        table_names = []

        for table_name, schema in self.schemas.items():
            description = get_table_description(table_name, schema)
            self.table_descriptions[table_name] = description
            descriptions.append(description)
            table_names.append(table_name)

        # Batch encode all descriptions (much faster than one-by-one)
        embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store embeddings
        for table_name, embedding in zip(table_names, embeddings):
            self.table_embeddings[table_name] = embedding

        print(f"✓ Computed {len(self.table_embeddings)} embeddings")

        # Save to cache
        if self.cache_path:
            try:
                cache_dir = Path(self.cache_path).parent
                cache_dir.mkdir(parents=True, exist_ok=True)

                with open(self.cache_path, 'wb') as f:
                    pickle.dump({
                        'model_name': self.embedding_model_name,
                        'embeddings': self.table_embeddings,
                        'descriptions': self.table_descriptions
                    }, f)
                print(f"✓ Saved embeddings to cache: {self.cache_path}")
            except Exception as e:
                print(f"⚠️  Could not save cache: {e}")

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Get embedding for text using sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_top_k(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant tables based on semantic similarity.

        Args:
            question: Natural language question
            k: Number of tables to retrieve

        Returns:
            List of table schemas with metadata, sorted by relevance

        Raises:
            ValueError: If question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Embed the question
        question_embedding = self._embed_text(question)

        # Calculate similarity with each table
        similarities: List[tuple[str, float]] = []

        for table_name, table_embedding in self.table_embeddings.items():
            similarity = self._cosine_similarity(question_embedding, table_embedding)
            similarities.append((table_name, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for table_name, similarity in similarities[:k]:
            results.append({
                'table_name': table_name,
                'schema': self.schemas[table_name],
                'score': similarity,
                'match_type': 'semantic'
            })

        return results

    def search_similar_tables(
        self,
        table_name: str,
        k: int = 3,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find tables similar to a given table (useful for JOIN suggestions).

        Args:
            table_name: Name of the reference table
            k: Number of similar tables to retrieve
            exclude_self: Whether to exclude the reference table from results

        Returns:
            List of similar table schemas
        """
        if table_name not in self.table_embeddings:
            raise ValueError(f"Table '{table_name}' not found in schemas")

        reference_embedding = self.table_embeddings[table_name]

        # Calculate similarity with all tables
        similarities: List[tuple[str, float]] = []

        for other_table, other_embedding in self.table_embeddings.items():
            if exclude_self and other_table == table_name:
                continue

            similarity = self._cosine_similarity(reference_embedding, other_embedding)
            similarities.append((other_table, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for other_table, similarity in similarities[:k]:
            results.append({
                'table_name': other_table,
                'schema': self.schemas[other_table],
                'score': similarity,
                'match_type': 'semantic_similar'
            })

        return results
