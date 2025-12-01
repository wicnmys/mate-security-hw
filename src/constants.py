"""Constants for SQL Query Agent."""

# Retrieval settings
MIN_KEYWORD_LENGTH = 3
DEFAULT_TOP_K_TABLES = 5

# Caching
DEFAULT_CACHE_TTL_SECONDS = 3600

# Embedding model
DEFAULT_EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# LLM model
DEFAULT_LLM_MODEL = "claude-sonnet-4-5"

# Validation
DANGEROUS_SQL_OPERATIONS = frozenset({
    'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
    'ALTER', 'CREATE', 'GRANT', 'REVOKE'
})
