"""Integration tests for caching parameter validation."""

import pytest
import os
from agno.models.anthropic import Claude


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="ANTHROPIC_API_KEY required for integration tests"
)
class TestCachingParameters:
    """Test that our caching parameters are valid for Anthropic API."""

    def test_valid_caching_parameters(self, integration_model):
        """Test that cache_system_prompt and cache_ttl are accepted."""
        # This should NOT raise TypeError
        model = Claude(
            id=integration_model,
            cache_system_prompt=True,
            cache_ttl=3600  # integer seconds
        )
        assert model is not None

    def test_invalid_cache_tool_definitions_fails(self, integration_model):
        """Test that invalid cache_tool_definitions parameter is rejected."""
        with pytest.raises(TypeError, match="cache_tool_definitions"):
            Claude(
                id=integration_model,
                cache_system_prompt=True,
                cache_tool_definitions=True  # NOT SUPPORTED in Agno 2.3.4
            )

    def test_valid_integer_cache_ttl(self, integration_model):
        """Test that integer cache_ttl is accepted."""
        # This should NOT raise TypeError
        model = Claude(
            id=integration_model,
            cache_ttl=3600  # integer seconds
        )
        assert model is not None
