"""Shared fixtures for integration tests."""

import pytest
import os


@pytest.fixture
def integration_model():
    """
    Get model to use for integration tests.

    Defaults to 'claude-sonnet-4-5' (supports structured outputs).
    Override with INTEGRATION_TEST_MODEL environment variable.
    """
    return os.getenv('INTEGRATION_TEST_MODEL', 'claude-sonnet-4-5')
