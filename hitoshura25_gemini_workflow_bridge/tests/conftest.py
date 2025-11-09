"""
Shared pytest configuration and fixtures for tests.
"""

import pytest
import os


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("DEFAULT_SPEC_DIR", "./specs")
    monkeypatch.setenv("DEFAULT_REVIEW_DIR", "./reviews")
    monkeypatch.setenv("DEFAULT_CONTEXT_DIR", "./.workflow-context")
