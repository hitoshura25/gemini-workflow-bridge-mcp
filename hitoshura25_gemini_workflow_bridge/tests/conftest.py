"""
Shared pytest configuration and fixtures for tests.
"""

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests."""
    # No API key needed - uses Gemini CLI
    # "auto" lets CLI choose best model automatically
    monkeypatch.setenv("GEMINI_MODEL", "auto")
    monkeypatch.setenv("DEFAULT_SPEC_DIR", "./specs")
    monkeypatch.setenv("DEFAULT_REVIEW_DIR", "./reviews")
    monkeypatch.setenv("DEFAULT_CONTEXT_DIR", "./.workflow-context")
