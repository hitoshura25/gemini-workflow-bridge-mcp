"""
Tests for core business logic.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from hitoshura25_gemini_workflow_bridge.generator import (
    analyze_codebase_with_gemini,
    create_specification_with_gemini,
    review_code_with_gemini,
    generate_documentation_with_gemini,
    ask_gemini,
)


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client to avoid real API calls."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock:
        client = Mock()
        client.generate_content = AsyncMock(return_value='{"analysis": "test analysis", "architecture_summary": "test summary", "relevant_files": [], "patterns_identified": [], "integration_points": []}')
        client.analyze_with_context = AsyncMock(return_value='{"analysis": "test analysis", "architecture_summary": "test summary", "relevant_files": [], "patterns_identified": [], "integration_points": []}')
        client.cache_context = Mock()
        client.get_cached_context = Mock(return_value=None)
        mock.return_value = client
        yield client


@pytest.fixture
def mock_codebase_loader():
    """Mock codebase loader to avoid file system access."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_codebase_loader') as mock:
        loader = Mock()
        loader.load_files = Mock(return_value={"test.py": "print('hello')"})
        loader.get_project_structure = Mock(return_value="test/\n├── test.py")
        mock.return_value = loader
        yield loader


def test_analyze_codebase_with_gemini(mock_gemini_client, mock_codebase_loader):
    """Test analyze_codebase_with_gemini function."""
    result = analyze_codebase_with_gemini(
        focus_description="test analysis",
        directories=None,
        file_patterns=None,
        exclude_patterns=None
    )

    assert isinstance(result, dict)
    assert "analysis" in result
    assert "cached_context_id" in result
    # Should have called the client
    assert mock_gemini_client.analyze_with_context.called


def test_analyze_codebase_with_custom_patterns(mock_gemini_client, mock_codebase_loader):
    """Test analyze_codebase_with_gemini with custom patterns."""
    result = analyze_codebase_with_gemini(
        focus_description="test analysis",
        directories=["src"],
        file_patterns=["*.py"],
        exclude_patterns=["test*"]
    )

    assert isinstance(result, dict)
    assert "analysis" in result
    # Check that loader was called with correct parameters
    mock_codebase_loader.load_files.assert_called_once()


def test_create_specification_with_gemini(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test create_specification_with_gemini function."""
    # Mock the generate_content to return a spec-like response
    mock_gemini_client.generate_content = AsyncMock(return_value="# Test Spec\n\n## Tasks\n- Task 1\n\n## Files to Create\n- file1.py\n\n## Files to Modify\n- file2.py")

    output_file = tmp_path / "test-spec.md"

    result = create_specification_with_gemini(
        feature_description="test feature",
        context_id=None,
        spec_template="standard",
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "spec_path" in result
    assert "spec_content" in result
    assert "implementation_tasks" in result
    assert "estimated_complexity" in result
    assert output_file.exists()


def test_create_specification_with_context(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test create_specification_with_gemini with cached context."""
    # Mock cached context
    mock_gemini_client.get_cached_context = Mock(return_value={
        "analysis": {
            "architecture_summary": "test",
            "relevant_files": ["file1.py"],
            "patterns_identified": ["pattern1"]
        }
    })
    mock_gemini_client.analyze_with_context = AsyncMock(return_value="# Test Spec\n\n## Tasks\n- Task 1")

    output_file = tmp_path / "test-spec.md"

    result = create_specification_with_gemini(
        feature_description="test feature",
        context_id="ctx_test123",
        spec_template="minimal",
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "spec_path" in result
    # Should use analyze_with_context when context is provided
    assert mock_gemini_client.analyze_with_context.called


@patch('hitoshura25_gemini_workflow_bridge.generator._get_git_diff')
def test_review_code_with_gemini(mock_git_diff, mock_gemini_client, tmp_path):
    """Test review_code_with_gemini function."""
    mock_git_diff.return_value = "diff --git a/test.py"
    mock_gemini_client.generate_content = AsyncMock(return_value='{"issues_found": [], "summary": "Looks good", "has_blocking_issues": false, "recommendations": []}')

    output_file = tmp_path / "review.md"

    result = review_code_with_gemini(
        files=None,
        review_focus=None,
        spec_path=None,
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "review_path" in result
    assert "review_content" in result
    assert "issues_found" in result
    assert "has_blocking_issues" in result
    assert "summary" in result
    assert output_file.exists()


def test_review_code_with_files(mock_gemini_client, tmp_path):
    """Test review_code_with_gemini with specific files."""
    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    mock_gemini_client.generate_content = AsyncMock(return_value='{"issues_found": [], "summary": "Looks good", "has_blocking_issues": false, "recommendations": []}')

    output_file = tmp_path / "review.md"

    result = review_code_with_gemini(
        files=[str(test_file)],
        review_focus=["security", "performance"],
        spec_path=None,
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "review_path" in result


def test_generate_documentation_with_gemini(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test generate_documentation_with_gemini function."""
    mock_gemini_client.analyze_with_context = AsyncMock(return_value="# API Documentation\n\nTest documentation content")

    output_file = tmp_path / "api-docs.md"

    result = generate_documentation_with_gemini(
        documentation_type="api",
        scope="test API",
        output_path=str(output_file),
        include_examples=True
    )

    assert isinstance(result, dict)
    assert "doc_path" in result
    assert "doc_content" in result
    assert "sections" in result
    assert "word_count" in result
    assert output_file.exists()


def test_ask_gemini_simple(mock_gemini_client):
    """Test ask_gemini function without context."""
    mock_gemini_client.generate_content = AsyncMock(return_value="This is the answer")

    result = ask_gemini(
        prompt="What is 2+2?",
        include_codebase_context=False,
        context_id=None,
        temperature=0.7
    )

    assert isinstance(result, dict)
    assert "response" in result
    assert "context_used" in result
    assert "token_count" in result
    assert result["context_used"] is False
    assert mock_gemini_client.generate_content.called


def test_ask_gemini_with_context(mock_gemini_client, mock_codebase_loader):
    """Test ask_gemini function with codebase context."""
    mock_gemini_client.analyze_with_context = AsyncMock(return_value="This is the answer with context")

    result = ask_gemini(
        prompt="Explain this codebase",
        include_codebase_context=True,
        context_id=None,
        temperature=0.5
    )

    assert isinstance(result, dict)
    assert "response" in result
    assert result["context_used"] is True
    assert mock_gemini_client.analyze_with_context.called


def test_ask_gemini_with_cached_context(mock_gemini_client):
    """Test ask_gemini function with cached context."""
    mock_gemini_client.get_cached_context = Mock(return_value={
        "analysis": {
            "architecture_summary": "test",
            "relevant_files": [],
            "patterns_identified": []
        }
    })
    mock_gemini_client.analyze_with_context = AsyncMock(return_value="Answer using cached context")

    result = ask_gemini(
        prompt="Question about the code",
        include_codebase_context=False,
        context_id="ctx_test123",
        temperature=0.7
    )

    assert isinstance(result, dict)
    assert result["context_used"] is True
    assert mock_gemini_client.get_cached_context.called


def test_cli_not_found():
    """Test error when Gemini CLI is not installed."""
    with patch('shutil.which', return_value=None):
        from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient
        with pytest.raises(RuntimeError, match="Gemini CLI not found"):
            GeminiClient()


def test_cli_not_working():
    """Test error when Gemini CLI is found but not working."""
    with patch('shutil.which', return_value='/usr/local/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            # Simulate CLI returning error
            mock_run.return_value = Mock(returncode=1, stderr="Authentication error")

            from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient
            with pytest.raises(RuntimeError, match="Gemini CLI found but not working"):
                GeminiClient()


@pytest.mark.asyncio
async def test_cli_timeout():
    """Test timeout handling for long-running CLI calls."""
    with patch('shutil.which', return_value='/usr/local/bin/gemini'):
        with patch('subprocess.run', return_value=Mock(returncode=0, stderr="")):
            from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient
            client = GeminiClient()

            # Mock asyncio.wait_for to raise TimeoutError
            import asyncio
            original_wait_for = asyncio.wait_for

            async def mock_wait_for(coro, timeout):
                # Close the coroutine to avoid unawaited coroutine warning
                coro.close()
                raise asyncio.TimeoutError()

            with patch('asyncio.wait_for', side_effect=mock_wait_for):
                with patch('asyncio.create_subprocess_exec') as mock_exec:
                    mock_process = AsyncMock()
                    mock_process.communicate = AsyncMock()
                    mock_process.kill = AsyncMock()
                    mock_process.wait = AsyncMock()
                    mock_exec.return_value = mock_process

                    with pytest.raises(RuntimeError, match="timed out"):
                        await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_cli_json_parsing():
    """Test handling of malformed JSON from CLI."""
    with patch('shutil.which', return_value='/usr/local/bin/gemini'):
        with patch('subprocess.run', return_value=Mock(returncode=0, stderr="")):
            from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient
            client = GeminiClient()

            # Mock subprocess returning invalid JSON
            async def mock_communicate():
                return (b"This is not JSON", b"")

            with patch('asyncio.create_subprocess_exec') as mock_exec:
                mock_process = AsyncMock()
                mock_process.communicate = mock_communicate
                mock_process.returncode = 0
                mock_exec.return_value = mock_process

                # Should fallback to returning raw output
                result = await client.generate_content("test prompt")
                assert result == "This is not JSON"
