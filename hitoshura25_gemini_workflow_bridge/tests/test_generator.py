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


@pytest.mark.asyncio
async def test_analyze_codebase_with_gemini(mock_gemini_client, mock_codebase_loader):
    """Test analyze_codebase_with_gemini function."""
    result = await analyze_codebase_with_gemini(
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


@pytest.mark.asyncio
async def test_analyze_codebase_with_custom_patterns(mock_gemini_client, mock_codebase_loader):
    """Test analyze_codebase_with_gemini with custom patterns."""
    result = await analyze_codebase_with_gemini(
        focus_description="test analysis",
        directories=["src"],
        file_patterns=["*.py"],
        exclude_patterns=["test*"]
    )

    assert isinstance(result, dict)
    assert "analysis" in result
    # Check that loader was called with correct parameters
    mock_codebase_loader.load_files.assert_called_once()


@pytest.mark.asyncio
async def test_create_specification_with_gemini(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test create_specification_with_gemini function."""
    # Mock get_current_context to return None (no cached context)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    # Mock the generate_content to return a spec-like response
    mock_gemini_client.generate_content = AsyncMock(return_value="# Test Spec\n\n## Tasks\n- Task 1\n\n## Files to Create\n- file1.py\n\n## Files to Modify\n- file2.py")

    output_file = tmp_path / "test-spec.md"

    result = await create_specification_with_gemini(
        feature_description="test feature",
        spec_template="standard",
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "spec_path" in result
    assert "spec_content" in result
    assert "implementation_tasks" in result
    assert "estimated_complexity" in result
    assert output_file.exists()


@pytest.mark.asyncio
async def test_create_specification_with_context(mock_gemini_client, mock_codebase_loader, tmp_path):
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

    # Mock get_current_context to return cached context
    mock_gemini_client.get_current_context = Mock(return_value=(
        {
            "analysis": {
                "architecture_summary": "test",
                "relevant_files": ["file1.py"],
                "patterns_identified": ["pattern1"]
            }
        },
        "ctx_test123"
    ))

    result = await create_specification_with_gemini(
        feature_description="test feature",
        spec_template="minimal",
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "spec_path" in result
    # Should use analyze_with_context when context is provided
    assert mock_gemini_client.analyze_with_context.called


@pytest.mark.asyncio
@patch('hitoshura25_gemini_workflow_bridge.generator._get_git_diff')
async def test_review_code_with_gemini(mock_git_diff, mock_gemini_client, tmp_path):
    """Test review_code_with_gemini function."""
    # Mock get_current_context to return None (no cached context)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    mock_git_diff.return_value = "diff --git a/test.py"
    mock_gemini_client.generate_content = AsyncMock(return_value='{"issues_found": [], "summary": "Looks good", "has_blocking_issues": false, "recommendations": []}')

    output_file = tmp_path / "review.md"

    result = await review_code_with_gemini(
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


@pytest.mark.asyncio
async def test_review_code_with_files(mock_gemini_client, tmp_path):
    """Test review_code_with_gemini with specific files."""
    # Mock get_current_context to return None (no cached context)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    mock_gemini_client.generate_content = AsyncMock(return_value='{"issues_found": [], "summary": "Looks good", "has_blocking_issues": false, "recommendations": []}')

    output_file = tmp_path / "review.md"

    result = await review_code_with_gemini(
        files=[str(test_file)],
        review_focus=["security", "performance"],
        spec_path=None,
        output_path=str(output_file)
    )

    assert isinstance(result, dict)
    assert "review_path" in result


@pytest.mark.asyncio
async def test_generate_documentation_with_gemini(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test generate_documentation_with_gemini function."""
    # Mock get_current_context to return None (no cached context)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    mock_gemini_client.analyze_with_context = AsyncMock(return_value="# API Documentation\n\nTest documentation content")

    output_file = tmp_path / "api-docs.md"

    result = await generate_documentation_with_gemini(
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


@pytest.mark.asyncio
async def test_ask_gemini_simple(mock_gemini_client):
    """Test ask_gemini function without context."""
    mock_gemini_client.generate_content = AsyncMock(return_value="This is the answer")

    result = await ask_gemini(
        prompt="What is 2+2?",
        include_codebase_context=False,
        temperature=0.7
    )

    assert isinstance(result, dict)
    assert "response" in result
    assert "context_used" in result
    assert "token_count" in result
    assert result["context_used"] is False
    assert mock_gemini_client.generate_content.called


@pytest.mark.asyncio
async def test_ask_gemini_with_context(mock_gemini_client, mock_codebase_loader):
    """Test ask_gemini function with codebase context."""
    # Mock get_current_context to return None (no cached context, will auto-load)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    mock_gemini_client.analyze_with_context = AsyncMock(return_value="This is the answer with context")

    result = await ask_gemini(
        prompt="Explain this codebase",
        include_codebase_context=True,
        temperature=0.5
    )

    assert isinstance(result, dict)
    assert "response" in result
    assert result["context_used"] is True
    assert mock_gemini_client.analyze_with_context.called


@pytest.mark.asyncio
async def test_ask_gemini_with_cached_context(mock_gemini_client):
    """Test ask_gemini function with cached context (automatic reuse)."""
    # Mock get_current_context to return cached context
    mock_gemini_client.get_current_context = Mock(return_value=(
        {
            "analysis": {
                "architecture_summary": "test",
                "relevant_files": [],
                "patterns_identified": []
            }
        },
        "ctx_test123"
    ))
    mock_gemini_client.analyze_with_context = AsyncMock(return_value="Answer using cached context")

    result = await ask_gemini(
        prompt="Question about the code",
        include_codebase_context=True,
        temperature=0.7
    )

    assert isinstance(result, dict)
    assert result["context_used"] is True
    assert mock_gemini_client.get_current_context.called


@pytest.mark.asyncio
async def test_auto_load_context(mock_gemini_client, mock_codebase_loader):
    """Test _auto_load_context helper function."""
    from hitoshura25_gemini_workflow_bridge.generator import _auto_load_context

    # Mock the analysis response
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value='{"architecture_summary": "Test architecture", "relevant_files": ["file1.py"], "patterns_identified": ["pattern1"]}'
    )

    # Call the helper
    context, context_id = await _auto_load_context(
        focus_description="test feature"
    )

    # Verify it loaded files
    assert mock_codebase_loader.load_files.called
    assert mock_codebase_loader.get_project_structure.called

    # Verify it called analyze_with_context
    assert mock_gemini_client.analyze_with_context.called

    # Verify it cached the context
    assert mock_gemini_client.cache_context.called

    # Verify it returned formatted context and ID
    assert isinstance(context, str)
    assert isinstance(context_id, str)
    assert context_id.startswith("ctx_")


@pytest.mark.asyncio
async def test_create_spec_without_context_id_auto_loads(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that create_specification auto-loads codebase context."""
    # Mock get_current_context to return None (no cached context, will auto-load)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    # Mock responses
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\n\n## Tasks\n- Task 1"
    )

    output_file = tmp_path / "spec.md"

    # Call - should auto-load codebase
    result = await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(output_file)
    )

    # Verify auto-loading happened
    assert mock_codebase_loader.load_files.called
    assert mock_gemini_client.analyze_with_context.called

    # Verify result structure (no context_id returned - automatic reuse now!)
    assert "spec_path" in result
    assert "spec_content" in result
    assert "context_id" not in result  # No longer returned


@pytest.mark.asyncio
async def test_context_reuse_across_calls(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that context is automatically reused across multiple tool calls."""
    # First call - auto-loads and caches
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\n\n## Tasks\n- Task 1"
    )
    # First call has no current context
    mock_gemini_client.get_current_context = Mock(return_value=None)

    spec_output = tmp_path / "spec.md"
    spec_result = await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(spec_output)
    )

    # Verify first call loaded codebase
    initial_load_count = mock_codebase_loader.load_files.call_count
    assert initial_load_count == 1

    # Mock get_current_context to return cached context for second call
    mock_gemini_client.get_current_context = Mock(return_value=(
        {
            "files_content": {"test.py": "content"},
            "project_structure": "structure",
            "analysis": {
                "architecture_summary": "test",
                "relevant_files": [],
                "patterns_identified": []
            }
        },
        "ctx_auto123"
    ))

    # Second call - automatically reuses cached context (should NOT reload codebase)
    doc_output = tmp_path / "doc.md"
    doc_result = await generate_documentation_with_gemini(
        documentation_type="api",
        scope="test API",
        output_path=str(doc_output)
    )

    # Verify codebase was NOT reloaded on second call
    assert mock_codebase_loader.load_files.call_count == initial_load_count

    # Verify get_current_context was called (automatic reuse)
    assert mock_gemini_client.get_current_context.called


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


@pytest.mark.asyncio
async def test_cache_miss_fallback_in_create_spec(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that create_specification auto-loads when no cached context available."""
    # Mock cache miss - no current context
    mock_gemini_client.get_current_context = Mock(return_value=None)
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\\n\\n## Tasks\\n- Task 1"
    )

    output_file = tmp_path / "spec.md"

    # Call - should auto-load since no current context
    result = await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(output_file)
    )

    # Should have auto-loaded codebase when no cached context
    assert mock_codebase_loader.load_files.called
    assert mock_gemini_client.analyze_with_context.called

    # Should return valid result (no context_id in response anymore)
    assert "spec_path" in result
    assert "spec_content" in result


@pytest.mark.asyncio
async def test_cache_miss_fallback_in_review(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that review_code auto-loads when no cached context available."""
    # Mock cache miss - no current context
    mock_gemini_client.get_current_context = Mock(return_value=None)
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value='{"issues_found": [], "summary": "Looks good", "has_blocking_issues": false, "recommendations": []}'
    )

    with patch('hitoshura25_gemini_workflow_bridge.generator._get_git_diff', return_value="diff --git a/test.py"):
        output_file = tmp_path / "review.md"

        # Call - should auto-load since no current context
        result = await review_code_with_gemini(
            files=None,
            output_path=str(output_file)
        )

        # Should have auto-loaded codebase when no cached context
        assert mock_codebase_loader.load_files.called
        assert mock_gemini_client.analyze_with_context.called

        # Should return valid result (no context_id in response anymore)
        assert "review_path" in result
        assert "review_content" in result


@pytest.mark.asyncio
async def test_error_handling_in_auto_load(mock_gemini_client, mock_codebase_loader):
    """Test that _auto_load_context handles errors gracefully."""
    from hitoshura25_gemini_workflow_bridge.generator import _auto_load_context

    # Mock codebase loader to raise an exception
    mock_codebase_loader.load_files = Mock(side_effect=Exception("Filesystem error"))

    # Should not raise exception, should return fallback context
    context, context_id = await _auto_load_context(focus_description="test")

    # Should return fallback context with error info
    assert isinstance(context, str)
    assert "Context Loading Failed" in context
    assert "Filesystem error" in context

    # Should return error context_id
    assert isinstance(context_id, str)
    assert context_id.startswith("ctx_error_")


@pytest.mark.asyncio
async def test_spec_creation_with_error_fallback(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that create_specification continues with fallback context on load error."""
    # Mock get_current_context to return None (no cached context, will try to auto-load)
    mock_gemini_client.get_current_context = Mock(return_value=None)

    # Mock codebase loader to raise an exception
    mock_codebase_loader.load_files = Mock(side_effect=Exception("Disk full"))

    # Mock Gemini to still respond (with fallback context)
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Spec\\n\\n## Tasks\\n- Task 1"
    )

    output_file = tmp_path / "spec.md"

    # Should not raise exception, should continue with fallback context
    result = await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(output_file)
    )

    # Should still return valid result (context_id no longer returned)
    assert "spec_path" in result
    assert "spec_content" in result


@pytest.mark.asyncio
async def test_create_specification_handles_empty_response(tmp_path, mock_codebase_loader):
    """Test that create_specification_with_gemini handles empty CLI responses."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        mock_client.analyze_with_context = AsyncMock(return_value="")  # Empty response
        mock_client.get_current_context = Mock(return_value=None)
        mock_client_fn.return_value = mock_client

        output_file = tmp_path / "spec.md"

        # Should raise RuntimeError with clear message
        result = await create_specification_with_gemini(
            feature_description="test feature",
            output_path=str(output_file)
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("spec_content", "")


@pytest.mark.asyncio
async def test_review_code_handles_none_response(tmp_path, mock_codebase_loader):
    """Test that review_code_with_gemini handles None/empty responses."""
    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        # Simulate empty response (though validation should catch this)
        mock_client.analyze_with_context = AsyncMock(return_value="   ")  # Whitespace only
        mock_client.get_current_context = Mock(return_value=None)
        mock_client_fn.return_value = mock_client

        output_file = tmp_path / "review.md"

        # Should raise RuntimeError with clear message
        result = await review_code_with_gemini(
            files=[str(test_file)],
            output_path=str(output_file)
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("review_content", "")


@pytest.mark.asyncio
async def test_analyze_codebase_handles_empty_response(mock_codebase_loader):
    """Test that analyze_codebase_with_gemini handles empty responses."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        mock_client.analyze_with_context = AsyncMock(return_value="")  # Empty response
        mock_client.cache_context = Mock()
        mock_client_fn.return_value = mock_client

        # Should raise RuntimeError with clear message
        result = await analyze_codebase_with_gemini(
            focus_description="test analysis"
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("analysis", "")


@pytest.mark.asyncio
async def test_generate_documentation_handles_empty_response(tmp_path, mock_codebase_loader):
    """Test that generate_documentation_with_gemini handles empty responses."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        mock_client.analyze_with_context = AsyncMock(return_value="")  # Empty response
        mock_client.get_current_context = Mock(return_value=None)
        mock_client_fn.return_value = mock_client

        output_file = tmp_path / "docs.md"

        # Should raise RuntimeError with clear message
        result = await generate_documentation_with_gemini(
            documentation_type="api",
            scope="test",
            output_path=str(output_file)
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("doc_content", "")


@pytest.mark.asyncio
async def test_ask_gemini_handles_empty_response_with_context(mock_codebase_loader):
    """Test that ask_gemini handles empty responses when using context."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        mock_client.analyze_with_context = AsyncMock(return_value="")  # Empty response
        mock_client.get_current_context = Mock(return_value=None)
        mock_client_fn.return_value = mock_client

        # Should raise RuntimeError with clear message
        result = await ask_gemini(
            prompt="test question",
            include_codebase_context=True
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("response", "")


@pytest.mark.asyncio
async def test_ask_gemini_handles_empty_response_without_context():
    """Test that ask_gemini handles empty responses without context."""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock_client_fn:
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(return_value="")  # Empty response
        mock_client_fn.return_value = mock_client

        # Should raise RuntimeError with clear message
        result = await ask_gemini(
            prompt="test question",
            include_codebase_context=False
        )

        # Should return error result (wrapped in try-except)
        assert "error" in result
        assert "empty" in result["error"].lower() or "Empty" in result.get("response", "")
