"""
Tests for GeminiClient error handling and response validation.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient


@pytest.mark.asyncio
async def test_generate_content_empty_stdout():
    """Test that empty stdout raises clear error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_none_response():
    """Test that None response value raises clear error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": null}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": null}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="None response"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_empty_string_response():
    """Test that empty string response raises clear error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": ""}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": ""}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response string"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_whitespace_only_response():
    """Test that whitespace-only response raises clear error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": "   \n\t   "}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": "   \\n\\t   "}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response string"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_json_decode_error_empty():
    """Test that JSON decode error with empty output raises clear error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Invalid JSON that's empty (whitespace only)
    mock_process.communicate = AsyncMock(return_value=(b"   ", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_valid_response():
    """Test that valid responses work correctly."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": "Valid response text"}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await client.generate_content("test prompt")
        assert result == "Valid response text"


@pytest.mark.asyncio
async def test_generate_content_with_nodejs_warnings():
    """Test that Node.js warnings in stderr don't cause failures when command succeeds."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Simulate Node.js warning in stderr but successful response in stdout
    warning_stderr = b"""Warning: Detected unsettled top-level await at
     file:///opt/homebrew/Cellar/gemini-cli/0.16.0/libexec/lib/node_modules/@google/gemini-cli/node_modules/yoga-layout/dist/src/index.js:13
     const Yoga = wrapAssembly(await loadYoga());"""
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": "Valid response despite warnings"}', warning_stderr)
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await client.generate_content("test prompt")
        assert result == "Valid response despite warnings"


@pytest.mark.asyncio
async def test_generate_content_warnings_only_on_failure():
    """Test that warnings-only stderr with non-zero exit code doesn't raise error."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 1  # Non-zero exit code
    # Only warnings in stderr, no actual errors
    warning_stderr = b"""Warning: Detected unsettled top-level await at
     file:///path/to/module.js:13
     const Foo = await loadFoo();"""
    mock_process.communicate = AsyncMock(
        return_value=(b"", warning_stderr)
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Warnings should be ignored (not cause "Gemini CLI error" exception)
        # However, the empty response will still cause an error
        with pytest.raises(RuntimeError, match="empty response"):
            # Expected: RuntimeError about empty response, NOT about warnings
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_real_errors_are_caught():
    """Test that actual errors in stderr are properly raised even with warnings present."""
    with patch('shutil.which', return_value='/usr/bin/gemini'):
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr='')
            client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 1  # Non-zero exit code
    # Mix of warnings and actual errors
    mixed_stderr = b"""Warning: Detected unsettled top-level await at
     file:///path/to/module.js:13
     const Foo = await loadFoo();
Error: Authentication failed
TypeError: Cannot read property 'data' of undefined"""
    mock_process.communicate = AsyncMock(
        return_value=(b"", mixed_stderr)
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Should raise error with the actual error messages, filtering out warnings
        with pytest.raises(RuntimeError, match="Error: Authentication failed"):
            await client.generate_content("test prompt")
