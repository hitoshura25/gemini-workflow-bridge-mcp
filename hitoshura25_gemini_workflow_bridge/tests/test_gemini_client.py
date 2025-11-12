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
