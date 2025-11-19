"""JSON parsing utilities for handling Gemini CLI responses."""

import json
import re
from typing import Any


def strip_markdown_code_blocks(text: str) -> str:
    """
    Strip markdown code blocks from text.

    Gemini often returns JSON wrapped in markdown code blocks like:
    ```json
    {"key": "value"}
    ```

    This function removes the markdown wrapper to extract just the JSON.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Text with markdown code blocks removed
    """
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    # Pattern matches: optional language identifier, content, closing backticks
    pattern = r'^```(?:json|JSON)?\s*\n(.*?)\n```\s*$'
    match = re.match(pattern, text.strip(), re.DOTALL)

    if match:
        return match.group(1).strip()

    return text.strip()


def parse_json_response(response: str) -> Any:
    """
    Parse JSON response from Gemini CLI, handling markdown code blocks.

    Args:
        response: Response string from Gemini that may contain JSON

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If response is not valid JSON after cleanup
    """
    # First try direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try stripping markdown code blocks
        cleaned = strip_markdown_code_blocks(response)
        return json.loads(cleaned)
