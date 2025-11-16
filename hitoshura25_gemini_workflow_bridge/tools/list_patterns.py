"""Tool for extracting and categorizing patterns across codebase."""

import json
import time
from typing import Any, Dict, Literal

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def list_error_patterns(
    pattern_type: Literal["error_handling", "logging", "async_patterns", "database_queries"],
    directory: str = ".",
    group_by: Literal["file", "pattern", "severity"] = "pattern"
) -> Dict[str, Any]:
    """
    Extract and categorize patterns across codebase (demonstrates "filtering at the edge").

    This tool analyzes large codebases and extracts pattern information, returning
    only compressed summaries instead of full code. This demonstrates the "filtering
    at the edge" concept from Anthropic's MCP article.

    Args:
        pattern_type: Type of pattern to extract
        directory: Directory to analyze
        group_by: How to group results

    Returns:
        Dictionary with patterns found, inconsistencies, and summary

    Raises:
        ValueError: If Gemini returns invalid JSON response
        Exception: If pattern extraction fails due to client errors or file loading errors
    """
    start_time = time.time()

    # Initialize clients
    gemini_client = GeminiClient()
    codebase_loader = CodebaseLoader()

    # Load codebase
    # If directory is ".", pass None to search current directory
    # Otherwise, pass as a list
    directories = None if directory == "." else [directory]
    files_content = codebase_loader.load_files(
        file_patterns=["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.go"],
        exclude_patterns=["node_modules/", "dist/", "build/", "__pycache__/"],
        directories=directories
    )

    # Build codebase context
    context_parts = ["# Codebase Files\n"]
    for file_path, content in files_content.items():
        context_parts.append(f"## File: {file_path}")
        context_parts.append(f"```\n{content}\n```\n")

    codebase_context = "\n".join(context_parts)
    input_tokens = count_tokens(codebase_context)

    # Load pattern extraction system prompt
    system_prompt = load_system_prompt("pattern_extraction_prompt")

    # Build task based on pattern type
    pattern_descriptions = {
        "error_handling": "error handling patterns (try/catch, throw, reject, error classes)",
        "logging": "logging patterns (console.log, logger calls, log levels)",
        "async_patterns": "async patterns (async/await, promises, callbacks)",
        "database_queries": "database query patterns (ORM calls, raw SQL, query builders)"
    }

    task = f"""Analyze the codebase for {pattern_descriptions[pattern_type]}.

Group results by: {group_by}

Provide your response as JSON with this structure:
{{
  "patterns_found": {{
    "pattern_name": {{
      "count": number,
      "example": "code example",
      "files": ["file1.js", "file2.js"],
      "recommendation": "optional suggestion"
    }}
  }},
  "inconsistencies": [
    {{
      "issue": "description",
      "locations": ["file:line"],
      "suggestion": "how to standardize"
    }}
  ],
  "summary": "High-level summary of findings"
}}

Focus on COUNTING and CATEGORIZING patterns, not judging them.
Identify inconsistencies where different patterns are used for the same purpose."""

    # Build complete prompt
    full_prompt = build_prompt_with_context(
        system_prompt=system_prompt,
        user_task=task,
        context=codebase_context
    )

    # Query Gemini
    response = await gemini_client.generate_content(
        prompt=full_prompt,
        temperature=0.3
    )

    # Parse response
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        # If Gemini returns non-JSON response, fail fast
        raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}. Response: {response[:200]}") from e

    # Calculate output tokens and compression
    output_tokens = count_tokens(json.dumps(result))
    analysis_time = time.time() - start_time

    return {
        **result,
        "metadata": format_token_stats(input_tokens, output_tokens, analysis_time)
    }
