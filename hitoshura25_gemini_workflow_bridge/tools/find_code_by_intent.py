"""Tool for semantic search that returns summaries with references."""

import json
import time
from typing import Any, Dict, Literal, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def find_code_by_intent(
    intent: str,
    return_format: Literal["summary_with_references", "detailed_with_snippets"] = "summary_with_references",
    max_files: int = 10,
    scope: Optional[str] = None
) -> Dict[str, Any]:
    """
    Semantic search that returns summaries, not full code (filtering at the edge).

    This tool uses Gemini to find code by natural language intent and returns
    compressed summaries instead of full code, demonstrating "filtering at the edge".

    Args:
        intent: Natural language description of what to find
        return_format: How to format results (summary or detailed)
        max_files: Limit number of files to return (default: 10)
        scope: Directory to search in (default: current directory)

    Returns:
        Dictionary with summary, primary files, patterns, and dependencies

    Raises:
        ValueError: If Gemini returns invalid JSON response
        Exception: If code search fails due to client errors or file loading errors
    """
    start_time = time.time()

    # Initialize clients
    gemini_client = GeminiClient()
    codebase_loader = CodebaseLoader()

    # Load codebase
    directories = [scope] if scope else None
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

    # Load fact extraction system prompt
    system_prompt = load_system_prompt("fact_extraction_system_prompt")

    # Build task
    task = f"""Find code related to: {intent}

Return at most {max_files} most relevant files.

Provide your response as JSON with this structure:
{{
  "summary": "High-level summary of findings",
  "primary_files": [
    {{
      "path": "file/path",
      "purpose": "What this file does",
      "key_functions": ["function1", "function2"],
      "lines": "start-end line range",
      "relevance_score": 0.0-1.0
    }}
  ],
  "pattern": "Common pattern used (e.g., MVC, Factory)",
  "dependencies": ["library1", "library2"],
  "related_searches": ["suggested follow-up intent 1"]
}}"""

    # Add code snippets request for detailed format
    if return_format == "detailed_with_snippets":
        task += "\n\nInclude code snippets for the most relevant sections."

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
