"""Tool for checking consistency of new code/spec with existing patterns."""

import json
import time
from typing import Any, Dict, Literal, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def check_consistency(
    focus: Literal["naming_conventions", "error_handling", "testing", "api_design", "all"],
    new_code_or_spec: str,
    scope: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify new code or spec follows existing codebase patterns.

    This tool checks whether new code or specifications align with existing
    patterns, conventions, and practices in the codebase.

    Args:
        focus: What aspect to check for consistency
        new_code_or_spec: Code or spec to check
        scope: Which part of codebase to compare against (default: all)

    Returns:
        Dictionary with consistency score, matches, violations, and recommendations

    Raises:
        ValueError: If Gemini returns invalid JSON response
        Exception: If consistency check fails due to client errors or file loading errors
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
    context_parts = ["# Existing Codebase\n"]
    for file_path, content in files_content.items():
        context_parts.append(f"## File: {file_path}")
        context_parts.append(f"```\n{content}\n```\n")

    codebase_context = "\n".join(context_parts)
    input_tokens = count_tokens(codebase_context + new_code_or_spec)

    # Load fact extraction system prompt (for pattern analysis)
    system_prompt = load_system_prompt("fact_extraction_system_prompt")

    # Build task based on focus
    focus_instructions = {
        "naming_conventions": "naming patterns (variables, functions, classes, files)",
        "error_handling": "error handling patterns (try/catch, error classes, logging)",
        "testing": "testing patterns (test structure, mocking, assertions)",
        "api_design": "API design patterns (routes, controllers, response formats)",
        "all": "all patterns (naming, error handling, testing, API design)"
    }

    task = f"""Analyze consistency between the existing codebase and new code/spec.

Focus on: {focus_instructions[focus]}

New code or spec to check:
{new_code_or_spec}

Provide your response as JSON with this structure:
{{
  "consistency_score": 0.0-1.0,
  "matches": [
    {{
      "pattern": "pattern name",
      "example_from_codebase": "example",
      "example_from_new_code": "example",
      "aligned": true/false
    }}
  ],
  "violations": [
    {{
      "pattern_expected": "what pattern should be",
      "pattern_found": "what was actually found",
      "location": "where in new code",
      "fix_suggestion": "how to fix"
    }}
  ],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}

Be specific about what patterns exist in the codebase and how the new code aligns or diverges."""

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

    # Calculate output tokens
    output_tokens = count_tokens(json.dumps(result))
    analysis_time = time.time() - start_time

    return {
        **result,
        "metadata": format_token_stats(input_tokens, output_tokens, analysis_time)
    }
