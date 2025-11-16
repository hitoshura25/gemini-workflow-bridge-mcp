"""Tool for tracing feature execution flow through codebase."""

import json
import time
from typing import Any, Dict, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def trace_feature(
    feature: str,
    entry_point: Optional[str] = None,
    max_depth: int = 10,
    include_data_flow: bool = False
) -> Dict[str, Any]:
    """
    Follow a feature's execution path through the codebase.

    This tool traces how a feature is implemented across multiple files,
    showing the flow of execution and data transformations.

    Args:
        feature: Feature to trace (e.g., "user authentication")
        entry_point: Starting point like a route or function (optional)
        max_depth: How deep to trace (default: 10)
        include_data_flow: Include data transformations (default: False)

    Returns:
        Dictionary with flow steps, dependencies, DB operations, and external calls

    Raises:
        ValueError: If Gemini returns invalid JSON response
        Exception: If tracing fails due to client errors or file loading errors
    """
    start_time = time.time()

    # Initialize clients
    gemini_client = GeminiClient()
    codebase_loader = CodebaseLoader()

    # Load codebase
    files_content = codebase_loader.load_files(
        file_patterns=["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.go"],
        exclude_patterns=["node_modules/", "dist/", "build/", "__pycache__/"]
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
    data_flow_instruction = ""
    if include_data_flow:
        data_flow_instruction = """
For each step, also include:
- data_in: Input data type/structure
- data_out: Output data type/structure"""

    entry_point_text = f"\nStarting from: {entry_point}" if entry_point else ""

    task = f"""Trace the execution flow for: {feature}{entry_point_text}

Follow the code execution up to {max_depth} steps deep.{data_flow_instruction}

Provide your response as JSON with this structure:
{{
  "flow": [
    {{
      "step": 1,
      "file": "path/to/file",
      "function": "functionName",
      "description": "What this step does",
      "data_in": "input type (if include_data_flow=true)",
      "data_out": "output type (if include_data_flow=true)"
    }}
  ],
  "dependencies": ["library1", "library2"],
  "database_operations": [
    {{
      "table": "table_name",
      "operation": "read|write|update|delete",
      "location": "file:line"
    }}
  ],
  "external_calls": [
    {{
      "service": "service_name",
      "endpoint": "endpoint",
      "location": "file:line"
    }}
  ]
}}"""

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
