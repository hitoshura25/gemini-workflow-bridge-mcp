"""Tool for querying codebase with multiple questions and returning compressed facts."""

import json
import time
from typing import Any, Dict, List, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def query_codebase(
    questions: List[str],
    scope: Optional[str] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_tokens_per_answer: int = 300
) -> Dict[str, Any]:
    """
    Multi-question factual analysis with massive context compression.

    This tool uses Gemini to analyze codebases and extract factual information.
    It's designed to compress large codebases (50K+ tokens) into small, high-signal
    summaries (300 tokens per answer) for Claude to use in planning.

    Args:
        questions: List of 1-10 specific questions to answer
        scope: Directory to analyze (default: current directory)
        include_patterns: File patterns to include (default: common code files)
        exclude_patterns: Exclude patterns (default: node_modules, etc.)
        max_tokens_per_answer: Target token budget per answer (default: 300)

    Returns:
        Dictionary with answers array and metadata including compression ratio

    Raises:
        ValueError: If Gemini returns invalid JSON response
        Exception: If querying fails due to client errors or file loading errors
    """
    start_time = time.time()

    # Initialize clients
    gemini_client = GeminiClient()
    codebase_loader = CodebaseLoader()

    # Set default patterns
    if include_patterns is None:
        include_patterns = [
            "*.py", "*.js", "*.ts", "*.tsx", "*.jsx",
            "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h"
        ]

    if exclude_patterns is None:
        exclude_patterns = [
            "node_modules/", "dist/", "build/", "__pycache__/",
            ".git/", "venv/", ".venv/", "vendor/"
        ]

    # Load codebase
    directories = [scope] if scope else None
    files_content = codebase_loader.load_files(
        file_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
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

    # Build task for questions
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    task = f"""Answer the following questions about the codebase. For each question, provide:
- Factual statements with file:line references
- Code snippets only if specifically requested
- Keep each answer under {max_tokens_per_answer} tokens

Questions:
{questions_text}

Provide your response as a JSON array with this structure:
[
  {{
    "question": "the question",
    "facts": ["fact 1 with file:line", "fact 2 with file:line"],
    "code_snippets": [
      {{"file": "path", "lines": "42-56", "snippet": "code"}}
    ],
    "file_count": number,
    "instance_count": number
  }}
]"""

    # Build complete prompt
    full_prompt = build_prompt_with_context(
        system_prompt=system_prompt,
        user_task=task,
        context=codebase_context
    )

    # Query Gemini
    response = await gemini_client.generate_content(
        prompt=full_prompt,
        temperature=0.3  # Low temperature for factual extraction
    )

    # Parse response
    try:
        answers = json.loads(response)
        if not isinstance(answers, list):
            # If response is a dict with "answers" key, extract it
            if isinstance(answers, dict) and "answers" in answers:
                answers = answers["answers"]
            else:
                # Wrap single answer in array
                answers = [answers]
    except json.JSONDecodeError as e:
        # If Gemini returns non-JSON response, fail fast
        raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}. Response: {response[:200]}") from e

    # Calculate output tokens and compression
    output_tokens = count_tokens(json.dumps(answers))
    analysis_time = time.time() - start_time

    return {
        "answers": answers,
        "metadata": format_token_stats(input_tokens, output_tokens, analysis_time)
    }
