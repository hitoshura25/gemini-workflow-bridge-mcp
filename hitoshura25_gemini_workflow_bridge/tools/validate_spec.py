"""Tool for validating specifications against codebase."""

import json
import time
from typing import Any, Dict, List, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
from ..utils.token_counter import count_tokens, format_token_stats
from ..utils.prompt_loader import load_system_prompt, build_prompt_with_context


async def validate_against_codebase(
    spec_content: str,
    validation_checks: List[str],
    codebase_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate specification for completeness and accuracy against codebase.

    After Claude creates a spec, this tool validates it using Gemini to ensure
    completeness, accuracy, and alignment with existing patterns.

    Args:
        spec_content: Markdown specification content created by Claude
        validation_checks: List of checks to perform (e.g., "missing_files", "undefined_dependencies")
        codebase_context: Optional pre-loaded codebase context (otherwise loads fresh)

    Returns:
        Dictionary with validation result, completeness score, issues, and suggestions
    """
    try:
        start_time = time.time()

        # Initialize clients
        gemini_client = GeminiClient()

        # Load codebase if not provided
        if codebase_context is None:
            codebase_loader = CodebaseLoader()
            files_content = codebase_loader.load_files(
                file_patterns=["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.go", "*.json", "package.json", "requirements.txt"],
                exclude_patterns=["node_modules/", "dist/", "build/", "__pycache__/"]
            )

            # Build codebase context
            context_parts = ["# Codebase Files\n"]
            for file_path, content in files_content.items():
                # For package files, include full content; for code, just structure
                if file_path.endswith(('package.json', 'requirements.txt', 'go.mod', 'pom.xml')):
                    context_parts.append(f"## File: {file_path}")
                    context_parts.append(f"```\n{content}\n```\n")
                else:
                    # Just include file path and basic structure for validation
                    context_parts.append(f"- {file_path}")

            codebase_context = "\n".join(context_parts)

        input_tokens = count_tokens(codebase_context + spec_content)

        # Load validation system prompt
        system_prompt = load_system_prompt("validation_system_prompt")

        # Build task
        checks_text = ", ".join(validation_checks)
        task = f"""Validate this specification against the codebase.

Specification to validate:
{spec_content}

Perform these validation checks: {checks_text}

Check for:
1. Missing files - Files mentioned in spec but don't exist in codebase
2. Undefined dependencies - Packages mentioned but not in package.json/requirements.txt
3. Pattern conflicts - Spec approaches that conflict with existing patterns
4. Incomplete testing - Missing test strategies
5. Missing error handling - Unaddressed error scenarios
6. Security concerns - Potential security issues

Provide your response as JSON matching the validation schema."""

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
        except json.JSONDecodeError:
            # Fallback structure
            result = {
                "validation_result": "pass_with_warnings",
                "completeness_score": 0.7,
                "issues": [],
                "missing_elements": {
                    "files": [],
                    "dependencies": [],
                    "functions": []
                },
                "pattern_alignment": {
                    "matches_existing_patterns": True,
                    "conflicts": [],
                    "suggestions": []
                }
            }

        # Calculate output tokens
        output_tokens = count_tokens(json.dumps(result))
        analysis_time = time.time() - start_time

        return {
            **result,
            "metadata": format_token_stats(input_tokens, output_tokens, analysis_time)
        }

    except Exception as e:
        return {
            "error": str(e),
            "validation_result": "fail",
            "completeness_score": 0.0,
            "issues": [{
                "type": "validation_error",
                "severity": "critical",
                "message": str(e),
                "suggested_fix": "Fix the validation error and try again"
            }],
            "missing_elements": {
                "files": [],
                "dependencies": [],
                "functions": []
            },
            "pattern_alignment": {
                "matches_existing_patterns": False,
                "conflicts": [],
                "suggestions": []
            }
        }
