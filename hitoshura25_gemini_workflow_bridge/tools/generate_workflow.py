"""Tool for generating complete, executable workflows for features."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader


async def generate_feature_workflow(
    feature_description: str,
    workflow_style: Literal["interactive", "automated", "template"] = "interactive",
    save_to: Optional[str] = None,
    include_validation_steps: bool = True
) -> Dict[str, Any]:
    """
    Create a complete, executable workflow for a feature (implements "progressive disclosure").

    This tool generates a markdown workflow file that Claude can follow step-by-step,
    implementing progressive disclosure by loading steps on-demand rather than all at once.

    Args:
        feature_description: Description of the feature to implement
        workflow_style: Style of workflow (interactive, automated, or template)
        save_to: Path to save workflow file (default: .claude/workflows/)
        include_validation_steps: Include validation steps in workflow

    Returns:
        Dictionary with workflow_path, workflow_content, estimated_steps, and tools_required

    Raises:
        ValueError: If Gemini returns invalid JSON or missing required keys
        Exception: If workflow generation fails due to client errors or file I/O errors
    """
    # Initialize clients
    gemini_client = GeminiClient()
    codebase_loader = CodebaseLoader()

    # Load codebase for context
    files_content = codebase_loader.load_files(
        file_patterns=["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.java", "*.go", "*.md"],
        exclude_patterns=["node_modules/", "dist/", "build/", "__pycache__/"]
    )

    # Determine which template to use
    # Simple heuristic: if description contains "refactor", use refactor template
    is_refactor = "refactor" in feature_description.lower()
    template_name = "refactor_template.md" if is_refactor else "feature_template.md"
    template_path = Path(__file__).parent.parent / "workflows" / template_name
    template_content = template_path.read_text()

    # Build context about codebase (simplified since we don't have get_project_structure)
    context_parts = [
        "# Files in Project",
        "\n".join([f"- {path}" for path in files_content.keys()])
    ]
    context = "\n".join(context_parts)

    # Build prompt to customize workflow
    task_type = "refactoring" if is_refactor else "implementing"
    prompt = f"""Create a detailed workflow for {task_type} this feature: {feature_description}

Use this template as a starting point:
{template_content}

Based on the project structure and files, customize the workflow by:
1. Identifying specific questions to ask in the analysis phase
2. Listing specific files that will need to be created or modified
3. Identifying relevant dependencies that might be needed
4. Estimating the number of steps and time required

Provide your response as JSON with this structure:
{{
  "workflow_content": "complete markdown workflow",
  "estimated_steps": number,
  "estimated_time_minutes": number,
  "tools_required": ["tool1", "tool2"],
  "files_to_create": ["file1", "file2"],
  "files_to_modify": ["file1", "file2"],
  "dependencies": ["dep1", "dep2"]
}}"""

    # Build complete prompt with context
    full_prompt = f"""Context:
{context}

Task:
{prompt}"""

    # Query Gemini using generate_content method
    response = await gemini_client.generate_content(
        prompt=full_prompt,
        temperature=0.7
    )

    # Parse response
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        # If Gemini returns non-JSON response, fail fast
        raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}. Response: {response[:200]}") from e

    # Validate required keys
    if "workflow_content" not in result:
        raise ValueError(
            f"Gemini response missing required 'workflow_content' key. "
            f"Response keys: {list(result.keys())}"
        )

    # Determine save path
    if not save_to:
        workflow_dir = Path(os.getenv("DEFAULT_WORKFLOW_DIR", "./.claude/workflows"))
        workflow_dir.mkdir(parents=True, exist_ok=True)
        feature_slug = feature_description.lower().replace(' ', '-')[:50]
        save_to = str(workflow_dir / f"{feature_slug}.md")

    # Save workflow
    workflow_file = Path(save_to)
    workflow_file.parent.mkdir(parents=True, exist_ok=True)
    workflow_file.write_text(result["workflow_content"])

    return {
        "workflow_path": str(save_to),
        "workflow_content": result["workflow_content"],
        "estimated_steps": result.get("estimated_steps", 20),
        "estimated_time_minutes": result.get("estimated_time_minutes", 60),
        "tools_required": result.get("tools_required", []),
        "files_to_create": result.get("files_to_create", []),
        "files_to_modify": result.get("files_to_modify", []),
        "dependencies": result.get("dependencies", [])
    }
