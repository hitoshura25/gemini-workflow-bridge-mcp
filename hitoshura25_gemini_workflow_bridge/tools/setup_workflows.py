"""Tool for setting up recommended workflow files and slash commands."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workflow_templates import WORKFLOW_TEMPLATES


async def setup_workflows(
    workflows: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    include_commands: bool = True
) -> Dict[str, Any]:
    """
    Set up recommended workflow files and slash commands for the Gemini MCP Server.

    This tool automatically generates the recommended workflow files and slash commands,
    making it easy to start using features like the spec-only workflow immediately after
    installation.

    Args:
        workflows: List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']
                  Default: ['spec-only']
        output_dir: Base directory for outputs (workflows go in .claude/workflows/, commands in .claude/commands/)
                   Default: current directory
        overwrite: Whether to overwrite existing files. Default: False
        include_commands: Whether to also create slash commands for the workflows. Default: True

    Returns:
        Dictionary with success status, workflows_created, skipped items, and message
    """
    try:
        # Default to spec-only if not specified
        if workflows is None:
            workflows = ["spec-only"]

        # Expand 'all' to all available workflows, then remove duplicates
        if "all" in workflows:
            workflows = ["spec-only", "feature", "refactor", "review"]
        # Remove duplicates while preserving order
        workflows = list(dict.fromkeys(workflows))

        # Resolve base directory path
        if output_dir:
            base_dir = Path(output_dir).resolve()
        else:
            base_dir = Path.cwd()

        # Define output directories (respect environment variables)
        workflow_dir = base_dir / os.getenv("DEFAULT_WORKFLOW_DIR", ".claude/workflows")
        command_dir = base_dir / os.getenv("DEFAULT_COMMAND_DIR", ".claude/commands")

        # Check write permissions
        try:
            workflow_dir.mkdir(parents=True, exist_ok=True)
            if include_commands:
                command_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return {
                "success": False,
                "workflows_created": [],
                "skipped": [],
                "message": f"Permission denied: unable to create directories in {base_dir}"
            }

        results = {
            "success": True,
            "workflows_created": [],
            "skipped": [],
            "message": ""
        }

        # Process each workflow
        for workflow_name in workflows:
            # Validate workflow name against allowed list
            if workflow_name not in WORKFLOW_TEMPLATES:
                workflow_path = workflow_dir / f"{workflow_name}.md"
                command_path = command_dir / f"{workflow_name}.md" if include_commands else None
                results["skipped"].append({
                    "name": workflow_name,
                    "workflow_path": str(workflow_path.relative_to(base_dir)),
                    "command_path": str(command_path.relative_to(base_dir)) if command_path else None,
                    "reason": f"Unknown workflow type: {workflow_name}. Available: {list(WORKFLOW_TEMPLATES.keys())}"
                })
                continue

            template = WORKFLOW_TEMPLATES[workflow_name]
            workflow_path = workflow_dir / f"{workflow_name}.md"
            command_path = command_dir / f"{workflow_name}.md" if include_commands else None

            workflow_result = {
                "name": workflow_name,
                "workflow_path": str(workflow_path.relative_to(base_dir)),
                "command_path": str(command_path.relative_to(base_dir)) if command_path else None,
                "status": "created"
            }

            # Check if workflow file already exists
            if workflow_path.exists() and not overwrite:
                workflow_result["status"] = "skipped (already exists)"
                workflow_result["reason"] = "Workflow file already exists"
                results["skipped"].append(workflow_result)
                continue

            # Create workflow file
            try:
                workflow_path.write_text(template["workflow_content"])
            except (PermissionError, OSError) as e:
                results["skipped"].append({
                    "name": workflow_name,
                    "workflow_path": str(workflow_path.relative_to(base_dir)),
                    "command_path": str(command_path.relative_to(base_dir)) if command_path else None,
                    "reason": f"Failed to write workflow file: {str(e)}"
                })
                continue

            # Create command file if requested
            if include_commands and command_path:
                if command_path.exists() and not overwrite:
                    workflow_result["status"] = "workflow created, command skipped (already exists)"
                else:
                    try:
                        command_path.write_text(template["command_content"])
                    except (PermissionError, OSError) as e:
                        workflow_result["status"] = f"workflow created, command failed: {str(e)}"

            results["workflows_created"].append(workflow_result)

        # Generate summary message
        created_count = len(results["workflows_created"])
        skipped_count = len(results["skipped"])

        if created_count > 0:
            results["message"] = f"Successfully set up {created_count} workflow(s)"
            if include_commands:
                # Count how many commands were actually created (status == "created" means both workflow and command were created)
                command_created_count = sum(
                    1 for w in results["workflows_created"]
                    if w["status"] == "created"
                )
                results["message"] += f" and {command_created_count} command(s)"
        else:
            results["message"] = "No workflows were created"

        if skipped_count > 0:
            results["message"] += f", skipped {skipped_count} item(s)"

        # If nothing was created, mark as unsuccessful
        if created_count == 0:
            results["success"] = False

        return results

    except Exception as e:
        return {
            "success": False,
            "workflows_created": [],
            "skipped": [],
            "message": f"Error setting up workflows: {str(e)}"
        }
