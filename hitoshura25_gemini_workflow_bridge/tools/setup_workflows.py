"""Tool for setting up recommended workflow files and slash commands."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workflow_templates import WORKFLOW_TEMPLATES
from ..resources import workflow_resources


async def setup_workflows(
    workflows: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    include_commands: bool = True,
    command_prefix: Optional[str] = None,
    workflow_prefix: Optional[str] = None
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
        command_prefix: Command prefix (default from GEMINI_COMMAND_PREFIX env var)
        workflow_prefix: Workflow prefix (default from GEMINI_WORKFLOW_PREFIX env var)

    Returns:
        Dictionary with success status, workflows_created, skipped items, and message
    """
    try:
        # Load prefixes (use parameter if provided, otherwise env var)
        cmd_prefix = command_prefix if command_prefix is not None else workflow_resources.command_prefix
        wf_prefix = workflow_prefix if workflow_prefix is not None else workflow_resources.workflow_prefix

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
                # Apply prefixes to filenames even for unknown workflows
                workflow_filename = f"{wf_prefix}{workflow_name}.md"
                command_filename = f"{cmd_prefix}{workflow_name}.md"
                workflow_path = workflow_dir / workflow_filename
                command_path = command_dir / command_filename if include_commands else None
                results["skipped"].append({
                    "name": workflow_name,
                    "prefixed_name": f"{wf_prefix}{workflow_name}",
                    "workflow_path": str(workflow_path.relative_to(base_dir)),
                    "command_path": str(command_path.relative_to(base_dir)) if command_path else None,
                    "reason": f"Unknown workflow type: {workflow_name}. Available: {list(WORKFLOW_TEMPLATES.keys())}"
                })
                continue

            template = WORKFLOW_TEMPLATES[workflow_name]

            # Apply prefixes to filenames
            workflow_filename = f"{wf_prefix}{workflow_name}.md"
            command_filename = f"{cmd_prefix}{workflow_name}.md"

            workflow_path = workflow_dir / workflow_filename
            command_path = command_dir / command_filename if include_commands else None

            workflow_result = {
                "name": workflow_name,
                "prefixed_name": f"{wf_prefix}{workflow_name}",
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

            # Get template content
            workflow_content = template["workflow_content"]
            command_content = template["command_content"]

            # Replace command name placeholders in templates
            # The command name used in slash commands should include the prefix
            prefixed_command = f"{cmd_prefix}{workflow_name}"
            command_content = command_content.replace("/{COMMAND_NAME}", f"/{prefixed_command}")
            command_content = command_content.replace("{COMMAND_NAME}", prefixed_command)

            # Create workflow file
            try:
                workflow_path.write_text(workflow_content)
            except (PermissionError, OSError) as e:
                results["skipped"].append({
                    "name": workflow_name,
                    "prefixed_name": f"{wf_prefix}{workflow_name}",
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
                        command_path.write_text(command_content)
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
