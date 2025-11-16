"""Tool for auto-generating Claude Code slash commands."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


async def generate_slash_command(
    command_name: str,
    workflow_type: Literal["feature", "refactor", "debug", "review", "custom"],
    description: str,
    steps: Optional[List[str]] = None,
    save_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Auto-generate Claude Code slash commands for common workflows (ultimate DX).

    This tool creates custom slash commands that users can invoke in Claude Code
    to automate complete workflows.

    Args:
        command_name: Name of the command (e.g., "add-feature")
        workflow_type: Type of workflow
        description: Description of what the command does
        steps: Custom steps if workflow_type="custom"
        save_to: Where to save command file (default: .claude/commands/)

    Returns:
        Dictionary with command_path, command_content, and usage_example

    Raises:
        Exception: If command generation fails due to file I/O errors or invalid parameters
    """
    # Define command templates
    command_templates = {
            "feature": """# /{command_name} - Complete Feature Implementation Workflow

## Description
{description}

This command:
1. Analyzes codebase context using Gemini
2. Generates detailed specification using Claude
3. Validates completeness
4. Implements with tests
5. Reviews for consistency

## Usage
/{command_name} <feature description>

## Examples
/{command_name} Add JWT refresh token support
/{command_name} Implement rate limiting for API
/{command_name} Add email verification for new users

## Steps (Auto-executed)

### 1. Gather Context
- Uses `query_codebase` to understand relevant parts of codebase
- Asks targeted questions based on feature type

### 2. Create Specification
- Claude creates detailed spec using Gemini's analysis
- Includes implementation steps, file changes, testing strategy

### 3. Validate Completeness
- Uses `validate_against_codebase` to check spec
- Ensures no missing dependencies or pattern conflicts

### 4. Implementation
- Follows spec step-by-step with task tracking
- Creates tests alongside implementation

### 5. Final Review
- Uses `check_consistency` to verify patterns
- Addresses any issues before completion

## Configuration
Set in `.claude/config.json`:
- `gemini_workflow.auto_validate`: true/false
- `gemini_workflow.testing_required`: true/false
""",
            "refactor": """# /{command_name} - Refactoring Workflow

## Description
{description}

This command automates the refactoring process with analysis, planning, implementation, and validation.

## Usage
/{command_name} <refactoring description>

## Examples
/{command_name} Extract authentication logic into service
/{command_name} Standardize error handling across controllers
/{command_name} Migrate from callbacks to async/await

## Steps (Auto-executed)

### 1. Analyze Current State
- Uses `list_error_patterns` or relevant pattern analysis
- Uses `trace_feature` to understand dependencies

### 2. Plan Refactoring
- Creates refactoring specification
- Validates against codebase
- Identifies breaking changes

### 3. Implementation
- Implements new approach
- Updates tests
- Maintains backward compatibility where possible

### 4. Validation
- Runs tests
- Checks consistency
- Verifies no regressions
""",
            "review": """# /{command_name} - Code Review Workflow

## Description
{description}

This command performs automated code review using Gemini's large context window.

## Usage
/{command_name} [files...]

## Examples
/{command_name}
/{command_name} src/services/*.js

## Steps (Auto-executed)

### 1. Load Context
- Analyzes codebase patterns
- Loads relevant specs and documentation

### 2. Review Code
- Checks for security issues
- Validates against patterns
- Identifies potential bugs

### 3. Report
- Generates detailed review report
- Categorizes issues by severity
- Provides actionable suggestions
""",
            "custom": """# /{command_name} - {description}

## Usage
/{command_name} <parameters>

## Steps
{custom_steps}
"""
    }

    # Get template
    if workflow_type == "custom" and steps:
        custom_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        template = command_templates["custom"].format(
            command_name=command_name,
            description=description,
            custom_steps=custom_steps
        )
    else:
        template = command_templates[workflow_type].format(
            command_name=command_name,
            description=description
        )

    # Determine save path
    if not save_to:
        command_dir = Path(os.getenv("DEFAULT_COMMAND_DIR", "./.claude/commands"))
        command_dir.mkdir(parents=True, exist_ok=True)
        save_to = str(command_dir / f"{command_name}.md")

    # Save command
    command_file = Path(save_to)
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(template)

    # Generate usage example
    usage_example = f"/{command_name} <description>"

    return {
        "command_path": str(save_to),
        "command_content": template,
        "usage_example": usage_example
    }
