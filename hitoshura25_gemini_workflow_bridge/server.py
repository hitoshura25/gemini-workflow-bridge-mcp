#!/usr/bin/env python3
"""
MCP Server for hitoshura25-gemini-workflow-bridge.

MCP server that bridges Claude Code to Gemini CLI for context compression and factual code analysis.

Key Features:
- Gemini acts as a "context compression engine" (fact extraction only)
- Claude handles all reasoning, planning, and specification creation
- Tools for querying, tracing, validating, and workflow generation
"""

from mcp.server.fastmcp import FastMCP
from typing import List
import json

from . import generator
from .resources import WorkflowResources
from .tools import (
    query_codebase,
    find_code_by_intent,
    trace_feature,
    list_error_patterns,
    validate_against_codebase,
    check_consistency,
    generate_feature_workflow,
    generate_slash_command,
    setup_workflows
)
from .utils import validate_enum_parameter

# Initialize FastMCP server
mcp = FastMCP("hitoshura25_gemini_workflow_bridge")

# Initialize resources
workflow_resources = WorkflowResources()


# ============================================================================
# Tier 1: Fact Extraction Tools
# ============================================================================

@mcp.tool()
async def query_codebase_tool(
    questions: List[str],
    scope: str = None,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    max_tokens_per_answer: int = 300
) -> str:
    """Multi-question factual analysis with massive context compression

    This tool uses Gemini to analyze codebases and extract factual information.
    It compresses large codebases (50K+ tokens) into small summaries (300 tokens
    per answer) for Claude to use in planning.

    Args:
        questions: List of 1-10 specific questions to answer
        scope: Directory to analyze (default: current directory)
        include_patterns: File patterns to include
        exclude_patterns: Patterns to exclude
        max_tokens_per_answer: Target token budget per answer (default: 300)

    Returns:
        JSON string with answers array and compression metadata
    """
    result = await query_codebase(
        questions=questions,
        scope=scope,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_tokens_per_answer=max_tokens_per_answer
    )
    return json.dumps(result)


@mcp.tool()
async def find_code_by_intent_tool(
    intent: str,
    return_format: str = "summary_with_references",
    max_files: int = 10,
    scope: str = None
) -> str:
    """Semantic search that returns summaries with references (filtering at the edge)

    Find code by natural language intent and return compressed summaries instead
    of full code, demonstrating "filtering at the edge".

    Args:
        intent: Natural language description of what to find
        return_format: "summary_with_references" or "detailed_with_snippets"
        max_files: Limit number of files to return (default: 10)
        scope: Directory to search in

    Returns:
        JSON string with summary, primary files, patterns, dependencies
    """
    # Runtime validation (defense in depth - Literal types only provide static type checking)
    is_valid, error_msg = validate_enum_parameter(
        return_format,
        "return_format",
        ["summary_with_references", "detailed_with_snippets"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    result = await find_code_by_intent(
        intent=intent,
        return_format=return_format,
        max_files=max_files,
        scope=scope
    )
    return json.dumps(result)


@mcp.tool()
async def trace_feature_tool(
    feature: str,
    entry_point: str = None,
    max_depth: int = 10,
    include_data_flow: bool = False
) -> str:
    """Follow a feature's execution path through the codebase

    Trace how a feature is implemented across multiple files, showing the flow
    of execution and data transformations.

    Args:
        feature: Feature to trace (e.g., "user authentication")
        entry_point: Starting point like a route or function
        max_depth: How deep to trace (default: 10)
        include_data_flow: Include data transformations

    Returns:
        JSON string with flow steps, dependencies, DB operations, external calls
    """
    result = await trace_feature(
        feature=feature,
        entry_point=entry_point,
        max_depth=max_depth,
        include_data_flow=include_data_flow
    )
    return json.dumps(result)


@mcp.tool()
async def list_error_patterns_tool(
    pattern_type: str,
    directory: str = ".",
    group_by: str = "pattern"
) -> str:
    """Extract and categorize patterns across codebase (filtering at the edge)

    Analyze large codebases and extract pattern information, returning only
    compressed summaries instead of full code.

    Args:
        pattern_type: "error_handling", "logging", "async_patterns", "database_queries"
        directory: Directory to analyze
        group_by: "file", "pattern", or "severity"

    Returns:
        JSON string with patterns found, inconsistencies, and summary
    """
    # Runtime validation (defense in depth - Literal types only provide static type checking)
    is_valid, error_msg = validate_enum_parameter(
        pattern_type,
        "pattern_type",
        ["error_handling", "logging", "async_patterns", "database_queries"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    is_valid, error_msg = validate_enum_parameter(
        group_by,
        "group_by",
        ["file", "pattern", "severity"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    result = await list_error_patterns(
        pattern_type=pattern_type,
        directory=directory,
        group_by=group_by
    )
    return json.dumps(result)


# ============================================================================
# Tier 2: Validation Tools
# ============================================================================

@mcp.tool()
async def validate_against_codebase_tool(
    spec_content: str,
    validation_checks: List[str],
    codebase_context: str = None
) -> str:
    """Validate specification for completeness and accuracy

    After Claude creates a spec, this validates it using Gemini to ensure
    completeness, accuracy, and alignment with existing patterns.

    Args:
        spec_content: Markdown specification content
        validation_checks: List of checks (e.g., ["missing_files", "undefined_dependencies"])
        codebase_context: Optional pre-loaded context

    Returns:
        JSON string with validation result, completeness score, issues, suggestions
    """
    result = await validate_against_codebase(
        spec_content=spec_content,
        validation_checks=validation_checks,
        codebase_context=codebase_context
    )
    return json.dumps(result)


@mcp.tool()
async def check_consistency_tool(
    focus: str,
    new_code_or_spec: str,
    scope: str = None
) -> str:
    """Verify new code or spec follows existing codebase patterns

    Check whether new code or specifications align with existing patterns,
    conventions, and practices in the codebase.

    Args:
        focus: "naming_conventions", "error_handling", "testing", "api_design", "all"
        new_code_or_spec: Code or spec to check
        scope: Which part of codebase to compare against

    Returns:
        JSON string with consistency score, matches, violations, recommendations
    """
    # Runtime validation (defense in depth - Literal types only provide static type checking)
    is_valid, error_msg = validate_enum_parameter(
        focus,
        "focus",
        ["naming_conventions", "error_handling", "testing", "api_design", "all"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    result = await check_consistency(
        focus=focus,
        new_code_or_spec=new_code_or_spec,
        scope=scope
    )
    return json.dumps(result)


# ============================================================================
# Tier 3: Workflow Automation Tools
# ============================================================================

@mcp.tool()
async def generate_feature_workflow_tool(
    feature_description: str,
    workflow_style: str = "interactive",
    save_to: str = None,
    include_validation_steps: bool = True
) -> str:
    """Generate complete, executable workflow for a feature (progressive disclosure)

    Generate a markdown workflow file that Claude can follow step-by-step,
    implementing progressive disclosure.

    Args:
        feature_description: Description of the feature to implement
        workflow_style: "interactive", "automated", or "template"
        save_to: Path to save workflow file
        include_validation_steps: Include validation steps

    Returns:
        JSON string with workflow_path, content, estimated_steps, tools_required
    """
    # Runtime validation (defense in depth - Literal types only provide static type checking)
    is_valid, error_msg = validate_enum_parameter(
        workflow_style,
        "workflow_style",
        ["interactive", "automated", "template"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    result = await generate_feature_workflow(
        feature_description=feature_description,
        workflow_style=workflow_style,
        save_to=save_to,
        include_validation_steps=include_validation_steps
    )
    return json.dumps(result)


@mcp.tool()
async def generate_slash_command_tool(
    command_name: str,
    workflow_type: str,
    description: str,
    steps: List[str] = None,
    save_to: str = None
) -> str:
    """Auto-generate Claude Code slash commands for common workflows

    Create custom slash commands that users can invoke in Claude Code to
    automate complete workflows.

    Args:
        command_name: Name of the command (e.g., "add-feature")
        workflow_type: "feature", "refactor", "debug", "review", "custom"
        description: Description of what the command does
        steps: Custom steps if workflow_type="custom"
        save_to: Where to save command file

    Returns:
        JSON string with command_path, command_content, usage_example
    """
    # Runtime validation (defense in depth - Literal types only provide static type checking)
    is_valid, error_msg = validate_enum_parameter(
        workflow_type,
        "workflow_type",
        ["feature", "refactor", "debug", "review", "custom"]
    )
    if not is_valid:
        return json.dumps({"error": error_msg})

    result = await generate_slash_command(
        command_name=command_name,
        workflow_type=workflow_type,
        description=description,
        steps=steps,
        save_to=save_to
    )
    return json.dumps(result)


@mcp.tool()
async def setup_workflows_tool(
    workflows: List[str] = None,
    output_dir: str = None,
    overwrite: bool = False,
    include_commands: bool = True
) -> str:
    """Set up recommended workflow files and slash commands for the Gemini MCP Server

    Automatically generates the recommended workflow files and slash commands,
    including the spec-only workflow. This makes it easy to start using workflows
    immediately after installation.

    Args:
        workflows: List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']
                  Default: ['spec-only']
        output_dir: Base directory for outputs (workflows go in .claude/workflows/, commands in .claude/commands/)
                   Default: current directory
        overwrite: Whether to overwrite existing files. Default: False
        include_commands: Whether to also create slash commands for the workflows. Default: True

    Returns:
        JSON string with success status, workflows_created, skipped items, and message
    """
    result = await setup_workflows(
        workflows=workflows,
        output_dir=output_dir,
        overwrite=overwrite,
        include_commands=include_commands
    )
    return json.dumps(result, indent=2)


# ============================================================================
# Legacy Tools (Maintained for backward compatibility)
# ============================================================================

@mcp.tool()
async def analyze_codebase_with_gemini(
    focus_description: str,
    directories: str = None,
    file_patterns: str = None,
    exclude_patterns: str = None
) -> str:
    """Analyze codebase using Gemini - returns FACTS only

    This tool uses the fact extraction system prompt to return only factual
    information, not opinions or suggestions.

    Consider using query_codebase_tool() for multi-question analysis with
    better token compression.

    Args:
        focus_description: What to focus on in the analysis
        directories: Directories to analyze
        file_patterns: File patterns to include
        exclude_patterns: Patterns to exclude

    Returns:
        JSON string with factual analysis (no opinions or suggestions)
    """
    result = await generator.analyze_codebase_with_gemini(
        focus_description=focus_description,
        directories=directories,
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns
    )
    return str(result)


@mcp.tool()
async def ask_gemini(

    prompt: str,

    include_codebase_context: bool = None,

    temperature: float = None

) -> str:
    """General-purpose Gemini query with optional codebase context

    By default, queries are answered without codebase context. Set
    include_codebase_context=True to automatically load and use your
    codebase (or reuse recently cached context).

    Args:

        prompt: Question or task for Gemini

        include_codebase_context: Load full codebase context (default: False)

        temperature: Temperature for generation 0.0-1.0 (default: 0.7)



    Returns:
        JSON string containing response, context_used, and token_count
    """
    result = await generator.ask_gemini(

        prompt=prompt,

        include_codebase_context=include_codebase_context,

        temperature=temperature

    )
    return str(result)


# Resource handlers
@mcp.resource("workflow://specs/{name}")
def read_spec_resource(name: str) -> str:
    """Read a specification resource"""
    uri = f"workflow://specs/{name}"
    resource = workflow_resources.read_resource(uri)
    return resource["text"]


@mcp.resource("workflow://reviews/{name}")
def read_review_resource(name: str) -> str:
    """Read a review resource"""
    uri = f"workflow://reviews/{name}"
    resource = workflow_resources.read_resource(uri)
    return resource["text"]


@mcp.resource("workflow://context/{name}")
def read_context_resource(name: str) -> str:
    """Read a cached context resource"""
    uri = f"workflow://context/{name}"
    resource = workflow_resources.read_resource(uri)
    return resource["text"]


def main():
    """Main entry point for MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()