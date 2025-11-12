#!/usr/bin/env python3
"""
MCP Server for hitoshura25-gemini-workflow-bridge.

MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review
"""

from mcp.server.fastmcp import FastMCP

from . import generator
from .resources import WorkflowResources

# Initialize FastMCP server
mcp = FastMCP("hitoshura25_gemini_workflow_bridge")

# Initialize resources
workflow_resources = WorkflowResources()



@mcp.tool()
async def analyze_codebase_with_gemini(

    focus_description: str,

    directories: str = None,

    file_patterns: str = None,

    exclude_patterns: str = None

) -> str:
    """Analyze codebase using Gemini's 2M token context window


    Args:

        focus_description: What to focus on in the analysis

        directories: Directories to analyze

        file_patterns: File patterns to include

        exclude_patterns: Patterns to exclude



    Returns:
        Result from analyze_codebase_with_gemini
    """
    result = await generator.analyze_codebase_with_gemini(

        focus_description=focus_description,

        directories=directories,

        file_patterns=file_patterns,

        exclude_patterns=exclude_patterns

    )
    return str(result)


@mcp.tool()
async def create_specification_with_gemini(

    feature_description: str,

    context_id: str = None,

    spec_template: str = None,

    output_path: str = None

) -> str:
    """Generate detailed technical specification using full codebase context

    This tool automatically loads and analyzes your codebase to generate
    context-aware specifications. You can call it directly without any
    prior setup.

    Args:

        feature_description: What feature to specify

        context_id: Optional context ID from previous analysis.
                    If not provided, automatically loads codebase.
                    Provide this to reuse context from previous calls (faster).

        spec_template: Specification template to use (standard/minimal)

        output_path: Where to save the spec



    Returns:
        JSON string containing spec_path, spec_content, implementation_tasks,
        estimated_complexity, files_to_modify, files_to_create, and context_id
        (use context_id for subsequent tool calls to skip reloading)
    """
    result = await generator.create_specification_with_gemini(

        feature_description=feature_description,

        context_id=context_id,

        spec_template=spec_template,

        output_path=output_path

    )
    return str(result)


@mcp.tool()
async def review_code_with_gemini(

    files: str = None,

    review_focus: str = None,

    spec_path: str = None,

    output_path: str = None,

    context_id: str = None

) -> str:
    """Comprehensive code review using Gemini

    This tool automatically loads and analyzes your codebase to provide
    context-aware code reviews. It reviews git changes by default, or
    specific files if provided.

    Args:

        files: Files to review (defaults to git diff if not provided)

        review_focus: Areas to focus on (e.g., security, performance)

        spec_path: Path to spec to review against

        output_path: Where to save review

        context_id: Optional context ID from previous analysis.
                    If not provided, automatically loads codebase.
                    Provide this to reuse context from previous calls (faster).



    Returns:
        JSON string containing review_path, review_content, issues_found,
        has_blocking_issues, summary, recommendations, and context_id
        (use context_id for subsequent tool calls to skip reloading)
    """
    result = await generator.review_code_with_gemini(

        files=files,

        review_focus=review_focus,

        spec_path=spec_path,

        output_path=output_path,

        context_id=context_id

    )
    return str(result)


@mcp.tool()
async def generate_documentation_with_gemini(

    documentation_type: str,

    scope: str,

    output_path: str = None,

    include_examples: bool = None,

    context_id: str = None

) -> str:
    """Generate comprehensive documentation with full codebase context

    This tool automatically loads and analyzes your codebase to generate
    context-aware documentation with examples from your actual code.

    Args:

        documentation_type: Type of documentation (api, architecture, user-guide, etc.)

        scope: What to document (e.g., "authentication system", "REST API")

        output_path: Where to save documentation

        include_examples: Include code examples from the codebase

        context_id: Optional context ID from previous analysis.
                    If not provided, automatically loads codebase.
                    Provide this to reuse context from previous calls (faster).



    Returns:
        JSON string containing doc_path, doc_content, sections, word_count,
        and context_id (use context_id for subsequent tool calls to skip reloading)
    """
    result = await generator.generate_documentation_with_gemini(

        documentation_type=documentation_type,

        scope=scope,

        output_path=output_path,

        include_examples=include_examples,

        context_id=context_id

    )
    return str(result)


@mcp.tool()
async def ask_gemini(

    prompt: str,

    include_codebase_context: bool = None,

    context_id: str = None,

    temperature: float = None

) -> str:
    """General-purpose Gemini query with optional codebase context


    Args:
        
        prompt: Question or task for Gemini
        
        include_codebase_context: Load full codebase context
        
        context_id: Reuse cached context
        
        temperature: Temperature for generation
        


    Returns:
        Result from ask_gemini
    """
    result = await generator.ask_gemini(

        prompt=prompt,

        include_codebase_context=include_codebase_context,

        context_id=context_id,

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