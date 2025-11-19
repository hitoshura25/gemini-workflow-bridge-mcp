"""Central registry for all MCP tools with metadata for progressive disclosure"""

from enum import Enum
from typing import Literal, TypedDict


class ToolCategory(Enum):
    """Categories for organizing tools"""
    FACT_EXTRACTION = "fact_extraction"  # Query, analyze, find
    VALIDATION = "validation"             # Validate, check consistency
    GENERATION = "generation"             # Generate workflows, commands
    WORKFLOW = "workflow"                 # Setup workflows
    GENERAL = "general"                   # Ask Gemini


class ToolMetadata(TypedDict):
    """Metadata for a tool to enable search and progressive disclosure"""
    name: str
    category: ToolCategory
    short_description: str  # One-line summary (for search results)
    keywords: list[str]     # Searchable keywords
    use_cases: list[str]    # Common use cases (for search)
    complexity: Literal["simple", "moderate", "complex"]
    requires_codebase: bool # Whether tool needs codebase context


# Tool Registry: Maps tool names to their metadata
TOOL_REGISTRY: dict[str, ToolMetadata] = {
    "query_codebase_tool": {
        "name": "query_codebase_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Answer multiple questions about codebase with compressed facts",
        "keywords": [
            "query", "questions", "facts", "codebase", "analyze",
            "find", "search", "multiple questions", "compression"
        ],
        "use_cases": [
            "Answer specific questions about codebase structure",
            "Gather facts for specification creation",
            "Understand existing patterns and conventions",
            "Find relevant files and components"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "find_code_by_intent_tool": {
        "name": "find_code_by_intent_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Semantic search for code by natural language intent",
        "keywords": [
            "find", "search", "semantic", "intent", "code",
            "locate", "discover", "natural language"
        ],
        "use_cases": [
            "Find code by describing what it does",
            "Locate APIs or endpoints by intent",
            "Discover relevant files for a feature",
            "Search by functionality rather than exact names"
        ],
        "complexity": "simple",
        "requires_codebase": True
    },

    "trace_feature_tool": {
        "name": "trace_feature_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Trace execution flow of a feature through the codebase",
        "keywords": [
            "trace", "flow", "feature", "execution", "path",
            "follow", "dependencies", "data flow"
        ],
        "use_cases": [
            "Understand how a feature is implemented",
            "Trace data flow through the system",
            "Find dependencies for refactoring",
            "Map execution paths"
        ],
        "complexity": "complex",
        "requires_codebase": True
    },

    "list_error_patterns_tool": {
        "name": "list_error_patterns_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Extract and categorize patterns (error handling, logging, etc.)",
        "keywords": [
            "patterns", "error handling", "logging", "async",
            "database", "conventions", "consistency"
        ],
        "use_cases": [
            "Find error handling patterns in codebase",
            "Discover logging conventions",
            "Identify async patterns",
            "Analyze database query patterns"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "validate_against_codebase_tool": {
        "name": "validate_against_codebase_tool",
        "category": ToolCategory.VALIDATION,
        "short_description": "Validate specification for completeness and accuracy",
        "keywords": [
            "validate", "spec", "specification", "completeness",
            "accuracy", "check", "verify", "review"
        ],
        "use_cases": [
            "Validate specification against codebase",
            "Check for missing dependencies",
            "Verify spec completeness",
            "Ensure spec aligns with existing patterns"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "check_consistency_tool": {
        "name": "check_consistency_tool",
        "category": ToolCategory.VALIDATION,
        "short_description": "Verify code/spec follows existing codebase patterns",
        "keywords": [
            "consistency", "patterns", "conventions", "naming",
            "style", "standards", "verify", "check"
        ],
        "use_cases": [
            "Check if new code follows conventions",
            "Verify naming consistency",
            "Validate against existing patterns",
            "Ensure architectural alignment"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "generate_feature_workflow_tool": {
        "name": "generate_feature_workflow_tool",
        "category": ToolCategory.GENERATION,
        "short_description": "Generate complete executable workflow for a feature",
        "keywords": [
            "generate", "workflow", "feature", "create",
            "automated", "template", "steps"
        ],
        "use_cases": [
            "Generate feature implementation workflow",
            "Create step-by-step guides",
            "Automate feature development process",
            "Generate refactoring workflows"
        ],
        "complexity": "simple",
        "requires_codebase": True
    },

    "generate_slash_command_tool": {
        "name": "generate_slash_command_tool",
        "category": ToolCategory.GENERATION,
        "short_description": "Auto-generate Claude Code slash commands",
        "keywords": [
            "generate", "slash command", "command", "create",
            "custom", "workflow", "automation"
        ],
        "use_cases": [
            "Create custom slash commands",
            "Automate common workflows",
            "Generate command templates",
            "Create developer shortcuts"
        ],
        "complexity": "simple",
        "requires_codebase": False
    },

    "setup_workflows_tool": {
        "name": "setup_workflows_tool",
        "category": ToolCategory.WORKFLOW,
        "short_description": "Set up recommended workflows and slash commands",
        "keywords": [
            "setup", "install", "initialize", "workflows",
            "commands", "configure", "bootstrap"
        ],
        "use_cases": [
            "Initial setup after MCP installation",
            "Install spec-only workflow",
            "Create recommended workflows",
            "Bootstrap command structure"
        ],
        "complexity": "simple",
        "requires_codebase": False
    },

    "analyze_codebase_with_gemini": {
        "name": "analyze_codebase_with_gemini",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "General-purpose codebase analysis with Gemini",
        "keywords": [
            "analyze", "analysis", "codebase", "architecture",
            "patterns", "general", "overview"
        ],
        "use_cases": [
            "Get overview of codebase architecture",
            "Understand overall patterns",
            "General-purpose analysis",
            "Architecture documentation"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "ask_gemini": {
        "name": "ask_gemini",
        "category": ToolCategory.GENERAL,
        "short_description": "General-purpose Gemini query (optional codebase context)",
        "keywords": [
            "ask", "query", "question", "gemini", "general",
            "anything", "flexible"
        ],
        "use_cases": [
            "Ask any question to Gemini",
            "Get answers without codebase context",
            "General knowledge queries",
            "Flexible AI assistance"
        ],
        "complexity": "simple",
        "requires_codebase": False
    }
}


def get_tool_metadata(tool_name: str) -> ToolMetadata | None:
    """Get metadata for a specific tool"""
    return TOOL_REGISTRY.get(tool_name)


def get_all_tool_names() -> list[str]:
    """Get list of all tool names"""
    return list(TOOL_REGISTRY.keys())


def get_tools_by_category(category: ToolCategory) -> list[ToolMetadata]:
    """Get all tools in a specific category"""
    return [
        metadata for metadata in TOOL_REGISTRY.values()
        if metadata["category"] == category
    ]


def search_tools_by_keyword(keyword: str) -> list[ToolMetadata]:
    """Search tools by keyword (case-insensitive, partial match)"""
    keyword_lower = keyword.lower()
    results = []

    for metadata in TOOL_REGISTRY.values():
        # Check keywords
        keyword_match = any(keyword_lower in kw.lower() for kw in metadata["keywords"])

        # Check description
        desc_match = keyword_lower in metadata["short_description"].lower()

        # Check use cases
        use_case_match = any(
            keyword_lower in uc.lower() for uc in metadata["use_cases"]
        )

        if keyword_match or desc_match or use_case_match:
            results.append(metadata)

    return results
