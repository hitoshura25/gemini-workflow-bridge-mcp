# Specification: Workflow Setup Tool for Gemini MCP Server

## Overview
Create a new MCP tool called `setup_workflows_tool` that automatically generates the recommended workflow files and slash commands for the Gemini MCP Server. This allows developers to quickly set up the spec-only workflow and other common workflows without manually creating files.

## Problem Statement
Currently, developers who install the Gemini MCP Server need to manually create workflow files and slash commands to use features like the spec-only workflow. This creates friction in the onboarding experience and requires developers to understand the internal structure before they can benefit from the workflow capabilities.

## Goals
1. Provide a single command to set up all recommended workflows and slash commands
2. Make the spec-only workflow immediately available after installation
3. Support customization of which workflows to install
4. Follow existing patterns for file generation tools in the codebase
5. Respect user's existing files and provide options for handling conflicts

## Non-Goals
- Modifying existing workflow/command files without user consent
- Creating custom workflows (that's handled by `generate_feature_workflow_tool`)
- Managing workflow execution (only setup/installation)

## Technical Design

### New MCP Tool: `setup_workflows_tool`

#### Tool Name
`setup_workflows_tool`

#### Description
"Set up recommended workflow files and slash commands for the Gemini MCP Server, including the spec-only workflow"

#### Parameters

```python
{
    "workflows": {
        "type": "List[str]",
        "description": "List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']",
        "required": False,
        "default": ["spec-only"]
    },
    "output_dir": {
        "type": "Optional[str]",
        "description": "Base directory for outputs (workflows go in .claude/workflows/, commands in .claude/commands/)",
        "required": False,
        "default": None  # Uses current working directory
    },
    "overwrite": {
        "type": "bool",
        "description": "Whether to overwrite existing files",
        "required": False,
        "default": False
    },
    "include_commands": {
        "type": "bool",
        "description": "Whether to also create slash commands for the workflows",
        "required": False,
        "default": True
    }
}
```

#### Return Value

```json
{
    "success": true,
    "workflows_created": [
        {
            "name": "spec-only",
            "workflow_path": "./.claude/workflows/spec-only.md",
            "command_path": "./.claude/commands/spec-only.md",
            "status": "created"
        }
    ],
    "skipped": [],
    "message": "Successfully set up 1 workflow(s) and 1 command(s)"
}
```

### Implementation Plan

#### File Structure
```
hitoshura25_gemini_workflow_bridge/
├── tools/
│   ├── setup_workflows.py       # New file - main implementation
│   └── workflow_templates.py    # New file - workflow templates
├── server.py                     # Register new tool
```

#### Core Components

##### 1. Workflow Templates (`workflow_templates.py`)

Store predefined templates for common workflows:

```python
WORKFLOW_TEMPLATES = {
    "spec-only": {
        "workflow_content": """# Specification-Only Workflow

## Purpose
Create a detailed specification document without implementation.

## Steps

### 1. Gather Facts
Use `query_codebase_tool` to collect factual information:
- Existing patterns and conventions
- Related components and their interfaces
- Dependencies and integration points
- Current architecture decisions

### 2. Create Specification
Write a comprehensive specification including:
- Problem statement and goals
- Technical design and architecture
- API contracts and interfaces
- Data models and schemas
- Integration points
- Security considerations
- Testing strategy

### 3. Validate Specification
Use `validate_against_codebase_tool` with checks:
- missing_files
- undefined_dependencies
- pattern_violations
- completeness

### 4. Refine and Iterate
Address validation issues:
- Add missing details
- Resolve inconsistencies
- Clarify ambiguities
- Update based on codebase facts

### 5. Save Specification
Save to `specs/` directory with descriptive filename.
""",
        "command_content": """# /spec-only - Create specification document only (no implementation)

## Usage
/spec-only <feature_description>

## Description
Creates a detailed specification document for a feature without implementing it. Uses the Gemini MCP Server to gather facts about the codebase and validate the specification for completeness.

## Steps
1. Use query_codebase_tool to gather facts about relevant codebase areas
2. Create detailed specification document using the facts
3. Use validate_against_codebase_tool to check completeness
4. Address any validation issues
5. Save specification to specs/ directory

## Example
/spec-only Add user authentication with OAuth2 support
""",
        "description": "Spec-only workflow for creating specifications without implementation"
    },

    "feature": {
        "workflow_content": """# Feature Implementation Workflow

## Purpose
Implement a new feature with proper planning, validation, and testing.

## Steps

### 1. Create Specification
- Define feature requirements
- Document API contracts
- Plan architecture

### 2. Validate Against Codebase
- Check consistency with existing patterns
- Identify integration points
- Review dependencies

### 3. Implement Feature
- Write core functionality
- Follow codebase conventions
- Add error handling

### 4. Add Tests
- Write unit tests
- Add integration tests
- Verify edge cases

### 5. Review and Refine
- Run tests
- Check code quality
- Address issues
""",
        "command_content": """# /feature - Implement a new feature

## Usage
/feature <feature_description>

## Description
Implements a new feature following the Gemini workflow: spec creation, validation, implementation, and testing.

## Steps
1. Create specification using query_codebase_tool
2. Validate specification against codebase patterns
3. Implement the feature
4. Add comprehensive tests
5. Review and refine

## Example
/feature Add REST API endpoint for user profile updates
""",
        "description": "Full feature implementation workflow with spec, code, and tests"
    },

    "refactor": {
        "workflow_content": """# Refactoring Workflow

## Purpose
Refactor existing code while maintaining functionality and following codebase patterns.

## Steps

### 1. Analyze Current Implementation
- Use `trace_feature_tool` to understand execution flow
- Identify pain points and anti-patterns
- Document current behavior

### 2. Plan Refactoring
- Define refactoring goals
- Create specification for new structure
- Identify risks and mitigation strategies

### 3. Validate Plan
- Use `check_consistency_tool` to ensure alignment
- Verify backward compatibility requirements
- Review with existing patterns

### 4. Execute Refactoring
- Make incremental changes
- Maintain test coverage
- Preserve existing functionality

### 5. Verify and Test
- Run existing tests
- Add new tests if needed
- Verify no regressions
""",
        "command_content": """# /refactor - Refactor existing code

## Usage
/refactor <refactoring_description>

## Description
Refactors existing code following best practices: analyze, plan, validate, execute, and verify.

## Steps
1. Analyze current implementation using trace_feature_tool
2. Create refactoring specification
3. Validate plan with check_consistency_tool
4. Execute refactoring incrementally
5. Verify with tests and validation

## Example
/refactor Extract authentication logic into separate service
""",
        "description": "Refactoring workflow with analysis, planning, and validation"
    },

    "review": {
        "workflow_content": """# Code Review Workflow

## Purpose
Conduct thorough code review using Gemini analysis tools.

## Steps

### 1. Gather Context
- Use `query_codebase_tool` to understand affected areas
- Review related components
- Check recent changes

### 2. Analyze Patterns
- Use `list_error_patterns_tool` for consistency
- Check naming conventions
- Verify error handling patterns

### 3. Validate Consistency
- Use `check_consistency_tool` for alignment
- Verify architectural patterns
- Check testing standards

### 4. Trace Feature Flow
- Use `trace_feature_tool` for new features
- Verify data flow
- Check integration points

### 5. Provide Feedback
- Summarize findings
- Suggest improvements
- Highlight risks
""",
        "command_content": """# /review - Review code changes

## Usage
/review <code_location_or_description>

## Description
Conducts comprehensive code review using Gemini MCP tools to analyze patterns, consistency, and architecture.

## Steps
1. Gather context about code changes
2. Analyze patterns and conventions
3. Validate consistency with codebase
4. Trace feature execution flow
5. Provide structured feedback

## Example
/review Recent changes to authentication module
""",
        "description": "Code review workflow using Gemini analysis tools"
    }
}
```

##### 2. Setup Implementation (`setup_workflows.py`)

```python
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from .workflow_templates import WORKFLOW_TEMPLATES

def setup_workflows(
    workflows: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    include_commands: bool = True
) -> str:
    """
    Set up recommended workflow files and slash commands.

    Args:
        workflows: List of workflows to set up ['spec-only', 'feature', 'refactor', 'review', 'all']
        output_dir: Base directory for outputs (default: current directory)
        overwrite: Whether to overwrite existing files
        include_commands: Whether to create slash commands

    Returns:
        JSON string with setup results
    """

    if workflows is None:
        workflows = ["spec-only"]

    if "all" in workflows:
        workflows = ["spec-only", "feature", "refactor", "review"]

    base_dir = Path(output_dir) if output_dir else Path.cwd()
    workflow_dir = base_dir / ".claude" / "workflows"
    command_dir = base_dir / ".claude" / "commands"

    # Create directories
    workflow_dir.mkdir(parents=True, exist_ok=True)
    if include_commands:
        command_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "success": True,
        "workflows_created": [],
        "skipped": [],
        "message": ""
    }

    for workflow_name in workflows:
        if workflow_name not in WORKFLOW_TEMPLATES:
            results["skipped"].append({
                "name": workflow_name,
                "reason": f"Unknown workflow type: {workflow_name}"
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

        # Create workflow file
        if workflow_path.exists() and not overwrite:
            workflow_result["status"] = "skipped (already exists)"
            results["skipped"].append(workflow_result)
            continue

        workflow_path.write_text(template["workflow_content"])

        # Create command file if requested
        if include_commands and command_path:
            if command_path.exists() and not overwrite:
                workflow_result["status"] = "workflow created, command skipped (already exists)"
            else:
                command_path.write_text(template["command_content"])

        results["workflows_created"].append(workflow_result)

    # Generate summary message
    created_count = len(results["workflows_created"])
    skipped_count = len(results["skipped"])

    if created_count > 0:
        results["message"] = f"Successfully set up {created_count} workflow(s)"
        if include_commands:
            results["message"] += f" and {created_count} command(s)"

    if skipped_count > 0:
        results["message"] += f", skipped {skipped_count} item(s)"

    return json.dumps(results, indent=2)
```

##### 3. Server Registration (`server.py`)

Add tool registration similar to existing tools:

```python
# In server.py, add to list_tools():
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... existing tools ...
        Tool(
            name="setup_workflows_tool",
            description="Set up recommended workflow files and slash commands for the Gemini MCP Server, including the spec-only workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflows": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']. Default: ['spec-only']",
                        "default": ["spec-only"]
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Base directory for outputs (workflows go in .claude/workflows/, commands in .claude/commands/). Default: current directory"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing files. Default: false",
                        "default": False
                    },
                    "include_commands": {
                        "type": "boolean",
                        "description": "Whether to also create slash commands for the workflows. Default: true",
                        "default": True
                    }
                }
            }
        )
    ]

# In call_tool(), add handler:
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    # ... existing handlers ...
    elif name == "setup_workflows_tool":
        from .tools.setup_workflows import setup_workflows
        result = setup_workflows(
            workflows=arguments.get("workflows"),
            output_dir=arguments.get("output_dir"),
            overwrite=arguments.get("overwrite", False),
            include_commands=arguments.get("include_commands", True)
        )
        return [TextContent(type="text", text=result)]
```

## Usage Examples

### Example 1: Quick Setup (Spec-Only Workflow)
```
User calls: setup_workflows_tool
Result: Creates .claude/workflows/spec-only.md and .claude/commands/spec-only.md
```

### Example 2: Setup All Workflows
```
User calls: setup_workflows_tool with workflows=["all"]
Result: Creates all workflow and command files
```

### Example 3: Setup Without Commands
```
User calls: setup_workflows_tool with workflows=["spec-only"], include_commands=false
Result: Creates only .claude/workflows/spec-only.md
```

### Example 4: Custom Location
```
User calls: setup_workflows_tool with workflows=["spec-only"], output_dir="/path/to/project"
Result: Creates files in /path/to/project/.claude/
```

## Integration Points

### Existing Tools Used
- None directly, but templates reference: `query_codebase_tool`, `validate_against_codebase_tool`, `trace_feature_tool`, `check_consistency_tool`, `list_error_patterns_tool`

### Directory Structure
- Workflows: `.claude/workflows/` (respects `DEFAULT_WORKFLOW_DIR` environment variable)
- Commands: `.claude/commands/` (respects `DEFAULT_COMMAND_DIR` environment variable)

### Configuration
- Uses existing environment variable patterns from `resources.py`
- Respects user's directory preferences
- No new environment variables needed

## Security Considerations

1. **Path Validation**: Validate `output_dir` parameter to prevent directory traversal attacks
2. **File Overwrite Protection**: Default `overwrite=False` prevents accidental data loss
3. **Permission Checks**: Verify write permissions before attempting file creation
4. **Input Sanitization**: Validate workflow names against allowed list

## Testing Strategy

### Unit Tests
1. Test workflow template content is valid markdown
2. Test file creation with various parameter combinations
3. Test overwrite protection works correctly
4. Test error handling for invalid paths
5. Test skipping existing files when overwrite=False

### Integration Tests
1. Test tool registration in MCP server
2. Test tool execution through MCP protocol
3. Test directory creation with different base paths
4. Test interaction with environment variables

### Manual Testing Checklist
1. Install MCP server in fresh project
2. Call `setup_workflows_tool` with default parameters
3. Verify files created in correct locations
4. Verify slash command works in Claude Code
5. Test overwrite behavior
6. Test with custom output directory

## Success Metrics

1. **Time to First Workflow**: Reduce from manual setup (5-10 minutes) to one command (<30 seconds)
2. **User Adoption**: Track usage of `setup_workflows_tool` vs manual workflow creation
3. **Error Rate**: Less than 1% of setups should fail due to tool issues
4. **Workflow Usage**: Increase in spec-only workflow usage after tool introduction

## Future Enhancements

1. **Custom Templates**: Allow users to provide custom workflow templates
2. **Interactive Setup**: Prompt user for which workflows to install
3. **Upgrade Command**: Update existing workflows to latest templates
4. **Template Validation**: Validate template content against expected structure
5. **Workflow Discovery**: Auto-detect which workflows would be useful based on project type

## Documentation Updates

### README.md
Add quick start section:
```markdown
## Quick Start

After installing the MCP server, set up recommended workflows:

1. In Claude Code, call the setup tool:
   - "Set up the spec-only workflow"
   - Claude will use `setup_workflows_tool` to create the files

2. Use the workflow:
   - Type `/spec-only <feature_description>`
   - Claude will guide you through creating a specification

For all workflows: Call `setup_workflows_tool` with `workflows=["all"]`
```

### New Documentation File
Create `docs/workflows.md` documenting:
- Available workflow types
- How to use each workflow
- Customization options
- Examples

## Dependencies

### New Dependencies
- None (uses Python standard library)

### Modified Files
- `hitoshura25_gemini_workflow_bridge/server.py` - Add tool registration
- New: `hitoshura25_gemini_workflow_bridge/tools/setup_workflows.py`
- New: `hitoshura25_gemini_workflow_bridge/tools/workflow_templates.py`

### Backward Compatibility
- Fully backward compatible
- Does not modify existing functionality
- Existing workflow/command generation tools unchanged

## Open Questions

1. Should we include a "tutorial" workflow that walks users through MCP server features?
2. Should we auto-run `setup_workflows_tool` on first MCP server initialization?
3. Should we version workflow templates and provide migration tools?
4. Should workflows be customizable via configuration file?

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-15 | Default to spec-only workflow only | Most users want to start with spec creation; avoids overwhelming new users |
| 2025-11-15 | Separate workflow content from command content in templates | Allows flexibility in how workflows are executed |
| 2025-11-15 | Use overwrite=False as default | Prevents accidental loss of user customizations |
| 2025-11-15 | Store templates in separate file | Improves maintainability and allows future template discovery |

## References

- Existing workflow tools: `hitoshura25_gemini_workflow_bridge/tools/generate_workflow.py:79`
- Existing command tools: `hitoshura25_gemini_workflow_bridge/tools/generate_command.py:90`
- Directory configuration: `hitoshura25_gemini_workflow_bridge/resources.py:10-12`
- Current spec-only command: `.claude/commands/spec-only.md:6-10`
