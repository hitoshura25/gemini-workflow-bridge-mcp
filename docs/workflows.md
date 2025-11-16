# Workflows Guide

Complete guide to using workflows with the Gemini MCP Server.

## Table of Contents

- [Quick Start](#quick-start)
- [Setup Tool](#setup-tool)
- [Available Workflows](#available-workflows)
- [Using Slash Commands](#using-slash-commands)
- [Customization](#customization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Setting Up Workflows

The easiest way to get started is to use the `setup_workflows_tool`:

```
In Claude Code, just say:
"Set up the spec-only workflow"
```

This creates:
- `.claude/workflows/spec-only.md` - The workflow definition
- `.claude/commands/spec-only.md` - The slash command

Now you can use: `/spec-only <feature_description>`

### Setting Up All Workflows

```
"Set up all workflows for me"
```

This creates all 4 workflows:
- **spec-only** - Create specifications without implementation
- **feature** - Full feature implementation with tests
- **refactor** - Refactor code with analysis and validation
- **review** - Comprehensive code review

## Setup Tool

### `setup_workflows_tool`

Automatically generates workflow files and slash commands.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workflows` | List[str] | `["spec-only"]` | Workflows to set up: `spec-only`, `feature`, `refactor`, `review`, or `all` |
| `output_dir` | str | Current directory | Base directory for outputs |
| `overwrite` | bool | `false` | Whether to overwrite existing files |
| `include_commands` | bool | `true` | Whether to create slash commands |

#### Usage Examples

**Default (spec-only):**
```python
setup_workflows_tool()
```

**All workflows:**
```python
setup_workflows_tool(workflows=["all"])
```

**Specific workflows:**
```python
setup_workflows_tool(workflows=["spec-only", "feature"])
```

**Custom directory:**
```python
setup_workflows_tool(
    workflows=["all"],
    output_dir="/path/to/project"
)
```

**Workflows only (no commands):**
```python
setup_workflows_tool(
    workflows=["spec-only"],
    include_commands=False
)
```

#### Environment Variables

Customize output directories:

```bash
# In your MCP server config or environment
DEFAULT_WORKFLOW_DIR="custom/workflows"
DEFAULT_COMMAND_DIR="custom/commands"
```

#### Return Value

```json
{
  "success": true,
  "workflows_created": [
    {
      "name": "spec-only",
      "workflow_path": ".claude/workflows/spec-only.md",
      "command_path": ".claude/commands/spec-only.md",
      "status": "created"
    }
  ],
  "skipped": [],
  "message": "Successfully set up 1 workflow(s) and 1 command(s)"
}
```

## Available Workflows

### 1. Spec-Only Workflow

**Purpose:** Create detailed specifications without implementation

**When to use:**
- Planning a new feature
- Documenting requirements before coding
- Getting team alignment on approach
- Creating PRDs (Product Requirements Documents)

**Process:**
1. Gather Facts - Use `query_codebase_tool` to collect context
2. Create Specification - Write comprehensive spec including:
   - Problem statement and goals
   - Technical design and architecture
   - API contracts and interfaces
   - Data models and schemas
3. Validate Specification - Use `validate_against_codebase_tool`
4. Refine and Iterate - Address validation issues
5. Save Specification - Store in `specs/` directory

**Slash command:**
```
/spec-only <feature_description>
```

**Example:**
```
/spec-only Add user authentication with OAuth2 support
```

---

### 2. Feature Implementation Workflow

**Purpose:** Implement a complete feature with planning, validation, and testing

**When to use:**
- Building new features
- Adding functionality to existing systems
- Implementing user stories

**Process:**
1. Create Specification - Define requirements and architecture
2. Validate Against Codebase - Check consistency with existing patterns
3. Implement Feature - Write core functionality
4. Add Tests - Write unit and integration tests
5. Review and Refine - Ensure quality

**Slash command:**
```
/feature <feature_description>
```

**Example:**
```
/feature Add REST API endpoint for user profile updates
```

---

### 3. Refactor Workflow

**Purpose:** Refactor code while maintaining functionality

**When to use:**
- Improving code quality
- Extracting reusable components
- Standardizing patterns across codebase
- Addressing technical debt

**Process:**
1. Analyze Current Implementation - Use `trace_feature_tool`
2. Plan Refactoring - Define goals and approach
3. Validate Plan - Use `check_consistency_tool`
4. Execute Refactoring - Make incremental changes
5. Verify and Test - Ensure no regressions

**Slash command:**
```
/refactor <refactoring_description>
```

**Example:**
```
/refactor Extract authentication logic into separate service
```

---

### 4. Review Workflow

**Purpose:** Conduct thorough code review using Gemini analysis

**When to use:**
- Reviewing pull requests
- Auditing code quality
- Checking for pattern violations
- Security reviews

**Process:**
1. Gather Context - Use `query_codebase_tool`
2. Analyze Patterns - Use `list_error_patterns_tool`
3. Validate Consistency - Use `check_consistency_tool`
4. Trace Feature Flow - Use `trace_feature_tool`
5. Provide Feedback - Summarize findings

**Slash command:**
```
/review <code_location_or_description>
```

**Example:**
```
/review Recent changes to authentication module
```

## Using Slash Commands

### What are Slash Commands?

Slash commands are shortcuts in Claude Code that trigger predefined workflows. They make it easy to execute complex multi-step processes with a single command.

### How to Use

1. **Set up the workflows** (one-time):
   ```
   "Set up all workflows"
   ```

2. **Use the slash command**:
   ```
   /spec-only Add Redis caching to product API
   /feature Implement rate limiting for API endpoints
   /refactor Move validation logic to middleware
   /review src/services/auth.ts
   ```

3. **Claude follows the workflow** automatically:
   - Gathers context using Gemini tools
   - Creates specifications
   - Validates against codebase
   - Implements changes
   - Runs tests

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/spec-only` | Create specification only | `/spec-only Add OAuth2 support` |
| `/feature` | Implement complete feature | `/feature Add user notifications` |
| `/refactor` | Refactor existing code | `/refactor Extract payment logic` |
| `/review` | Review code changes | `/review Recent auth changes` |

## Customization

### Modifying Workflow Files

Workflow files are markdown files in `.claude/workflows/`. You can edit them to:

- Add custom steps
- Change the order of operations
- Add project-specific checks
- Customize validation criteria

**Example customization:**

Edit `.claude/workflows/spec-only.md`:
```markdown
### 1. Gather Facts
Use `query_codebase_tool` to collect factual information:
- Existing patterns and conventions
- Related components and their interfaces
- **YOUR CUSTOM STEP: Check company coding standards**
```

### Modifying Slash Commands

Command files are in `.claude/commands/`. Edit them to:

- Change command behavior
- Add additional context
- Customize prompts

**Example customization:**

Edit `.claude/commands/spec-only.md`:
```markdown
## Steps
1. Use `query_codebase_tool` to gather facts
2. **Check internal wiki for similar features**
3. Create detailed specification document
```

### Creating Custom Workflows

Use `generate_feature_workflow_tool` for custom workflows:

```python
generate_feature_workflow_tool(
    feature_description="Add GraphQL API support",
    workflow_style="interactive"
)
```

Or use `generate_slash_command_tool` for custom commands:

```python
generate_slash_command_tool(
    command_name="deploy",
    workflow_type="custom",
    description="Deploy application to production",
    steps=[
        "Run all tests",
        "Build production bundle",
        "Deploy to staging",
        "Run smoke tests",
        "Deploy to production"
    ]
)
```

## Examples

### Example 1: Planning a New Feature

```
User: /spec-only Add Redis caching to product API

Claude:
1. Using query_codebase_tool to gather facts...
   - Found existing product API at src/api/products.ts
   - Current response time: 200-500ms
   - Database queries: 3 per request

2. Creating specification...
   [Detailed spec with caching strategy]

3. Validating specification...
   - Completeness: 95%
   - Missing: Redis connection config

4. Refining specification...
   [Added Redis config section]

5. Saved to specs/redis-product-caching.md
```

### Example 2: Implementing a Feature

```
User: /feature Add user notifications system

Claude:
1. Creating specification using query_codebase_tool...
2. Validating against existing patterns...
3. Implementing notification service...
4. Adding tests...
5. Review complete - all tests passing
```

### Example 3: Refactoring Code

```
User: /refactor Extract authentication logic into separate service

Claude:
1. Analyzing current implementation with trace_feature_tool...
   - Auth code spread across 5 files
   - 3 different patterns identified

2. Planning refactoring...
   - New AuthService class
   - Centralized token management

3. Validating plan with check_consistency_tool...
4. Executing refactoring...
5. Verified - all tests passing, no regressions
```

### Example 4: Code Review

```
User: /review Recent changes to authentication module

Claude:
1. Gathering context...
2. Analyzing patterns...
   - Found inconsistent error handling
   - Missing input validation in 2 endpoints
3. Checking consistency...
4. Providing feedback:

   **Security Issues:**
   - Missing rate limiting on login endpoint

   **Pattern Violations:**
   - Inconsistent error response format

   **Recommendations:**
   - Add rate limiting middleware
   - Standardize error responses
```

## Troubleshooting

### Workflows Not Working

**Issue:** Slash command not recognized

**Solution:**
1. Verify files were created:
   ```bash
   ls .claude/workflows/
   ls .claude/commands/
   ```

2. Restart Claude Code

3. Re-run setup:
   ```
   "Set up all workflows with overwrite enabled"
   ```

---

**Issue:** Command created but workflow not followed

**Solution:**
1. Check workflow file syntax (must be valid markdown)
2. Verify tool names are correct in workflow (use backticks: `` `query_codebase_tool` ``)
3. Review Claude Code logs for errors

---

### File Location Issues

**Issue:** Files created in wrong directory

**Solution:**
1. Check current directory when running setup
2. Specify `output_dir` explicitly:
   ```python
   setup_workflows_tool(output_dir="/path/to/project")
   ```
3. Check environment variables:
   ```bash
   echo $DEFAULT_WORKFLOW_DIR
   echo $DEFAULT_COMMAND_DIR
   ```

---

### Overwrite Protection

**Issue:** "Already exists" message when trying to update

**Solution:**
1. Use `overwrite=True`:
   ```python
   setup_workflows_tool(workflows=["all"], overwrite=True)
   ```

2. Or manually delete files:
   ```bash
   rm .claude/workflows/spec-only.md
   rm .claude/commands/spec-only.md
   ```

---

### Permission Errors

**Issue:** "Permission denied" when creating files

**Solution:**
1. Check directory permissions:
   ```bash
   ls -la .claude/
   ```

2. Create directories manually:
   ```bash
   mkdir -p .claude/workflows
   mkdir -p .claude/commands
   chmod 755 .claude/workflows
   chmod 755 .claude/commands
   ```

## Advanced Usage

### Workflow Composition

Combine multiple workflows for complex tasks:

```
1. /spec-only Add payment processing
2. /review specs/payment-processing.md
3. /feature Implement payment processing (based on spec)
4. /review Recent payment implementation
```

### Integration with CI/CD

Use workflows in automation:

```bash
# In CI pipeline
claude-code "/review src/services/payment.ts" > review-report.md
```

### Team Workflows

Share custom workflows:

```bash
# Commit workflow files to version control
git add .claude/workflows/
git add .claude/commands/
git commit -m "Add team workflows"
```

## Best Practices

1. **Start with spec-only** - Always plan before implementing
2. **Use validation tools** - Leverage `validate_against_codebase_tool`
3. **Iterate on specs** - Refine specifications based on validation
4. **Keep workflows updated** - Customize for your project's needs
5. **Review regularly** - Use `/review` to maintain code quality
6. **Document decisions** - Save specs in version control

## Getting Help

- [Main README](../README.md) - Project overview
- [MCP Documentation](https://modelcontextprotocol.io) - MCP protocol details
- [Issues](https://github.com/hitoshura25/gemini-workflow-bridge-mcp/issues) - Report bugs or request features

---

**Last Updated:** November 16, 2025
