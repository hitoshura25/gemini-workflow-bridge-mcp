"""Predefined workflow templates for the Gemini MCP Server."""

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
1. Use `query_codebase_tool` to gather facts about relevant codebase areas
2. Create detailed specification document using the facts
3. Use `validate_against_codebase_tool` to check completeness
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
1. Create specification using `query_codebase_tool`
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
1. Analyze current implementation using `trace_feature_tool`
2. Create refactoring specification
3. Validate plan with `check_consistency_tool`
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
