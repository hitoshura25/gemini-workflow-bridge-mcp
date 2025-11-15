# /create-spec - Complete Feature Implementation Workflow

## Description
Create detailed specification using Gemini-powered analysis

This command:
1. Analyzes codebase context using Gemini
2. Generates detailed specification using Claude
3. Validates completeness
4. Implements with tests
5. Reviews for consistency

## Usage
/create-spec <feature description>

## Examples
/create-spec Add JWT refresh token support
/create-spec Implement rate limiting for API
/create-spec Add email verification for new users

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
