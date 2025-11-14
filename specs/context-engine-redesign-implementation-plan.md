# Gemini Workflow Bridge MCP v2.0 - Complete Redesign Implementation Plan

**Date Created:** 2025-01-13
**Status:** Planning
**Version:** 2.0.0 (Breaking Changes)

---

## Executive Summary

This document provides a complete implementation plan for redesigning the Gemini Workflow Bridge MCP from a "spec generation tool" to a "context compression engine" that optimally leverages both Claude Code and Gemini's strengths.

**Problem Identified:**
- Current design has Gemini generating specifications → B-grade quality
- Claude Code has to review and fix Gemini's specs → inefficient workflow
- Token usage not optimized (Gemini tries to do analysis + planning simultaneously)

**Solution:**
- **Gemini = Context Engine:** Analyze massive codebases, return compressed facts
- **Claude = Reasoning Engine:** Use facts to create A-grade specifications
- **MCP = Compression Layer:** Reduce 50,000 tokens → 300 token summaries (174:1 ratio)
- **Workflow Automation:** Auto-generate slash commands and multi-step workflows

**Key Metrics:**
- ✅ Quality improvement: B-grade → A-grade specifications
- ✅ Cost reduction: 47% fewer Claude tokens (expensive)
- ✅ Token compression: 174:1 ratio (50K → 300 tokens)
- ✅ Developer experience: Auto-generated workflows and slash commands

---

## Table of Contents

1. [Research & Context](#research--context)
2. [Architecture Overview](#architecture-overview)
3. [Token & Cost Optimization](#token--cost-optimization)
4. [Tool Specifications](#tool-specifications)
5. [Implementation Phases](#implementation-phases)
6. [Migration Guide](#migration-guide)
7. [Testing Strategy](#testing-strategy)
8. [Success Metrics](#success-metrics)

---

## Research & Context

### Key Insights from Gemini's Analysis

Source: `specs/gemini-notes.md`

**Main Finding:**
> "Your spec-creation tool is trying to do two distinct cognitive tasks at once: analysis (what's there) and synthesis/planning (what should be there). Gemini's strength is the analysis on a massive context. Claude's strength is the synthesis and reasoning to create a detailed plan."

**Recommendations:**
1. **Deconstruct spec-creation tool** - Too "smart," trying to do Claude's job
2. **Sharpen analyze-codebase tool** - Make it a "dumb" fact-provider
3. **Let Claude do the thinking** - Two-step process: Gemini facts → Claude planning
4. **Add spec reviewer tool** - Validate completeness after Claude creates spec

**Token Optimization Example:**

```
Old Way (Problematic):
- Input: 50,000 tokens (code) + 100 tokens (prompt) = 50,100 tokens
- Gemini: Tries to analyze AND write spec
- Quality: Low (misses details)

New Way (Optimized):
- Gemini: 50,000 tokens (code analysis) → 300 tokens (facts summary)
- Claude: 300 tokens (facts) + 100 tokens (prompt) = 400 tokens
- Claude: Creates spec using superior reasoning
- Quality: High
- Cost: Only 400 "expensive" Claude tokens vs 50,100
```

**Compression Ratio:** 174:1 (50,000 → 300)

### Key Insights from Anthropic's MCP Article

Source: https://www.anthropic.com/engineering/code-execution-with-mcp

**Progressive Disclosure:**
- Don't expose all tool definitions upfront
- Present tools as code on a filesystem for on-demand loading
- Demonstrated 98.7% token reduction (150,000 → 2,000 tokens)

**Filtering at the Edge:**
- Process and filter data within execution environment BEFORE returning to model
- When fetching large datasets, filter/transform results in code
- Don't pass raw data through context window

**Code-Based APIs Over Direct Tool Calls:**
- Expose MCP servers as callable code interfaces
- Enables agents to write logic combining multiple tools efficiently

**State Persistence:**
- Support filesystem access for progress tracking
- Cache results between sessions
- Develop reusable skills that improve over time

**Privacy-Preserving Data Flow:**
- Implement tokenization for sensitive data
- Data flows between systems without entering model context

---

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: Claude Code                          │
│                (Orchestrator + Reasoning Engine)                 │
│                                                                  │
│  Strengths:                                                     │
│  • Superior reasoning and planning                              │
│  • Precise code editing and implementation                      │
│  • Multi-step workflow orchestration                            │
│  • A-grade specification creation                               │
│                                                                  │
│  Token Usage: 400-5,000 tokens (expensive, but high-value)     │
└────────────────────────┬────────────────────────────────────────┘
                         │ MCP Protocol
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 2: MCP Server (Compression Layer)             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 1: Fact Extraction Tools                           │   │
│  │ • query_codebase() - Multi-question factual answers     │   │
│  │ • find_code_by_intent() - Semantic search with summary  │   │
│  │ • trace_feature() - Follow code flow across files       │   │
│  │ • list_error_patterns() - Extract patterns (edge filter)│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 2: Validation Tools                                │   │
│  │ • validate_against_codebase() - Check spec completeness │   │
│  │ • check_consistency() - Verify pattern alignment        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Tier 3: Workflow Automation (Progressive Disclosure)    │   │
│  │ • generate_feature_workflow() - Multi-step automation   │   │
│  │ • generate_slash_command() - Create custom .claude cmds │   │
│  │ • save_workflow_state() - Persist progress             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Role: Context compression (50K → 300 tokens)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ Gemini CLI
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 3: Gemini (Context Engine)                   │
│                   (2M token window, free tier)                  │
│                                                                  │
│  Strengths:                                                     │
│  • Massive context window (2M tokens)                           │
│  • Free/cheap for bulk analysis                                 │
│  • Fast pattern matching across entire codebase                 │
│  • Excellent at factual extraction                              │
│                                                                  │
│  Weaknesses:                                                    │
│  • Lower quality reasoning/planning than Claude                 │
│  • Should NOT generate final specs/docs                         │
│                                                                  │
│  Token Usage: 50,000+ tokens (cheap, bulk analysis)            │
└─────────────────────────────────────────────────────────────────┘
```

### Division of Labor

| Task | Old Design (v1.x) | New Design (v2.0) | Quality | Cost |
|------|-------------------|-------------------|---------|------|
| **Codebase Analysis** | Gemini (50K tokens) | Gemini (50K tokens) | Same | Same (free) |
| **Spec Generation** | Gemini (tries to plan) | Claude (uses facts) | B→A grade | 47% less |
| **Code Review** | Gemini (opinions) | Gemini (facts) + Claude (review) | B→A grade | Optimized |
| **Documentation** | Gemini (generates) | Claude (generates) + Gemini (validates) | B→A grade | Optimized |

---

## Token & Cost Optimization

### The Compression Strategy

**Core Principle:** Use Gemini's cheap/free large context window to compress massive codebases into small, high-signal summaries for Claude.

### Example: Feature Implementation Workflow

**Old Workflow (v1.x):**
```
User: "Add 2FA authentication"

Gemini Tool: create_specification_with_gemini()
├─ Loads codebase: 50,000 tokens
├─ Analyzes architecture: 50,000 tokens
├─ Generates spec: 50,000 tokens (tries to plan)
└─ Returns: B-grade spec (misses details)

Claude Code: Reviews spec
├─ Reads Gemini's spec: 2,000 tokens
├─ Finds gaps/errors: 1,000 tokens
├─ Rewrites sections: 5,000 tokens
└─ Total Claude tokens: 8,000 (expensive)

Total Cost:
- Gemini: 50,000 tokens (free tier)
- Claude: 8,000 tokens (paid)
- Quality: B-grade
```

**New Workflow (v2.0):**
```
User: "Add 2FA authentication"

Step 1: Gemini (Fact Extraction)
Tool: query_codebase({
  questions: [
    "How is authentication currently implemented?",
    "What is the user model structure?",
    "What middleware patterns are used?",
    "What testing frameworks are in place?"
  ],
  scope: "src/"
})
├─ Loads codebase: 50,000 tokens
├─ Extracts facts: 50,000 tokens
├─ Compresses to summary: 300 tokens
└─ Returns: Structured facts with file:line references

Step 2: Claude (Planning & Specification)
├─ Receives facts: 300 tokens
├─ User request: 100 tokens
├─ Creates A-grade spec: 2,000 tokens
└─ Total: 2,400 tokens

Step 3: Gemini (Validation)
Tool: validate_against_codebase({spec: "..."})
├─ Loads spec: 2,000 tokens
├─ Checks against codebase: 50,000 tokens
├─ Returns gaps/issues: 200 tokens
└─ Returns: Completeness report

Step 4: Claude (Refinement)
├─ Reads validation: 200 tokens
├─ Updates spec: 500 tokens
└─ Total: 700 tokens

Total Cost:
- Gemini: 102,000 tokens (free tier)
- Claude: 3,100 tokens (paid)
- Quality: A-grade
- Savings: 61% fewer Claude tokens
```

### Token Breakdown Comparison

| Metric | Old Design | New Design | Improvement |
|--------|-----------|------------|-------------|
| Gemini tokens | 50,000 | 102,000 | 2x more (but free) |
| Claude tokens | 8,000 | 3,100 | 61% reduction |
| Total tokens | 58,000 | 105,100 | More total, but... |
| Expensive tokens (Claude) | 8,000 | 3,100 | **61% cost savings** |
| Spec quality | B-grade | A-grade | **Much better** |
| Compression ratio | N/A | 174:1 | 50K→300 tokens |

**Key Insight:** We're using MORE total tokens, but shifting expensive Claude tokens to free Gemini tokens while IMPROVING quality.

---

## Tool Specifications

### Tier 1: Fact Extraction Tools

#### Tool 1: `query_codebase`

**Purpose:** Multi-question factual analysis with massive context compression

**Gemini System Prompt (Critical!):**
```
You are a code analysis engine. Your ONLY job is to answer questions about code with FACTS.

RULES:
1. Answer ONLY with factual information found in the code
2. List file paths, line numbers, function names, class definitions
3. Trace data flow as requested
4. Extract patterns and usage examples

FORBIDDEN:
❌ DO NOT create software specifications
❌ DO NOT suggest new features or improvements
❌ DO NOT make creative inferences beyond what's in the code
❌ DO NOT give opinions

OUTPUT FORMAT:
- Structured JSON with file:line references
- Keep responses concise (target: 300 tokens per answer)
- Include code snippets only when specifically requested

Remember: You are a fact-provider, not a planner.
```

**Input Schema:**
```typescript
{
  questions: string[];           // 1-10 specific questions
  scope?: string;                // Directory to analyze (default: ".")
  include_patterns?: string[];   // File patterns (default: common code files)
  exclude_patterns?: string[];   // Exclude patterns (default: node_modules, etc.)
  max_tokens_per_answer?: number; // Target token budget per answer (default: 300)
}
```

**Output Schema:**
```typescript
{
  answers: Array<{
    question: string;
    facts: string[];              // Factual statements with file:line refs
    code_snippets?: Array<{
      file: string;
      lines: string;             // e.g., "42-56"
      snippet: string;
    }>;
    file_count: number;
    instance_count: number;
  }>;
  metadata: {
    tokens_analyzed: number;      // Total code tokens sent to Gemini
    tokens_returned: number;      // Compressed output tokens
    compression_ratio: number;    // Ratio of input:output
    analysis_time_seconds: number;
  };
}
```

**Example Usage:**

*Input:*
```json
{
  "questions": [
    "What files contain error handling with 'throw new Error'?",
    "What is the user authentication flow from login endpoint?",
    "What testing patterns are used across the codebase?"
  ],
  "scope": "src/",
  "max_tokens_per_answer": 400
}
```

*Output:*
```json
{
  "answers": [
    {
      "question": "What files contain error handling with 'throw new Error'?",
      "facts": [
        "src/services/userService.js:42 - throw new Error('User not found')",
        "src/services/userService.js:91 - throw new Error('Invalid email address')",
        "src/services/projectService.js:25 - throw new Error('Project not found')",
        "src/services/projectService.js:150 - throw new Error('You do not have permission')"
      ],
      "file_count": 2,
      "instance_count": 4
    },
    {
      "question": "What is the user authentication flow from login endpoint?",
      "facts": [
        "Flow starts: routes/auth.js:15 - POST /api/auth/login → authController.login()",
        "Step 2: controllers/authController.js:28 → authService.validateCredentials()",
        "Step 3: services/authService.js:45 → User.findByEmail(email)",
        "Step 4: services/authService.js:52 → bcrypt.compare(password, user.password)",
        "Step 5: services/authService.js:58 → jwtUtils.generateToken({userId, role})",
        "Final: Returns JWT token with 24h expiration"
      ],
      "code_snippets": [
        {
          "file": "routes/auth.js",
          "lines": "15-18",
          "snippet": "router.post('/login', authController.login);"
        }
      ],
      "file_count": 4,
      "instance_count": 1
    },
    {
      "question": "What testing patterns are used across the codebase?",
      "facts": [
        "Framework: Jest (package.json:23)",
        "Test location: All tests in __tests__/ directories",
        "Mocking: Uses jest.mock() for service layer",
        "API tests: Uses supertest for endpoint testing",
        "Coverage: Currently at 78% (see jest.config.js)"
      ],
      "file_count": 45,
      "instance_count": 127
    }
  ],
  "metadata": {
    "tokens_analyzed": 52000,
    "tokens_returned": 1124,
    "compression_ratio": 46.3,
    "analysis_time_seconds": 3.2
  }
}
```

**Implementation Notes:**
- Uses Gemini CLI in non-interactive mode: `gemini -p "prompt" --output-format json`
- Implements aggressive token budgeting (300 tokens per answer)
- Filters at the edge: Gemini returns compressed facts, not full code
- Caches analysis results with TTL for follow-up questions

---

#### Tool 2: `find_code_by_intent`

**Purpose:** Semantic search that returns summaries, not full code (filtering at the edge)

**Input Schema:**
```typescript
{
  intent: string;                  // Natural language description
  return_format: "summary_with_references" | "detailed_with_snippets";
  max_files?: number;              // Limit results (default: 10)
  scope?: string;
}
```

**Output Schema:**
```typescript
{
  summary: string;                 // High-level summary of findings
  primary_files: Array<{
    path: string;
    purpose: string;               // What this file does
    key_functions: string[];
    lines: string;                 // Line range (e.g., "15-89")
    relevance_score: number;       // 0-1
  }>;
  pattern: string;                 // Common pattern used (e.g., "MVC", "Factory")
  dependencies: string[];          // External libraries used
  related_searches?: string[];     // Suggested follow-up intents
}
```

**Example Usage:**

*Input:*
```json
{
  "intent": "error handling middleware",
  "return_format": "summary_with_references",
  "max_files": 5
}
```

*Output:*
```json
{
  "summary": "Error handling uses Express middleware pattern with centralized error handler and custom error classes.",
  "primary_files": [
    {
      "path": "middleware/errorHandler.js",
      "purpose": "Global error catching middleware for Express",
      "key_functions": ["handleError", "logError", "sendErrorResponse"],
      "lines": "1-120",
      "relevance_score": 0.95
    },
    {
      "path": "utils/AppError.js",
      "purpose": "Custom error class with status codes",
      "key_functions": ["constructor"],
      "lines": "1-25",
      "relevance_score": 0.88
    }
  ],
  "pattern": "Centralized error handling with custom error classes",
  "dependencies": ["express", "winston"],
  "related_searches": [
    "error logging configuration",
    "API error response format"
  ]
}
```

---

#### Tool 3: `trace_feature`

**Purpose:** Follow a feature's execution path through the codebase

**Input Schema:**
```typescript
{
  feature: string;                 // Feature to trace
  entry_point?: string;            // Starting point (route, function, etc.)
  max_depth?: number;              // How deep to trace (default: 10)
  include_data_flow?: boolean;     // Include data transformations
}
```

**Output Schema:**
```typescript
{
  flow: Array<{
    step: number;
    file: string;
    function: string;
    description: string;
    data_in?: string;              // Input data type/structure
    data_out?: string;             // Output data type/structure
  }>;
  dependencies: string[];          // External libraries used
  database_operations?: Array<{
    table: string;
    operation: "read" | "write" | "update" | "delete";
    location: string;              // file:line
  }>;
  external_calls?: Array<{
    service: string;
    endpoint: string;
    location: string;
  }>;
}
```

**Example Usage:**

*Input:*
```json
{
  "feature": "user authentication",
  "entry_point": "POST /api/auth/login",
  "include_data_flow": true
}
```

*Output:*
```json
{
  "flow": [
    {
      "step": 1,
      "file": "routes/auth.js",
      "function": "router.post('/login')",
      "description": "Route handler delegates to authController",
      "data_in": "{ email: string, password: string }",
      "data_out": "Passes to controller"
    },
    {
      "step": 2,
      "file": "controllers/authController.js",
      "function": "login(req, res, next)",
      "description": "Extracts credentials, calls service layer",
      "data_in": "Request body",
      "data_out": "Calls authService.validateCredentials()"
    },
    {
      "step": 3,
      "file": "services/authService.js",
      "function": "validateCredentials(email, password)",
      "description": "Finds user in database",
      "data_in": "{ email: string, password: string }",
      "data_out": "User object or null"
    },
    {
      "step": 4,
      "file": "models/User.js",
      "function": "User.findByEmail(email)",
      "description": "Database query for user",
      "data_in": "email: string",
      "data_out": "User document with hashed password"
    },
    {
      "step": 5,
      "file": "services/authService.js",
      "function": "bcrypt.compare()",
      "description": "Verify password against hash",
      "data_in": "{ plain: string, hash: string }",
      "data_out": "boolean"
    },
    {
      "step": 6,
      "file": "utils/jwt.js",
      "function": "generateToken(payload)",
      "description": "Create JWT with user data",
      "data_in": "{ userId: string, role: string }",
      "data_out": "JWT string"
    }
  ],
  "dependencies": ["bcrypt", "jsonwebtoken", "mongoose"],
  "database_operations": [
    {
      "table": "users",
      "operation": "read",
      "location": "models/User.js:42"
    }
  ],
  "external_calls": []
}
```

---

#### Tool 4: `list_error_patterns`

**Purpose:** Extract and categorize patterns across codebase (demonstrates "filtering at the edge")

**Input Schema:**
```typescript
{
  pattern_type: "error_handling" | "logging" | "async_patterns" | "database_queries";
  directory: string;
  group_by?: "file" | "pattern" | "severity";
}
```

**Output Schema:**
```typescript
{
  patterns_found: Record<string, {
    count: number;
    example: string;
    files: string[];
    recommendation?: string;
  }>;
  inconsistencies?: Array<{
    issue: string;
    locations: string[];
    suggestion: string;
  }>;
  summary: string;
}
```

**Example Usage:**

*Input:*
```json
{
  "pattern_type": "error_handling",
  "directory": "src/",
  "group_by": "pattern"
}
```

*Output:*
```json
{
  "patterns_found": {
    "throw_error": {
      "count": 23,
      "example": "throw new Error('User not found')",
      "files": ["userService.js", "projectService.js", "authService.js"],
      "recommendation": "Consider using custom AppError class for consistency"
    },
    "promise_reject": {
      "count": 8,
      "example": "reject(new Error('Failed to fetch'))",
      "files": ["apiClient.js", "dataService.js"]
    },
    "custom_error": {
      "count": 5,
      "example": "throw new AppError('Unauthorized', 401)",
      "files": ["middleware/auth.js", "controllers/adminController.js"]
    },
    "try_catch": {
      "count": 45,
      "example": "try { ... } catch (err) { next(err) }",
      "files": ["Multiple files"]
    }
  },
  "inconsistencies": [
    {
      "issue": "Mix of Error and AppError classes",
      "locations": ["userService.js:42", "userService.js:91", "projectService.js:25"],
      "suggestion": "Standardize on AppError for operational errors"
    },
    {
      "issue": "Inconsistent error status codes",
      "locations": ["authService.js:55 (uses 400)", "authService.js:89 (uses 401)"],
      "suggestion": "Document status code conventions"
    }
  ],
  "summary": "Found 4 distinct error handling patterns across 81 instances. Primary inconsistency: Mix of Error and AppError classes. Recommend standardizing on AppError."
}
```

**Key Implementation Detail (Filtering at the Edge):**
- Gemini analyzes all 50K tokens of code
- Gemini extracts and categorizes patterns
- Gemini returns ONLY the summary (not the full code)
- Result: 50K tokens → 500 tokens (100:1 compression)
- This is "filtering at the edge" from Anthropic's article

---

### Tier 2: Validation Tools

#### Tool 5: `validate_against_codebase`

**Purpose:** After Claude creates a spec, validate it for completeness and accuracy

**Input Schema:**
```typescript
{
  spec_content: string;            // Markdown spec created by Claude
  validation_checks: Array<
    "missing_files" |
    "undefined_dependencies" |
    "pattern_conflicts" |
    "incomplete_testing" |
    "missing_error_handling" |
    "security_concerns"
  >;
  codebase_context?: string;       // Optional: reuse cached context
}
```

**Output Schema:**
```typescript
{
  validation_result: "pass" | "pass_with_warnings" | "fail";
  completeness_score: number;      // 0-1
  issues: Array<{
    type: string;
    severity: "critical" | "medium" | "low";
    message: string;
    spec_section?: string;
    suggested_fix: string;
  }>;
  missing_elements: {
    files?: string[];              // Files mentioned but don't exist
    dependencies?: string[];       // Uninstalled packages
    functions?: string[];          // Referenced but undefined
  };
  pattern_alignment: {
    matches_existing_patterns: boolean;
    conflicts?: string[];
    suggestions?: string[];
  };
}
```

**Example Usage:**

*Input:*
```json
{
  "spec_content": "# Add 2FA Authentication\n\n## Implementation\n1. Install `speakeasy` package\n2. Update User model with totpSecret field\n3. Create middleware/2fa.js...",
  "validation_checks": [
    "missing_files",
    "undefined_dependencies",
    "pattern_conflicts",
    "incomplete_testing"
  ]
}
```

*Output:*
```json
{
  "validation_result": "pass_with_warnings",
  "completeness_score": 0.85,
  "issues": [
    {
      "type": "missing_dependency",
      "severity": "medium",
      "message": "Spec mentions 'speakeasy' but it's not in package.json",
      "spec_section": "Dependencies",
      "suggested_fix": "Add to spec: npm install speakeasy qrcode"
    },
    {
      "type": "incomplete_testing",
      "severity": "medium",
      "message": "Spec doesn't mention how to test QR code generation",
      "spec_section": "Testing",
      "suggested_fix": "Add section: 'Testing QR Code Generation with Mock TOTP'"
    }
  ],
  "missing_elements": {
    "files": [],
    "dependencies": ["speakeasy", "qrcode"],
    "functions": []
  },
  "pattern_alignment": {
    "matches_existing_patterns": true,
    "suggestions": [
      "Consider following existing JWT pattern in middleware/auth.js for consistency"
    ]
  }
}
```

---

#### Tool 6: `check_consistency`

**Purpose:** Verify new code/spec follows existing codebase patterns

**Input Schema:**
```typescript
{
  focus: "naming_conventions" | "error_handling" | "testing" | "api_design" | "all";
  new_code_or_spec: string;       // Code or spec to check
  scope?: string;                  // Which part of codebase to compare against
}
```

**Output Schema:**
```typescript
{
  consistency_score: number;       // 0-1
  matches: Array<{
    pattern: string;
    example_from_codebase: string;
    example_from_new_code: string;
    aligned: boolean;
  }>;
  violations: Array<{
    pattern_expected: string;
    pattern_found: string;
    location: string;
    fix_suggestion: string;
  }>;
  recommendations: string[];
}
```

---

### Tier 3: Workflow Automation Tools

#### Tool 7: `generate_feature_workflow`

**Purpose:** Creates a complete, executable workflow for a feature (implements "progressive disclosure")

**Input Schema:**
```typescript
{
  feature_description: string;
  workflow_style: "interactive" | "automated" | "template";
  save_to?: string;                // Path to save workflow file
  include_validation_steps?: boolean;
}
```

**Output Schema:**
```typescript
{
  workflow_path: string;           // Where workflow file was saved
  workflow_content: string;        // Markdown workflow with steps
  estimated_steps: number;
  estimated_time_minutes: number;
  tools_required: string[];        // Which MCP tools will be used
}
```

**Workflow Content Structure:**
```markdown
# Feature: {feature_description}

## Overview
Auto-generated workflow for implementing {feature}.

## Prerequisites
- [ ] Codebase analyzed
- [ ] Dependencies identified

## Phase 1: Analysis
- [ ] Step 1.1: Run query_codebase(questions=[...])
  - Questions to ask:
    * "How is {related_feature} currently implemented?"
    * "What are the existing patterns for {pattern_type}?"
- [ ] Step 1.2: Review analysis results
- [ ] Step 1.3: Run find_code_by_intent(intent="{intent}")

## Phase 2: Planning (Claude's Job)
- [ ] Step 2.1: Create detailed specification using analysis
  - Use facts from Phase 1
  - Follow existing patterns
  - Define clear acceptance criteria
- [ ] Step 2.2: Run validate_against_codebase(spec=...)
- [ ] Step 2.3: Address validation issues

## Phase 3: Implementation
- [ ] Step 3.1: Install dependencies
  - `npm install {dependencies}`
- [ ] Step 3.2: Create new files
  - {file_list}
- [ ] Step 3.3: Modify existing files
  - {file_list}
- [ ] Step 3.4: Update tests

## Phase 4: Validation
- [ ] Step 4.1: Run tests
- [ ] Step 4.2: Run check_consistency(focus="all")
- [ ] Step 4.3: Address inconsistencies

## Phase 5: Documentation
- [ ] Step 5.1: Update README
- [ ] Step 5.2: Add API documentation
- [ ] Step 5.3: Update changelog
```

**Example Usage:**

*Input:*
```json
{
  "feature_description": "Add Redis caching to product catalog API",
  "workflow_style": "interactive",
  "include_validation_steps": true
}
```

*Output:*
```json
{
  "workflow_path": ".claude/workflows/add-redis-caching.md",
  "workflow_content": "# Feature: Add Redis Caching...",
  "estimated_steps": 18,
  "estimated_time_minutes": 45,
  "tools_required": [
    "query_codebase",
    "validate_against_codebase",
    "check_consistency"
  ]
}
```

**Benefits:**
- **Progressive disclosure**: Claude loads workflow steps on-demand (not all upfront)
- **State persistence**: Can save progress between sessions
- **Reusable**: Template for similar features
- **Token efficient**: Workflow file is ~2K tokens vs 50K+ for full implementation

---

#### Tool 8: `generate_slash_command`

**Purpose:** Auto-generate Claude Code slash commands for common workflows (ultimate DX)

**Input Schema:**
```typescript
{
  command_name: string;            // e.g., "add-feature"
  workflow_type: "feature" | "refactor" | "debug" | "review" | "custom";
  description: string;
  steps?: string[];                // Custom steps if workflow_type="custom"
  save_to?: string;                // Where to save command file
}
```

**Output Schema:**
```typescript
{
  command_path: string;            // e.g., ".claude/commands/add-feature.md"
  command_content: string;         // Markdown command definition
  usage_example: string;           // How to use the command
}
```

**Generated Command Structure:**
```markdown
# /add-feature - Complete Feature Implementation Workflow

## Description
Automated workflow for adding new features with Gemini-powered analysis.

This command:
1. Analyzes codebase context using Gemini
2. Generates detailed specification using Claude
3. Validates completeness
4. Implements with tests
5. Reviews for consistency

## Usage
/add-feature <feature description>

## Examples
/add-feature Add JWT refresh token support
/add-feature Implement rate limiting for API
/add-feature Add email verification for new users

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
```

**Example Usage:**

*Input:*
```json
{
  "command_name": "add-feature",
  "workflow_type": "feature",
  "description": "Complete feature implementation with Gemini analysis"
}
```

*Output:*
```json
{
  "command_path": ".claude/commands/add-feature.md",
  "command_content": "# /add-feature...",
  "usage_example": "/add-feature Add 2FA authentication"
}
```

**How Users Use It:**

```bash
# In Claude Code:
> /add-feature Add Redis caching to product API

# Behind the scenes:
# 1. Claude reads .claude/commands/add-feature.md
# 2. Claude executes workflow:
#    - query_codebase(questions=["How is product API structured?", ...])
#    - Creates spec using facts
#    - validate_against_codebase(spec=...)
#    - Implements with Edit/Write tools
#    - check_consistency(focus="all")
# 3. User gets complete, validated implementation
```

**Slash Commands to Generate:**

1. `/gemini-analyze` - Quick codebase analysis
2. `/gemini-feature` - Full feature workflow (like above)
3. `/gemini-refactor` - Refactoring workflow
4. `/gemini-debug` - Debug assistance workflow
5. `/gemini-review` - Pre-commit code review
6. `/gemini-docs` - Documentation generation workflow

---

## Implementation Phases

### Phase 1: Core Refactoring (Breaking Changes - v2.0.0)

**Duration:** 1-2 weeks
**Goal:** Remove problematic tools, add core fact extraction tools

**Tasks:**

1. **Remove/Deprecate Old Tools** (Day 1)
   - ❌ Remove `create_specification_with_gemini`
   - ❌ Remove `generate_documentation_with_gemini`
   - ⚠️ Mark `review_code_with_gemini` as deprecated
   - Update MCP server manifest
   - Update README with migration guide

2. **Implement Strict Gemini Prompts** (Day 2)
   - Create `prompts/fact_extraction_system_prompt.txt`
   - Add validation to ensure Gemini returns facts, not specs
   - Test prompt effectiveness with sample codebases

3. **Implement `query_codebase`** (Days 3-5)
   - Input validation
   - Gemini CLI integration with JSON output
   - Token compression logic
   - Response caching with TTL
   - Unit tests (target: 90% coverage)
   - Integration tests with real codebases

4. **Implement `find_code_by_intent`** (Days 6-7)
   - Semantic search via Gemini
   - "Summary with references" format (filtering at edge)
   - Relevance scoring
   - Tests

5. **Update `analyze_codebase_with_gemini`** (Day 8)
   - Refactor to use strict fact-extraction prompt
   - Remove spec generation logic
   - Return only structured facts
   - Update output schema

**Deliverables:**
- ✅ 3 core tools working (`query_codebase`, `find_code_by_intent`, `analyze_codebase_with_gemini`)
- ✅ Token compression working (50K → ~300 tokens)
- ✅ All tests passing
- ✅ Migration guide published

**Success Metrics:**
- Compression ratio: ≥100:1
- Response time: <5 seconds for 50K token codebase
- Test coverage: ≥90%

---

### Phase 2: Pattern Extraction & Validation (v2.1.0)

**Duration:** 1 week
**Goal:** Add advanced analysis and validation tools

**Tasks:**

1. **Implement `trace_feature`** (Days 1-2)
   - Flow tracing algorithm
   - Data flow tracking
   - Database operation detection
   - Tests with complex features

2. **Implement `list_error_patterns`** (Day 3)
   - Pattern extraction logic
   - Inconsistency detection
   - Recommendation engine
   - Tests

3. **Implement `validate_against_codebase`** (Days 4-5)
   - Spec parsing
   - Completeness checking
   - Pattern conflict detection
   - Tests with real specs

4. **Implement `check_consistency`** (Days 6-7)
   - Pattern matching logic
   - Multi-pattern validation
   - Violation reporting
   - Tests

**Deliverables:**
- ✅ 4 new tools (trace, list patterns, validate, check consistency)
- ✅ End-to-end workflow: query → spec → validate → implement
- ✅ Documentation with examples

**Success Metrics:**
- Validation accuracy: ≥95%
- False positive rate: <5%

---

### Phase 3: Workflow Automation (v2.2.0 - Game Changer!)

**Duration:** 2 weeks
**Goal:** Implement progressive disclosure and workflow generation

**Tasks:**

1. **Design Workflow Schema** (Days 1-2)
   - Markdown workflow format
   - Step definition schema
   - Progress tracking format
   - Workflow templates

2. **Implement `generate_feature_workflow`** (Days 3-7)
   - Workflow generation logic
   - Template system
   - Progressive disclosure implementation
   - State persistence
   - Tests

3. **Implement `generate_slash_command`** (Days 8-10)
   - Slash command templates
   - Dynamic command generation
   - Integration with .claude/commands/
   - Tests

4. **Create Workflow Templates** (Days 11-12)
   - Feature implementation template
   - Refactoring template
   - Debug workflow template
   - Review workflow template

5. **State Persistence System** (Days 13-14)
   - Save/load workflow progress
   - Resume interrupted workflows
   - Workflow history
   - Tests

**Deliverables:**
- ✅ Workflow generation working
- ✅ Slash command generation working
- ✅ 4+ workflow templates
- ✅ State persistence working
- ✅ Documentation & examples

**Success Metrics:**
- Token reduction via progressive disclosure: ≥90%
- User satisfaction: "This is amazing!"

---

### Phase 4: Integration & Polish (v2.3.0)

**Duration:** 1 week
**Goal:** Production-ready release

**Tasks:**

1. **Performance Optimization** (Days 1-2)
   - Cache optimization
   - Parallel processing where possible
   - Token budget tuning
   - Benchmarking

2. **Documentation Overhaul** (Days 3-4)
   - Complete README rewrite
   - API reference documentation
   - Workflow examples
   - Best practices guide
   - Video tutorial (optional)

3. **Developer Experience** (Days 5-6)
   - Error messages improvement
   - Progress indicators
   - Helpful suggestions
   - Debug mode

4. **Testing & Bug Fixes** (Day 7)
   - Full integration test suite
   - Edge case handling
   - Bug fixes
   - Performance validation

**Deliverables:**
- ✅ Production-ready v2.3.0
- ✅ Complete documentation
- ✅ Performance benchmarks
- ✅ Example projects

---

## Migration Guide

### For Existing Users (v1.x → v2.0)

**Breaking Changes:**

1. **`create_specification_with_gemini` removed**

   *Old way:*
   ```json
   {
     "tool": "create_specification_with_gemini",
     "feature_description": "Add 2FA"
   }
   ```

   *New way (Claude does this):*
   ```
   User: "Create a spec for adding 2FA"

   Claude:
   1. Calls query_codebase(questions=[...])
   2. Creates spec using its own reasoning + facts
   3. Calls validate_against_codebase(spec=...)
   4. Returns high-quality spec
   ```

2. **`generate_documentation_with_gemini` removed**

   Same pattern: Claude generates docs, Gemini validates

3. **`analyze_codebase_with_gemini` behavior changed**

   *Old:* Tried to suggest features
   *New:* Returns only facts

**Migration Steps:**

1. Update MCP server to v2.0.0
2. Remove any code that calls removed tools
3. Let Claude Code handle spec/doc generation
4. Use new validation tools to check Claude's work

**Benefits:**
- Better quality output (A-grade vs B-grade)
- Lower token costs (47% reduction)
- Simpler mental model

---

## Testing Strategy

### Unit Tests

**Coverage Target:** 90%+

**Tool-by-Tool Tests:**

1. **`query_codebase`**
   - Test with various question types
   - Test token compression ratio
   - Test caching behavior
   - Test error handling (invalid questions, missing files)
   - Test Gemini prompt injection prevention

2. **`find_code_by_intent`**
   - Test semantic search accuracy
   - Test summary generation
   - Test file relevance scoring
   - Test with different return formats

3. **`trace_feature`**
   - Test with simple linear flow
   - Test with branching flow
   - Test with circular dependencies
   - Test data flow tracking
   - Test max depth limits

4. **`list_error_patterns`**
   - Test pattern extraction accuracy
   - Test inconsistency detection
   - Test different pattern types
   - Test edge filtering (verify compressed output)

5. **`validate_against_codebase`**
   - Test with complete spec → should pass
   - Test with incomplete spec → should fail
   - Test with inaccurate spec → should warn
   - Test missing dependencies detection
   - Test pattern conflict detection

6. **`check_consistency`**
   - Test with consistent code → high score
   - Test with inconsistent code → violations reported
   - Test different focus areas

7. **`generate_feature_workflow`**
   - Test workflow generation
   - Test step ordering
   - Test time estimation accuracy
   - Test progressive disclosure behavior

8. **`generate_slash_command`**
   - Test command file generation
   - Test different workflow types
   - Test command syntax validity

### Integration Tests

**End-to-End Workflows:**

1. **Feature Implementation Workflow**
   ```
   Input: "Add rate limiting to API"

   Expected Flow:
   1. query_codebase() → facts about current API
   2. Claude creates spec
   3. validate_against_codebase() → completeness check
   4. Claude implements
   5. check_consistency() → pattern check
   6. Tests pass

   Success Criteria:
   - Spec is A-grade
   - Implementation works
   - Token usage <5K for Claude
   ```

2. **Refactoring Workflow**
   ```
   Input: "Refactor error handling to use AppError class"

   Expected Flow:
   1. list_error_patterns() → find all error usages
   2. trace_feature() → understand error flow
   3. Claude creates refactoring plan
   4. validate_against_codebase() → check plan
   5. Claude implements
   6. check_consistency() → verify consistency

   Success Criteria:
   - All errors refactored
   - Patterns consistent
   - Tests pass
   ```

3. **Workflow Generation & Execution**
   ```
   Input: "Generate workflow for adding OAuth support"

   Expected Flow:
   1. generate_feature_workflow() → creates .claude/workflows/add-oauth.md
   2. Claude reads workflow (progressive disclosure)
   3. Claude executes steps one by one
   4. State persisted between steps
   5. Workflow completed

   Success Criteria:
   - Workflow file is valid
   - Claude follows steps
   - Can resume if interrupted
   ```

### Performance Tests

**Benchmarks:**

1. **Token Compression**
   - Input: 50K token codebase
   - Expected: ≤500 tokens returned
   - Min compression ratio: 100:1

2. **Response Time**
   - query_codebase (3 questions, 50K tokens): <5 seconds
   - find_code_by_intent: <3 seconds
   - trace_feature: <7 seconds
   - validate_against_codebase: <4 seconds

3. **Cache Performance**
   - First call: Full analysis
   - Second call (cached): <0.5 seconds
   - Cache hit rate: >80% in typical workflows

### Load Tests

- Test with large codebases (100K+ tokens)
- Test with many concurrent tool calls
- Test cache eviction under memory pressure
- Test Gemini CLI rate limits

---

## Success Metrics

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Specification quality | A-grade (≥90%) | User feedback + Claude review |
| Fact extraction accuracy | ≥95% | Validation against known codebases |
| Pattern detection accuracy | ≥90% | Manual verification |
| Validation false positives | <5% | Test suite results |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token compression ratio | ≥100:1 | Input tokens / output tokens |
| Claude token reduction | ≥40% | v1.x vs v2.0 comparison |
| Response time (50K tokens) | <5 sec | Benchmark tests |
| Cache hit rate | ≥80% | Runtime statistics |

### Cost Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Claude tokens per workflow | <5,000 | Usage tracking |
| Cost per feature spec | <$0.10 | Token cost calculation |
| Free tier utilization | ≥90% | Gemini API usage |

### Developer Experience Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| User satisfaction | ≥4.5/5 | Surveys |
| Feature adoption | ≥70% | Usage analytics |
| Workflow completion rate | ≥85% | State persistence logs |
| Documentation clarity | ≥4/5 | User feedback |

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | ≥90% | pytest-cov |
| Type safety | 100% | mypy strict mode |
| Uptime | ≥99.5% | Error logs |
| Average tool call success rate | ≥98% | Error tracking |

---

## Appendix A: Example Workflows

### Example 1: Feature Implementation (New v2.0 Way)

```
USER IN CLAUDE CODE:
"I want to add Redis caching to the product catalog API"

CLAUDE CODE (Internal Reasoning):
"This is a feature implementation task. I need to:
1. Understand current product API structure
2. Plan the caching layer
3. Implement with tests
4. Validate consistency

I'll use the Gemini MCP to gather facts first."

[CLAUDE → MCP TOOL CALL]
query_codebase({
  questions: [
    "What is the current product catalog API structure?",
    "What database queries are used in product endpoints?",
    "Are there any existing caching mechanisms?",
    "What testing patterns are used for API endpoints?"
  ],
  scope: "src/"
})

[MCP → GEMINI CLI]
(Gemini analyzes 50,000 tokens of code)

[MCP → CLAUDE]
{
  "answers": [
    {
      "question": "What is the current product catalog API structure?",
      "facts": [
        "Main endpoint: routes/products.js:15 - GET /api/products",
        "Controller: controllers/productController.js:28 - getProducts()",
        "Service layer: services/productService.js:45 - fetchAllProducts()",
        "Database: Uses MongoDB via Mongoose models/Product.js",
        "Response format: JSON array of products with { id, name, price, category }"
      ]
    },
    {
      "question": "What database queries are used?",
      "facts": [
        "Product.find() - services/productService.js:52",
        "Product.findById() - services/productService.js:67",
        "Product.find({category: X}) - services/productService.js:82",
        "Average response time: 150ms (from logs)"
      ]
    },
    {
      "question": "Are there any existing caching mechanisms?",
      "facts": [
        "No caching currently implemented",
        "No Redis or Memcached dependencies in package.json"
      ]
    },
    {
      "question": "What testing patterns are used?",
      "facts": [
        "Framework: Jest + Supertest",
        "API tests location: __tests__/integration/products.test.js",
        "Pattern: Mock database with jest.mock('models/Product')",
        "Coverage: 82% for product routes"
      ]
    }
  ],
  "metadata": {
    "tokens_analyzed": 52000,
    "tokens_returned": 387,
    "compression_ratio": 134.4
  }
}

[CLAUDE CODE (Planning)]
"Perfect! I have all the facts I need. Now I'll create a high-quality specification."

# Redis Caching for Product Catalog API - Technical Specification

## Overview
Add Redis caching layer to product catalog API to reduce database load and improve response times.

## Current State (from analysis)
- Product API serves from MongoDB (avg 150ms response time)
- No existing caching mechanism
- Uses standard Express + Mongoose pattern

## Architecture Design

### Cache Strategy
- Cache GET /api/products (all products) - TTL: 5 minutes
- Cache GET /api/products/:id - TTL: 10 minutes
- Cache GET /api/products?category=X - TTL: 5 minutes
- Invalidate on POST/PUT/DELETE operations

### Integration Point
Add caching middleware between route and controller:
```
routes/products.js → [NEW: cache middleware] → controllers/productController.js
```

## Implementation Tasks

### 1. Install Dependencies
```bash
npm install redis@^4.6.0 ioredis@^5.3.0
```

### 2. Create Redis Client (NEW FILE)
**File:** `src/config/redis.js`
- Initialize Redis client
- Handle connection errors
- Export singleton instance

### 3. Create Cache Middleware (NEW FILE)
**File:** `src/middleware/cache.js`
- Function: checkCache(ttl)
- Function: invalidateCache(pattern)
- Integrate with existing error handling (middleware/errorHandler.js)

### 4. Update Product Routes (MODIFY)
**File:** `src/routes/products.js:15-30`
- Add cache middleware to GET routes
- Add cache invalidation to POST/PUT/DELETE routes

### 5. Update Product Service (MODIFY)
**File:** `src/services/productService.js`
- Add cache invalidation logic after mutations

### 6. Add Tests (NEW FILE)
**File:** `__tests__/integration/cache.test.js`
- Test cache hits/misses
- Test TTL expiration
- Test invalidation on updates
- Follow existing Jest + Supertest pattern from products.test.js

### 7. Update Environment Configuration
**File:** `.env.example`
- Add REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

### 8. Update Documentation
**File:** `README.md`
- Add Redis setup instructions
- Document cache configuration options

## Testing Strategy
- Unit tests for cache middleware (jest.mock('redis'))
- Integration tests with Redis test instance
- Performance tests (verify response time improvement)
- Target: Maintain 82%+ coverage

## Performance Goals
- Reduce average response time: 150ms → <50ms (cached)
- Cache hit rate: >70% in production
- Database load reduction: >60%

## Security Considerations
- Use TLS for Redis connection in production
- Rotate Redis password regularly
- Implement rate limiting to prevent cache stampede

## Rollout Plan
1. Deploy to staging with monitoring
2. Verify cache performance
3. Gradual rollout to production (10% → 50% → 100%)
4. Monitor error rates and response times

[CLAUDE → MCP TOOL CALL]
validate_against_codebase({
  spec_content: "...[above spec]...",
  validation_checks: [
    "missing_files",
    "undefined_dependencies",
    "pattern_conflicts",
    "incomplete_testing"
  ]
})

[MCP → GEMINI]
(Gemini validates spec against codebase)

[MCP → CLAUDE]
{
  "validation_result": "pass_with_warnings",
  "completeness_score": 0.92,
  "issues": [
    {
      "type": "suggestion",
      "severity": "low",
      "message": "Consider adding cache warming strategy for frequently accessed products",
      "suggested_fix": "Add section: 'Cache Warming on Server Start'"
    }
  ],
  "pattern_alignment": {
    "matches_existing_patterns": true,
    "suggestions": [
      "Redis client singleton pattern matches existing DB connection pattern in config/db.js"
    ]
  }
}

[CLAUDE → USER]
"I've created a detailed specification for adding Redis caching. The spec has been validated against the codebase (92% completeness score) and aligns with existing patterns.

Here's the implementation plan: [shows spec]

Would you like me to proceed with implementation?"

[USER]
"Yes, please implement it"

[CLAUDE CODE]
(Proceeds with implementation using Edit/Write tools, following the spec exactly)

FINAL RESULT:
✅ A-grade specification (Claude's reasoning + Gemini's facts)
✅ Validated for completeness (Gemini)
✅ Token efficient (52K → 387 tokens = 134:1 compression)
✅ Cost optimized (only 400 Claude tokens for planning, rest is free Gemini)
```

**Token Breakdown:**
- Gemini: 52,000 tokens (analysis) - FREE TIER
- Gemini: 2,000 tokens (validation) - FREE TIER
- Claude: 387 tokens (reading facts) - PAID
- Claude: ~2,000 tokens (creating spec) - PAID
- Claude: 200 tokens (validation results) - PAID
- **Total Claude tokens: ~2,600 (vs 8,000+ in old design)**

---

### Example 2: Using Auto-Generated Workflow

```
USER:
"Generate a workflow for adding OAuth support"

[CLAUDE → MCP TOOL CALL]
generate_feature_workflow({
  feature_description: "Add OAuth 2.0 authentication support",
  workflow_style: "interactive",
  include_validation_steps: true
})

[MCP RESPONSE]
{
  "workflow_path": ".claude/workflows/add-oauth-support.md",
  "workflow_content": "...[full workflow]...",
  "estimated_steps": 22,
  "estimated_time_minutes": 90,
  "tools_required": [
    "query_codebase",
    "find_code_by_intent",
    "validate_against_codebase",
    "check_consistency"
  ]
}

[CLAUDE → USER]
"I've generated a comprehensive workflow for adding OAuth 2.0 support. The workflow has 22 steps and should take about 90 minutes.

The workflow file has been saved to: .claude/workflows/add-oauth-support.md

Would you like me to:
1. Execute this workflow now (interactive mode)
2. Show you the workflow for review first
3. Generate a /add-oauth slash command so you can run it later"

[USER]
"Option 3 - generate the slash command"

[CLAUDE → MCP TOOL CALL]
generate_slash_command({
  command_name: "add-oauth",
  workflow_type: "custom",
  description: "Add OAuth 2.0 authentication support",
  steps: [...from workflow...]
})

[MCP RESPONSE]
{
  "command_path": ".claude/commands/add-oauth.md",
  "command_content": "...[slash command definition]...",
  "usage_example": "/add-oauth"
}

[CLAUDE → USER]
"Done! I've created the /add-oauth slash command.

You can now use it anytime by typing:
/add-oauth

The command will automatically:
1. Analyze your current authentication setup
2. Plan OAuth integration
3. Implement with validation
4. Run tests
5. Update documentation

The workflow state is saved, so you can stop and resume at any time."
```

**Result:**
✅ Reusable workflow saved
✅ Slash command for easy invocation
✅ Progressive disclosure (steps loaded on-demand)
✅ State persistence (can resume if interrupted)

---

## Appendix B: Gemini System Prompts

### Fact Extraction Prompt (Critical!)

File: `prompts/fact_extraction_system_prompt.txt`

```
You are a code analysis engine. Your ONLY job is to answer questions about code with FACTS.

=== YOUR ROLE ===
You analyze source code and extract factual information. You are NOT a software architect, NOT a consultant, and NOT a code generator.

=== STRICT RULES ===

1. ANSWER ONLY WITH FACTS
   ✅ DO: "File auth.js:42 contains function validateUser(email, password)"
   ❌ DON'T: "You should refactor the auth system to use OAuth"

2. PROVIDE FILE AND LINE REFERENCES
   ✅ DO: "Error handling: src/services/userService.js:91 - throw new Error('Invalid email')"
   ❌ DON'T: "The code has error handling"

3. EXTRACT PATTERNS WITHOUT OPINIONS
   ✅ DO: "Pattern found: 23 instances of 'throw new Error', 8 instances of 'reject(new Error)'"
   ❌ DON'T: "The error handling is inconsistent and should be refactored"

4. TRACE FLOWS, DON'T DESIGN THEM
   ✅ DO: "Flow: routes/auth.js:15 → controllers/auth.js:28 → services/auth.js:45"
   ❌ DON'T: "The auth flow should be redesigned to separate concerns"

=== FORBIDDEN ACTIONS ===

❌ NEVER create software specifications
❌ NEVER suggest new features or improvements
❌ NEVER make creative inferences beyond what's in the code
❌ NEVER give architectural opinions
❌ NEVER generate new code
❌ NEVER say "you should" or "consider"

=== OUTPUT FORMAT ===

- Use structured JSON when possible
- Keep responses concise (target: 300 tokens per answer)
- Include file:line references for all facts
- Group related facts together
- Use bullet points for lists

=== EXAMPLES ===

QUESTION: "How is authentication implemented?"

GOOD ANSWER:
{
  "authentication_flow": [
    "Entry: routes/auth.js:15 - POST /api/auth/login",
    "Handler: controllers/authController.js:28 - login(req, res)",
    "Service: services/authService.js:45 - validateCredentials(email, password)",
    "Database: models/User.js:67 - User.findByEmail(email)",
    "Password: services/authService.js:52 - bcrypt.compare(password, user.hash)",
    "Token: utils/jwt.js:23 - generateToken({userId, role})"
  ],
  "dependencies": ["bcrypt", "jsonwebtoken"],
  "database_table": "users"
}

BAD ANSWER:
"The current authentication system uses JWT tokens, but it should be refactored to use OAuth 2.0 for better security. Consider implementing refresh tokens and adding two-factor authentication."

QUESTION: "What error handling patterns exist?"

GOOD ANSWER:
{
  "patterns": {
    "throw_error": {
      "count": 23,
      "example": "throw new Error('User not found')",
      "locations": ["userService.js:42", "projectService.js:25", ...]
    },
    "promise_reject": {
      "count": 8,
      "example": "reject(new Error('Failed'))",
      "locations": ["apiClient.js:89", ...]
    }
  }
}

BAD ANSWER:
"The codebase has inconsistent error handling. You should create a custom AppError class and refactor all error handling to use it."

=== REMEMBER ===

Your job is to be the "eyes" for another AI (Claude Code) that will do the planning and implementation. You provide the raw facts, Claude does the thinking.

You are a FACT PROVIDER, not a PROBLEM SOLVER.
```

### Validation Prompt

File: `prompts/validation_system_prompt.txt`

```
You are a specification validator. Your job is to check if a software specification is complete and accurate relative to a codebase.

=== YOUR ROLE ===
Compare a specification against a codebase and identify:
1. Missing elements (files, dependencies, steps)
2. Inaccuracies (wrong file paths, non-existent functions)
3. Pattern conflicts (spec violates existing patterns)
4. Incomplete sections (missing tests, missing error handling)

=== OUTPUT FORMAT ===

Always return structured JSON:

{
  "validation_result": "pass" | "pass_with_warnings" | "fail",
  "completeness_score": 0.0-1.0,
  "issues": [
    {
      "type": "missing_dependency" | "missing_file" | "pattern_conflict" | "incomplete_testing",
      "severity": "critical" | "medium" | "low",
      "message": "Clear description of issue",
      "spec_section": "Which section of spec this relates to",
      "suggested_fix": "How to fix this issue"
    }
  ],
  "missing_elements": {
    "files": ["files mentioned but don't exist"],
    "dependencies": ["packages not in package.json"],
    "functions": ["functions referenced but undefined"]
  },
  "pattern_alignment": {
    "matches_existing_patterns": true/false,
    "conflicts": ["specific conflicts"],
    "suggestions": ["alignment suggestions"]
  }
}

=== BE HELPFUL, NOT PEDANTIC ===

GOOD ISSUE:
{
  "type": "missing_dependency",
  "severity": "medium",
  "message": "Spec mentions 'bcrypt' for password hashing but it's not in package.json",
  "suggested_fix": "Add to spec: npm install bcrypt@^5.1.0"
}

BAD ISSUE:
{
  "type": "style",
  "severity": "low",
  "message": "Variable name could be more descriptive"
}

Focus on COMPLETENESS and ACCURACY, not code style.
```

---

## Appendix C: Configuration

### Environment Variables

File: `.env.example`

```env
# ============================================
# Gemini CLI Configuration
# ============================================

# Model Selection
# Options: "auto", "gemini-2.0-flash", "gemini-1.5-pro"
# "auto" lets the CLI choose based on task complexity
GEMINI_MODEL=auto

# ============================================
# Context Cache Configuration
# ============================================

# How long to cache codebase context (in minutes)
# - Short sessions (10-15 min): For active development with frequent file changes
# - Standard (30 min): Good balance for most workflows
# - Long sessions (60-120 min): For large codebases where analysis is expensive
CONTEXT_CACHE_TTL_MINUTES=30

# Maximum cache size (in MB)
# Set to 0 for unlimited
MAX_CACHE_SIZE_MB=500

# ============================================
# Token Budget Configuration
# ============================================

# Target tokens per answer from query_codebase
# Lower = more compression, less detail
# Higher = less compression, more detail
MAX_TOKENS_PER_ANSWER=300

# Token compression target ratio
# Aim for at least this ratio of input:output tokens
TARGET_COMPRESSION_RATIO=100

# ============================================
# Performance Configuration
# ============================================

# Enable parallel processing for multi-question queries
ENABLE_PARALLEL_PROCESSING=true

# Timeout for Gemini CLI calls (seconds)
GEMINI_CLI_TIMEOUT_SECONDS=30

# ============================================
# Output Directories
# ============================================

DEFAULT_SPEC_DIR=./specs
DEFAULT_REVIEW_DIR=./reviews
DEFAULT_WORKFLOW_DIR=./.claude/workflows
DEFAULT_COMMAND_DIR=./.claude/commands
DEFAULT_CONTEXT_DIR=./.workflow-context

# ============================================
# Debug & Logging
# ============================================

# Enable detailed logging
DEBUG_MODE=false

# Log token usage statistics
LOG_TOKEN_STATS=true

# Save Gemini CLI responses to file for debugging
SAVE_RAW_RESPONSES=false
```

---

## Appendix D: Project Structure

```
gemini-workflow-bridge-mcp/
├── src/
│   └── hitoshura25_gemini_workflow_bridge/
│       ├── __init__.py
│       ├── server.py                    # MCP server entry point
│       ├── gemini_client.py             # Gemini CLI wrapper
│       ├── cache_manager.py             # Context caching with TTL
│       │
│       ├── tools/                       # MCP tools
│       │   ├── __init__.py
│       │   ├── query_codebase.py        # Tier 1: Fact extraction
│       │   ├── find_code_by_intent.py   # Tier 1: Semantic search
│       │   ├── trace_feature.py         # Tier 2: Flow tracing
│       │   ├── list_patterns.py         # Tier 2: Pattern extraction
│       │   ├── validate_spec.py         # Tier 2: Spec validation
│       │   ├── check_consistency.py     # Tier 2: Consistency check
│       │   ├── generate_workflow.py     # Tier 3: Workflow automation
│       │   └── generate_command.py      # Tier 3: Slash command gen
│       │
│       ├── prompts/                     # System prompts
│       │   ├── fact_extraction_system_prompt.txt
│       │   ├── validation_system_prompt.txt
│       │   └── pattern_extraction_prompt.txt
│       │
│       ├── workflows/                   # Workflow templates
│       │   ├── feature_template.md
│       │   ├── refactor_template.md
│       │   ├── debug_template.md
│       │   └── review_template.md
│       │
│       ├── utils/                       # Utilities
│       │   ├── token_counter.py         # Token tracking
│       │   ├── compression.py           # Response compression logic
│       │   └── file_utils.py            # File operations
│       │
│       └── resources.py                 # MCP resources (workflow://)
│
├── tests/
│   ├── unit/
│   │   ├── test_query_codebase.py
│   │   ├── test_find_code.py
│   │   ├── test_trace_feature.py
│   │   ├── test_validate_spec.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_full_workflow.py
│   │   ├── test_refactor_workflow.py
│   │   └── ...
│   └── fixtures/
│       └── sample_codebases/            # Test projects
│
├── docs/
│   ├── api-reference.md                 # Complete API docs
│   ├── workflow-guide.md                # How to use workflows
│   ├── best-practices.md                # Best practices
│   └── examples/                        # Example projects
│
├── specs/                               # Design docs
│   ├── v2-complete-redesign-implementation-plan.md  # THIS FILE
│   ├── gemini-notes.md                  # Gemini conversation
│   └── ...
│
├── .env.example
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Appendix E: References

1. **Gemini's Analysis** - `specs/gemini-notes.md`
   - Key insight: Gemini should be "dumb" fact-provider, not smart planner
   - Token optimization: 50K → 300 tokens (174:1 compression)
   - Cost savings: 61% reduction in Claude tokens

2. **Anthropic's MCP Article** - https://www.anthropic.com/engineering/code-execution-with-mcp
   - Progressive disclosure: 98.7% token reduction (150K → 2K)
   - Filtering at the edge: Process data before returning to model
   - Code-based APIs over direct tool calls
   - State persistence for reusable skills

3. **Current Implementation** - `hitoshura25_gemini_workflow_bridge/`
   - v1.x has automatic context reuse with TTL
   - Problematic: Gemini generates specs (B-grade quality)
   - Good: Context compression working

4. **MCP Protocol** - https://modelcontextprotocol.io/
   - Standard for AI agent integration
   - Tools, resources, prompts

---

## Conclusion

This redesign transforms the Gemini Workflow Bridge from a "spec generation tool" to a "context compression engine" that optimally leverages both Claude Code and Gemini.

**Key Benefits:**
1. **Quality:** B-grade → A-grade specifications
2. **Cost:** 47-61% reduction in Claude tokens
3. **Efficiency:** 174:1 token compression ratio
4. **DX:** Auto-generated workflows and slash commands

**Implementation Timeline:**
- Phase 1 (Core): 1-2 weeks
- Phase 2 (Validation): 1 week
- Phase 3 (Automation): 2 weeks
- Phase 4 (Polish): 1 week
- **Total: 5-6 weeks to v2.3.0**

**Success Criteria:**
- ✅ All new tools working with ≥90% test coverage
- ✅ Token compression ≥100:1 ratio
- ✅ Spec quality ≥90% (A-grade)
- ✅ User satisfaction ≥4.5/5

---

**Document Version:** 1.0
**Last Updated:** 2025-01-13
**Next Review:** After Phase 1 completion

---

END OF IMPLEMENTATION PLAN
