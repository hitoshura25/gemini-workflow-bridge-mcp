# Gemini Workflow Bridge MCP Server

MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review, leveraging Gemini's massive 2M token context window and cost-effectiveness for read-heavy operations.

## Overview

This MCP server extends Claude Code's capabilities by providing tools that delegate specific workflow tasks to Google's Gemini 2.0 Flash model. It's designed to optimize your development workflow by using each AI model's strengths:

- **Gemini**: Heavy context loading, codebase analysis, spec generation (2M token window)
- **Claude Code**: Precise code editing, implementation, and orchestration

## Why CLI-Based?

**CLI-Only Design (v1.0.0+):** This server uses the Gemini CLI instead of API calls. Key benefits:

- **Zero API Costs:** Uses your existing Gemini Code Assist subscription
- **Simple Auth:** Reuses your CLI credentials, no API key management
- **No Extra Setup:** If you have Gemini CLI installed, you're ready to go
- **Same Power:** Access to all Gemini models including 2.0 Flash

Perfect for developers who already have Gemini Code Assist!

## Features

### Tools

1. **`analyze_codebase_with_gemini`** - Analyze large codebases using Gemini's 2M token context
2. **`create_specification_with_gemini`** - Generate detailed technical specifications
3. **`review_code_with_gemini`** - Comprehensive code review with multiple focus areas
4. **`generate_documentation_with_gemini`** - Create documentation with full codebase context
5. **`ask_gemini`** - General-purpose queries with optional codebase context

### Resources

- **`workflow://specs/{name}`** - Access saved specifications
- **`workflow://reviews/{name}`** - Access saved code reviews
- **`workflow://context/{name}`** - Access cached codebase analysis

## Installation

### Prerequisites

- Python 3.11+
- Gemini CLI installed and authenticated

### Step 1: Install Gemini CLI

```bash
npm install -g @google/gemini-cli
```

### Step 2: Authenticate Gemini CLI

```bash
gemini
# Follow the authentication prompts
# Your credentials will be cached automatically
```

### Step 3: Install MCP Server via pip

```bash
pip install hitoshura25-gemini-workflow-bridge
```

### Install via uvx (recommended)

```bash
uvx hitoshura25-gemini-workflow-bridge
```

## Configuration

### Verify Gemini CLI is Ready

```bash
# Check CLI is installed
gemini --version
# Should show: 0.13.0 or higher

# Test CLI works
echo "What is 2+2?" | gemini
# Should return a response from Gemini
```

### Optional: Configure Model

Create a `.env` file (optional):

```env
# NO API KEY NEEDED!
# Use "auto" to let the CLI choose the best model automatically
# Pro models for complex tasks, Flash for simple/fast tasks
GEMINI_MODEL=auto

# Or specify a specific model:
# GEMINI_MODEL=gemini-2.0-flash
# GEMINI_MODEL=gemini-1.5-pro

DEFAULT_SPEC_DIR=./specs
DEFAULT_REVIEW_DIR=./reviews
DEFAULT_CONTEXT_DIR=./.workflow-context
```

See `.env.example` for all available options.

### Configure Claude Code

Add the server to your Claude Code MCP configuration (`~/.claude/config.json` or workspace `.claude/config.json`):

**Using uvx (recommended):**
```json
{
  "mcpServers": {
    "gemini-workflow": {
      "command": "uvx",
      "args": ["hitoshura25-gemini-workflow-bridge"]
    }
  }
}
```

**Or using pip:**
```json
{
  "mcpServers": {
    "gemini-workflow": {
      "command": "python",
      "args": ["-m", "hitoshura25_gemini_workflow_bridge.server"]
    }
  }
}
```

**Note:** No API key needed in config! The MCP server uses your Gemini CLI credentials automatically.

## Usage Examples

### How It Works: Automatic Context Loading

**NEW in v1.1**: All tools now automatically load and analyze your codebase when called, so you can use them directly without manual setup!

**Single Call Workflow** (Recommended for most use cases):
```
User: "Create a spec for adding 2FA authentication"

Claude Code:
[Calls: create_specification_with_gemini({
  feature_description: "2FA authentication"
})]

Behind the scenes:
1. ✅ Automatically loads codebase (*.py, *.js, *.ts, etc.)
2. ✅ Performs inline architectural analysis
3. ✅ Generates context-aware specification
4. ✅ Returns context_id for reuse in subsequent calls

Result: High-quality spec that understands your existing auth system!
```

**Optimized Multi-Call Workflow** (For multiple related operations):
```
# First call - auto-loads codebase
spec_result = create_specification_with_gemini({
  feature_description: "2FA authentication"
})
# Returns: { ..., "context_id": "ctx_abc123" }

# Second call - reuses cached context (faster!)
review_result = review_code_with_gemini({
  files: ["auth.py", "middleware.py"],
  context_id: "ctx_abc123"  # Skip reload
})

Result: Fast subsequent calls that share the same codebase understanding!
```

**Manual Control** (Advanced - for custom analysis):
```
# Step 1: Explicit analysis with custom patterns
analysis = analyze_codebase_with_gemini({
  focus_description: "authentication and security patterns",
  file_patterns: ["*.py", "*.ts"],
  directories: ["src/auth", "src/middleware"]
})

# Step 2: Use the context_id for targeted operations
spec = create_specification_with_gemini({
  feature_description: "2FA with TOTP",
  context_id: analysis.cached_context_id
})
```

---

### Example 1: Analyze Codebase

Claude Code can use this to analyze your codebase before implementing a feature:

```
User: "I want to add Redis caching to the product catalog API"

Claude Code (internally):
[Calls: analyze_codebase_with_gemini({
  focus_description: "product catalog API structure and caching opportunities",
  file_patterns: ["*.py", "*.js"],
  directories: ["src/api", "src/services"]
})]

Response:
{
  "analysis": "The product catalog API is implemented in...",
  "architecture_summary": "Microservices architecture with...",
  "relevant_files": ["src/api/catalog.py", "src/services/product_service.py"],
  "cached_context_id": "ctx_abc123"
}
```

### Example 2: Generate Specification

```
User: "Create a detailed spec for the Redis caching feature"

Claude Code (internally):
[Calls: create_specification_with_gemini({
  feature_description: "Redis caching for product catalog API",
  context_id: "ctx_abc123",  // Reuse cached analysis
  spec_template: "standard"
})]

Response:
{
  "spec_path": "./specs/redis-caching-for-product-catalog-api-spec.md",
  "implementation_tasks": [
    {"task": "Install redis-py dependency", "order": 1},
    {"task": "Create cache middleware", "order": 2},
    ...
  ],
  "estimated_complexity": "medium"
}
```

### Example 3: Code Review

```
User: "Review my changes before I commit"

Claude Code (internally):
[Calls: review_code_with_gemini({
  review_focus: ["security", "performance"],
  spec_path: "./specs/redis-caching-spec.md"
})]

Response:
{
  "review_path": "./reviews/2025-01-10-123456-review.md",
  "has_blocking_issues": false,
  "summary": "Code looks good overall. Consider adding connection pooling."
}
```

### Example 4: Generate Documentation

```
User: "Generate API documentation for the catalog service"

Claude Code (internally):
[Calls: generate_documentation_with_gemini({
  documentation_type: "api",
  scope: "product catalog service",
  include_examples: true
})]

Response:
{
  "doc_path": "./docs/api-documentation.md",
  "word_count": 2500
}
```

### Example 5: Ask Gemini

```
User: "Ask Gemini about the best caching strategy for this codebase"

Claude Code (internally):
[Calls: ask_gemini({
  prompt: "What's the best caching strategy for this product catalog API?",
  include_codebase_context: true,
  temperature: 0.7
})]

Response:
{
  "response": "Based on your codebase architecture, I recommend...",
  "context_used": true
}
```

## Tool Reference

### analyze_codebase_with_gemini

Analyze codebase using Gemini's 2M token context window.

**Parameters:**
- `focus_description` (string, required): What to focus on in the analysis
- `directories` (array, optional): Directories to analyze
- `file_patterns` (array, optional): File patterns to include (default: `["*.py", "*.js", "*.ts", "*.java", "*.go"]`)
- `exclude_patterns` (array, optional): Patterns to exclude (default: `["node_modules/", "dist/", "build/"]`)

**Returns:**
```json
{
  "analysis": "Detailed analysis text",
  "architecture_summary": "High-level overview",
  "relevant_files": ["file1.py", "file2.js"],
  "patterns_identified": ["pattern1", "pattern2"],
  "integration_points": ["point1", "point2"],
  "cached_context_id": "ctx_abc123"
}
```

### create_specification_with_gemini

Generate detailed technical specification with automatic codebase loading.

**Parameters:**
- `feature_description` (string, required): What feature to specify
- `context_id` (string, optional): Context ID from previous analysis. **If not provided, automatically loads codebase.**
- `spec_template` (string, optional): Template to use ("standard", "minimal", "comprehensive")
- `output_path` (string, optional): Where to save the spec

**Returns:**
```json
{
  "spec_path": "./specs/feature-spec.md",
  "spec_content": "Full markdown content",
  "implementation_tasks": [{"task": "...", "order": 1}],
  "estimated_complexity": "medium",
  "files_to_modify": ["file1.py"],
  "files_to_create": ["file2.py"],
  "context_id": "ctx_abc123"
}
```

*Note: The `context_id` can be used for subsequent tool calls to skip reloading.*

### review_code_with_gemini

Comprehensive code review with automatic codebase loading.

**Parameters:**
- `files` (array, optional): Files to review (default: git diff)
- `review_focus` (array, optional): Focus areas (default: `["security", "performance", "best-practices", "testing"]`)
- `spec_path` (string, optional): Specification to review against
- `output_path` (string, optional): Where to save review
- `context_id` (string, optional): Context ID from previous analysis. **If not provided, automatically loads codebase.**

**Returns:**
```json
{
  "review_path": "./reviews/2025-01-10-review.md",
  "review_content": "Full markdown review",
  "issues_found": [{
    "severity": "high",
    "category": "security",
    "file": "auth.py",
    "line": 42,
    "issue": "Potential SQL injection",
    "suggestion": "Use parameterized queries"
  }],
  "has_blocking_issues": true,
  "summary": "Review summary",
  "recommendations": ["Add input validation", "Use ORM"],
  "context_id": "ctx_abc123"
}
```

*Note: The `context_id` can be used for subsequent tool calls to skip reloading.*

### generate_documentation_with_gemini

Generate comprehensive documentation with automatic codebase loading.

**Parameters:**
- `documentation_type` (string, required): Type ("api", "architecture", "user-guide", "readme", "contributing")
- `scope` (string, required): What to document
- `output_path` (string, optional): Where to save documentation
- `include_examples` (boolean, optional): Include code examples (default: true)
- `context_id` (string, optional): Context ID from previous analysis. **If not provided, automatically loads codebase.**

**Returns:**
```json
{
  "doc_path": "./docs/api-documentation.md",
  "doc_content": "Full markdown documentation",
  "sections": ["overview", "endpoints", "examples"],
  "word_count": 2500,
  "context_id": "ctx_abc123"
}
```

*Note: The `context_id` can be used for subsequent tool calls to skip reloading.*

### ask_gemini

General-purpose Gemini query with optional codebase context.

**Parameters:**
- `prompt` (string, required): Question or task
- `include_codebase_context` (boolean, optional): Load full codebase (default: false). **If true and no context_id, automatically loads codebase.**
- `context_id` (string, optional): Reuse cached context
- `temperature` (number, optional): Generation temperature 0.0-1.0 (default: 0.7)

**Returns:**
```json
{
  "response": "Gemini's response",
  "context_used": true,
  "token_count": 150000,
  "context_id": "ctx_abc123"
}
```

*Note: `context_id` is included when context is used and can be reused for subsequent calls.*

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code CLI                     │
│  (Orchestrator - makes all decisions)                │
│                                                      │
│  "Let me analyze the codebase with Gemini..."       │
│  [Invokes: analyze_codebase_with_gemini]            │
└──────────────────────┬──────────────────────────────┘
                       │ MCP Protocol
                       ↓
┌─────────────────────────────────────────────────────┐
│         MCP Server: gemini-workflow-bridge          │
│                                                      │
│  Tools:                                             │
│  • analyze_codebase_with_gemini                     │
│  • create_specification_with_gemini                 │
│  • review_code_with_gemini                          │
│  • generate_documentation_with_gemini               │
│  • ask_gemini                                       │
│                                                      │
│  Resources:                                         │
│  • workflow://specs/{feature-name}                  │
│  • workflow://reviews/{review-id}                   │
│  • workflow://context/{project-name}                │
└──────────────────────┬──────────────────────────────┘
                       │ Gemini API
                       ↓
┌─────────────────────────────────────────────────────┐
│              Google Gemini 2.0 Flash                │
│         (2M token context, fast, cost-effective)    │
└─────────────────────────────────────────────────────┘
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/hitoshura25/gemini-workflow-bridge-mcp
cd gemini-workflow-bridge-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Copy environment template (optional)
cp .env.example .env
# Edit .env to customize model or directories if needed
```

### Run Tests

```bash
pytest
```

### Run Linting

```bash
ruff check .
mypy .
```

## License

Apache-2.0 License - see [LICENSE](https://github.com/hitoshura25/gemini-workflow-bridge-mcp/blob/main/LICENSE) for details.

## Credits

- Built with [MCP](https://modelcontextprotocol.io/)
- Powered by [Google Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/)
- Generated with [mcp-server-generator](https://github.com/hitoshura25/mcp-server-generator)

## Support
- **Issues**: [GitHub Issues](https://github.com/hitoshura25/gemini-workflow-bridge-mcp/issues)
