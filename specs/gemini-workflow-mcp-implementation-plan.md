# Gemini Workflow MCP Server - Implementation Plan

## Executive Summary

**Goal:** Create an MCP server that allows Claude Code to delegate specific workflow tasks to Gemini CLI, leveraging Gemini's massive context window and cost-effectiveness for read-heavy operations.

**Feasibility:** ✅ **HIGHLY FEASIBLE** - This is exactly what MCP is designed for.

**Architecture:** Claude Code acts as the orchestrator, invoking MCP tools that bridge to Gemini CLI for specific tasks like codebase analysis, specification creation, and code review.

---

## Why This Approach Works

### The Key Insight
Instead of trying to orchestrate Claude Code from outside, we **extend Claude Code's capabilities** by giving it access to Gemini through MCP tools. Claude Code remains in control and decides when to use Gemini.

### Architecture Overview

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
│  • ask_gemini (general purpose)                     │
│                                                      │
│  Resources:                                         │
│  • workflow://specs/{feature-name}                  │
│  • workflow://reviews/{feature-name}                │
│  • workflow://context/{project-name}                │
└──────────────────────┬──────────────────────────────┘
                       │ Gemini API / CLI
                       ↓
┌─────────────────────────────────────────────────────┐
│              Google Gemini 2.0 Flash                │
│         (2M token context, fast, cost-effective)    │
└─────────────────────────────────────────────────────┘
```

### Benefits

1. **Claude Code stays in control** - It decides when to use Gemini
2. **Leverages both tools' strengths** - Gemini for heavy reads, Claude Code for implementation
3. **Seamless integration** - Just install the MCP server, and Claude Code gets new capabilities
4. **State management** - MCP server manages workflow artifacts (specs, reviews, etc.)
5. **Cost optimization** - Expensive analysis goes to Gemini, precise edits stay with Claude

---

## MCP Server Specification

### Project Details

**Name:** `gemini-workflow-bridge`
**Package:** `hitoshura25-gemini-workflow-bridge`
**Language:** Python (for easy Gemini API integration)
**Transport:** stdio (standard MCP)

### Tools to Implement

#### 1. `analyze_codebase_with_gemini`

**Purpose:** Load extensive codebase context using Gemini's 2M token window

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "focus_description": {
      "type": "string",
      "description": "What to focus on in the analysis (e.g., 'authentication system', 'API structure')"
    },
    "directories": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Directories to analyze (defaults to current project)"
    },
    "file_patterns": {
      "type": "array",
      "items": {"type": "string"},
      "description": "File patterns to include (e.g., ['*.py', '*.js'])"
    },
    "exclude_patterns": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Patterns to exclude (e.g., ['node_modules/', 'dist/'])"
    }
  },
  "required": ["focus_description"]
}
```

**Output:**
```json
{
  "analysis": "Detailed codebase analysis...",
  "architecture_summary": "High-level architecture...",
  "relevant_files": ["file1.py", "file2.js"],
  "patterns_identified": ["pattern1", "pattern2"],
  "integration_points": ["point1", "point2"],
  "cached_context_id": "ctx_abc123"
}
```

**Implementation Notes:**
- Loads files matching patterns
- Sends to Gemini with structured prompt
- Caches context for follow-up queries
- Returns structured analysis

---

#### 2. `create_specification_with_gemini`

**Purpose:** Generate detailed technical specifications using full codebase context

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "feature_description": {
      "type": "string",
      "description": "What feature to specify"
    },
    "context_id": {
      "type": "string",
      "description": "Optional: ID from previous analyze_codebase call to reuse context"
    },
    "spec_template": {
      "type": "string",
      "enum": ["standard", "minimal", "comprehensive"],
      "description": "Specification template to use"
    },
    "output_path": {
      "type": "string",
      "description": "Where to save the spec (defaults to ./specs/{feature-name}-spec.md)"
    }
  },
  "required": ["feature_description"]
}
```

**Output:**
```json
{
  "spec_path": "./specs/feature-name-spec.md",
  "spec_content": "Full specification markdown...",
  "implementation_tasks": [
    {"task": "Install dependencies", "order": 1},
    {"task": "Create auth middleware", "order": 2}
  ],
  "estimated_complexity": "medium",
  "files_to_modify": ["file1.py", "file2.js"],
  "files_to_create": ["new_file.py"]
}
```

**Implementation Notes:**
- Uses context from analyze_codebase if available
- Follows spec template
- Saves spec to file system
- Returns structured data for Claude Code to use

---

#### 3. `review_code_with_gemini`

**Purpose:** Comprehensive code review using Gemini's large context window

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "files": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Files to review (if empty, reviews git diff)"
    },
    "review_focus": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["security", "performance", "best-practices", "testing", "documentation"]
      },
      "description": "Areas to focus on in review"
    },
    "spec_path": {
      "type": "string",
      "description": "Optional: Path to spec to review against"
    },
    "output_path": {
      "type": "string",
      "description": "Where to save review (defaults to ./reviews/{timestamp}-review.md)"
    }
  }
}
```

**Output:**
```json
{
  "review_path": "./reviews/2025-01-10-review.md",
  "review_content": "Full review markdown...",
  "issues_found": [
    {
      "severity": "high",
      "category": "security",
      "file": "auth.py",
      "line": 42,
      "issue": "Potential SQL injection",
      "suggestion": "Use parameterized queries"
    }
  ],
  "has_blocking_issues": true,
  "summary": "Review summary..."
}
```

**Implementation Notes:**
- Loads all specified files or git diff
- Loads spec if provided for compliance check
- Sends to Gemini with structured review prompt
- Returns actionable feedback

---

#### 4. `generate_documentation_with_gemini`

**Purpose:** Generate comprehensive documentation with full codebase context

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "documentation_type": {
      "type": "string",
      "enum": ["api", "architecture", "user-guide", "readme", "contributing"],
      "description": "Type of documentation to generate"
    },
    "scope": {
      "type": "string",
      "description": "What to document (e.g., 'authentication system', 'entire API')"
    },
    "output_path": {
      "type": "string",
      "description": "Where to save documentation"
    },
    "include_examples": {
      "type": "boolean",
      "description": "Whether to include code examples",
      "default": true
    }
  },
  "required": ["documentation_type", "scope"]
}
```

**Output:**
```json
{
  "doc_path": "./docs/api-documentation.md",
  "doc_content": "Full documentation markdown...",
  "sections": ["overview", "endpoints", "examples", "troubleshooting"],
  "word_count": 2500
}
```

---

#### 5. `ask_gemini`

**Purpose:** General-purpose Gemini query with optional codebase context

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt": {
      "type": "string",
      "description": "Question or task for Gemini"
    },
    "include_codebase_context": {
      "type": "boolean",
      "description": "Whether to load full codebase context",
      "default": false
    },
    "context_id": {
      "type": "string",
      "description": "Optional: Reuse cached context"
    },
    "temperature": {
      "type": "number",
      "description": "Temperature for generation (0.0-1.0)",
      "default": 0.7
    }
  },
  "required": ["prompt"]
}
```

**Output:**
```json
{
  "response": "Gemini's response...",
  "context_used": true,
  "token_count": 150000
}
```

---

### Resources to Implement

MCP resources allow Claude Code to read workflow artifacts.

#### 1. `workflow://specs/{feature-name}`

**Purpose:** Access saved specifications

**Example URI:** `workflow://specs/jwt-authentication`

**Returns:**
```json
{
  "uri": "workflow://specs/jwt-authentication",
  "mimeType": "text/markdown",
  "text": "# JWT Authentication Specification\n..."
}
```

---

#### 2. `workflow://reviews/{review-id}`

**Purpose:** Access saved code reviews

**Example URI:** `workflow://reviews/2025-01-10-review`

**Returns:**
```json
{
  "uri": "workflow://reviews/2025-01-10-review",
  "mimeType": "text/markdown",
  "text": "# Code Review\n..."
}
```

---

#### 3. `workflow://context/{project-name}`

**Purpose:** Access cached codebase analysis

**Example URI:** `workflow://context/my-api-project`

**Returns:**
```json
{
  "uri": "workflow://context/my-api-project",
  "mimeType": "application/json",
  "text": "{\"analysis\": \"...\", \"architecture\": \"...\"}"
}
```

---

## Implementation Plan

### Phase 1: Project Setup with mcp-server-generator

#### Step 1.1: Use mcp-server-generator to scaffold project

**Action:** Invoke the mcp-server-generator MCP tool

```python
# Tool call to mcp-server-generator
{
  "project_name": "gemini-workflow-bridge",
  "description": "MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review",
  "author": "Your Name",
  "author_email": "your.email@example.com",
  "tools": [
    {
      "name": "analyze_codebase_with_gemini",
      "description": "Analyze codebase using Gemini's 2M token context window",
      "parameters": [
        {
          "name": "focus_description",
          "type": "string",
          "description": "What to focus on in the analysis",
          "required": true
        },
        {
          "name": "directories",
          "type": "array",
          "description": "Directories to analyze",
          "required": false
        },
        {
          "name": "file_patterns",
          "type": "array",
          "description": "File patterns to include",
          "required": false
        },
        {
          "name": "exclude_patterns",
          "type": "array",
          "description": "Patterns to exclude",
          "required": false
        }
      ]
    },
    {
      "name": "create_specification_with_gemini",
      "description": "Generate detailed technical specification using full codebase context",
      "parameters": [
        {
          "name": "feature_description",
          "type": "string",
          "description": "What feature to specify",
          "required": true
        },
        {
          "name": "context_id",
          "type": "string",
          "description": "Optional context ID from previous analysis",
          "required": false
        },
        {
          "name": "spec_template",
          "type": "string",
          "description": "Specification template to use",
          "required": false
        },
        {
          "name": "output_path",
          "type": "string",
          "description": "Where to save the spec",
          "required": false
        }
      ]
    },
    {
      "name": "review_code_with_gemini",
      "description": "Comprehensive code review using Gemini",
      "parameters": [
        {
          "name": "files",
          "type": "array",
          "description": "Files to review",
          "required": false
        },
        {
          "name": "review_focus",
          "type": "array",
          "description": "Areas to focus on",
          "required": false
        },
        {
          "name": "spec_path",
          "type": "string",
          "description": "Path to spec to review against",
          "required": false
        },
        {
          "name": "output_path",
          "type": "string",
          "description": "Where to save review",
          "required": false
        }
      ]
    },
    {
      "name": "generate_documentation_with_gemini",
      "description": "Generate comprehensive documentation with full codebase context",
      "parameters": [
        {
          "name": "documentation_type",
          "type": "string",
          "description": "Type of documentation",
          "required": true
        },
        {
          "name": "scope",
          "type": "string",
          "description": "What to document",
          "required": true
        },
        {
          "name": "output_path",
          "type": "string",
          "description": "Where to save documentation",
          "required": false
        },
        {
          "name": "include_examples",
          "type": "boolean",
          "description": "Include code examples",
          "required": false
        }
      ]
    },
    {
      "name": "ask_gemini",
      "description": "General-purpose Gemini query with optional codebase context",
      "parameters": [
        {
          "name": "prompt",
          "type": "string",
          "description": "Question or task for Gemini",
          "required": true
        },
        {
          "name": "include_codebase_context",
          "type": "boolean",
          "description": "Load full codebase context",
          "required": false
        },
        {
          "name": "context_id",
          "type": "string",
          "description": "Reuse cached context",
          "required": false
        },
        {
          "name": "temperature",
          "type": "number",
          "description": "Temperature for generation",
          "required": false
        }
      ]
    }
  ],
  "output_dir": "./gemini-workflow-bridge",
  "python_version": "3.11"
}
```

**Expected Output:**
- Project scaffolded with:
  - `src/gemini_workflow_bridge/server.py`
  - `src/gemini_workflow_bridge/tools/` (tool implementations)
  - `pyproject.toml`
  - `README.md`
  - `.github/workflows/` (CI/CD)
  - `tests/`

---

#### Step 1.2: Review and customize generated project

**Files to customize:**

1. **`pyproject.toml`** - Add Gemini dependencies
   ```toml
   dependencies = [
       "mcp>=1.1.2",
       "google-generativeai>=0.8.0",  # Add this
       "python-dotenv>=1.0.0",         # Add this
       "gitpython>=3.1.0",             # For git operations
       "pathspec>=0.12.0",             # For .gitignore support
   ]
   ```

2. **`.env.example`** - Add Gemini API key
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   DEFAULT_SPEC_DIR=./specs
   DEFAULT_REVIEW_DIR=./reviews
   DEFAULT_CONTEXT_DIR=./.workflow-context
   ```

---

### Phase 2: Core Implementation

#### Step 2.1: Implement Gemini Client

**File:** `src/gemini_workflow_bridge/gemini_client.py`

```python
"""Gemini API client wrapper"""
import os
from typing import Optional, Dict, List, Any
import google.generativeai as genai
from pathlib import Path
import json

class GeminiClient:
    """Wrapper for Gemini API with caching and context management"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.context_cache: Dict[str, Any] = {}

    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate content with Gemini"""
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        response = await self.model.generate_content_async(
            prompt,
            generation_config=generation_config
        )

        return response.text

    async def analyze_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """Generate content with provided context"""
        full_prompt = f"""Context:
{context}

Task:
{prompt}

Please provide a detailed, structured response."""

        return await self.generate_content(full_prompt, temperature)

    def cache_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Cache context for reuse"""
        self.context_cache[context_id] = context

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context"""
        return self.context_cache.get(context_id)
```

---

#### Step 2.2: Implement Codebase Loader

**File:** `src/gemini_workflow_bridge/codebase_loader.py`

```python
"""Codebase loading utilities"""
from pathlib import Path
from typing import List, Optional, Dict
import pathspec
import git

class CodebaseLoader:
    """Load and prepare codebase for analysis"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        """Load .gitignore patterns"""
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                patterns = f.read().splitlines()
            return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        return None

    def load_files(
        self,
        file_patterns: List[str] = ["*.py", "*.js", "*.ts", "*.java"],
        exclude_patterns: List[str] = [],
        directories: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Load files matching patterns"""
        files_content = {}

        search_dirs = [Path(d) for d in directories] if directories else [self.root_dir]

        for search_dir in search_dirs:
            for pattern in file_patterns:
                for file_path in search_dir.rglob(pattern):
                    # Skip if matches exclude patterns
                    if self._should_exclude(file_path, exclude_patterns):
                        continue

                    # Skip if in .gitignore
                    if self.gitignore_spec and self.gitignore_spec.match_file(
                        str(file_path.relative_to(self.root_dir))
                    ):
                        continue

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            relative_path = file_path.relative_to(self.root_dir)
                            files_content[str(relative_path)] = f.read()
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary or inaccessible files
                        continue

        return files_content

    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded"""
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return True
        return False

    def get_project_structure(self) -> str:
        """Get ASCII tree of project structure"""
        try:
            repo = git.Repo(self.root_dir)
            files = repo.git.ls_files().split('\n')
            return self._build_tree(files)
        except git.InvalidGitRepositoryError:
            # Not a git repo, fall back to directory listing
            return self._build_tree_from_dir()

    def _build_tree(self, files: List[str]) -> str:
        """Build ASCII tree from file list"""
        tree = {}
        for file in files:
            parts = Path(file).parts
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return self._format_tree(tree)

    def _format_tree(self, tree: dict, prefix: str = "") -> str:
        """Format tree as ASCII"""
        lines = []
        items = sorted(tree.items())
        for i, (name, subtree) in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            lines.append(f"{prefix}{current_prefix}{name}")

            if subtree:
                extension_prefix = "    " if is_last else "│   "
                lines.append(self._format_tree(subtree, prefix + extension_prefix))

        return "\n".join(lines)

    def _build_tree_from_dir(self) -> str:
        """Build tree from directory (fallback)"""
        # Simplified implementation
        return str(list(self.root_dir.rglob("*")))
```

---

#### Step 2.3: Implement analyze_codebase_with_gemini Tool

**File:** `src/gemini_workflow_bridge/tools/analyze_codebase.py`

```python
"""Codebase analysis tool"""
from typing import Any, Dict
from ..gemini_client import GeminiClient
from ..codebase_loader import CodebaseLoader
import hashlib
import json

async def analyze_codebase_with_gemini(
    focus_description: str,
    directories: list[str] | None = None,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    gemini_client: GeminiClient = None,
    codebase_loader: CodebaseLoader = None
) -> Dict[str, Any]:
    """Analyze codebase using Gemini's large context window"""

    # Initialize clients if not provided
    if gemini_client is None:
        gemini_client = GeminiClient()
    if codebase_loader is None:
        codebase_loader = CodebaseLoader()

    # Load codebase
    file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.java", "*.go"]
    exclude_patterns = exclude_patterns or ["node_modules/", "dist/", "build/", "__pycache__/"]

    files_content = codebase_loader.load_files(
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns,
        directories=directories
    )

    # Get project structure
    project_structure = codebase_loader.get_project_structure()

    # Build analysis prompt
    context = _build_codebase_context(files_content, project_structure)

    prompt = f"""Analyze this codebase with focus on: {focus_description}

Please provide:
1. **Architecture Summary**: High-level overview of the system architecture
2. **Relevant Files**: Files most relevant to the focus area
3. **Patterns Identified**: Common patterns, conventions, and practices used
4. **Integration Points**: Where new code would integrate with existing code
5. **Dependencies**: Key dependencies and libraries in use
6. **Recommendations**: Suggestions for the focus area

Format your response as structured JSON with these sections."""

    # Send to Gemini
    response = await gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=0.7
    )

    # Parse response
    try:
        analysis = json.loads(response)
    except json.JSONDecodeError:
        # If not JSON, structure it ourselves
        analysis = {
            "analysis": response,
            "architecture_summary": "",
            "relevant_files": [],
            "patterns_identified": [],
            "integration_points": []
        }

    # Generate context ID for caching
    context_id = _generate_context_id(focus_description, files_content)

    # Cache the context
    gemini_client.cache_context(context_id, {
        "files_content": files_content,
        "project_structure": project_structure,
        "analysis": analysis
    })

    return {
        "analysis": analysis.get("analysis", response),
        "architecture_summary": analysis.get("architecture_summary", ""),
        "relevant_files": analysis.get("relevant_files", []),
        "patterns_identified": analysis.get("patterns_identified", []),
        "integration_points": analysis.get("integration_points", []),
        "cached_context_id": context_id
    }

def _build_codebase_context(files_content: Dict[str, str], project_structure: str) -> str:
    """Build context string from codebase"""
    context_parts = [
        "# Project Structure",
        project_structure,
        "",
        "# File Contents",
        ""
    ]

    for file_path, content in files_content.items():
        context_parts.append(f"## File: {file_path}")
        context_parts.append("```")
        context_parts.append(content)
        context_parts.append("```")
        context_parts.append("")

    return "\n".join(context_parts)

def _generate_context_id(focus: str, files: Dict[str, str]) -> str:
    """Generate unique ID for this context"""
    content_hash = hashlib.sha256(
        json.dumps(sorted(files.keys())).encode()
    ).hexdigest()[:8]

    focus_hash = hashlib.sha256(focus.encode()).hexdigest()[:8]

    return f"ctx_{focus_hash}_{content_hash}"
```

---

#### Step 2.4: Implement create_specification_with_gemini Tool

**File:** `src/gemini_workflow_bridge/tools/create_specification.py`

```python
"""Specification creation tool"""
from typing import Any, Dict, Optional
from ..gemini_client import GeminiClient
from pathlib import Path
import os

SPEC_TEMPLATES = {
    "standard": """# {feature_name} - Technical Specification

## 1. Feature Overview
[2-3 sentence summary]

## 2. Architecture Overview
[High-level design and data flow]

## 3. Database Changes
### Schema Modifications
### Migration Scripts Needed

## 4. Files to Create
[List with purpose]

## 5. Files to Modify
[List with specific changes needed]

## 6. Implementation Tasks (Ordered)
1. [Task 1]
2. [Task 2]
...

## 7. Testing Requirements
### Unit Tests
### Integration Tests
### E2E Tests

## 8. Security Considerations
[Security concerns and mitigations]

## 9. Performance Considerations
[Performance implications]

## 10. Dependencies
[New packages/libraries needed]
""",
    "minimal": """# {feature_name}

## What
[Description]

## How
[Implementation approach]

## Tasks
- [ ] Task 1
- [ ] Task 2

## Files
- Create: [files]
- Modify: [files]
""",
    "comprehensive": """[Extended version with additional sections for API docs, examples, rollback, monitoring, etc.]"""
}

async def create_specification_with_gemini(
    feature_description: str,
    context_id: Optional[str] = None,
    spec_template: str = "standard",
    output_path: Optional[str] = None,
    gemini_client: GeminiClient = None
) -> Dict[str, Any]:
    """Generate detailed technical specification using Gemini"""

    if gemini_client is None:
        gemini_client = GeminiClient()

    # Get cached context if available
    context = ""
    if context_id:
        cached = gemini_client.get_cached_context(context_id)
        if cached:
            context = _format_cached_context(cached)

    # Get template
    template = SPEC_TEMPLATES.get(spec_template, SPEC_TEMPLATES["standard"])

    # Build prompt
    prompt = f"""Create a detailed technical specification for the following feature:

Feature: {feature_description}

Use this template structure:
{template}

{"Based on the codebase context provided, " if context else ""}ensure the specification:
1. Aligns with existing architecture and patterns
2. Lists specific files to create/modify
3. Provides ordered implementation tasks
4. Includes comprehensive testing strategy
5. Addresses security and performance
6. Lists all dependencies

Provide the complete specification in markdown format."""

    # Generate specification
    if context:
        spec_content = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.7
        )
    else:
        spec_content = await gemini_client.generate_content(
            prompt=prompt,
            temperature=0.7
        )

    # Determine output path
    if not output_path:
        feature_slug = feature_description.lower().replace(' ', '-')[:50]
        specs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./specs"))
        specs_dir.mkdir(exist_ok=True)
        output_path = str(specs_dir / f"{feature_slug}-spec.md")

    # Save specification
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(spec_content)

    # Extract tasks (parse markdown checkboxes)
    tasks = _extract_tasks(spec_content)

    # Extract file lists
    files_to_modify, files_to_create = _extract_file_lists(spec_content)

    return {
        "spec_path": str(output_path),
        "spec_content": spec_content,
        "implementation_tasks": tasks,
        "estimated_complexity": _estimate_complexity(spec_content),
        "files_to_modify": files_to_modify,
        "files_to_create": files_to_create
    }

def _format_cached_context(cached: Dict[str, Any]) -> str:
    """Format cached context for prompt"""
    parts = [
        "# Previous Codebase Analysis",
        "",
        "## Architecture Summary",
        cached.get("analysis", {}).get("architecture_summary", ""),
        "",
        "## Relevant Files",
        "\n".join(f"- {f}" for f in cached.get("analysis", {}).get("relevant_files", [])),
        "",
        "## Patterns Identified",
        "\n".join(f"- {p}" for p in cached.get("analysis", {}).get("patterns_identified", [])),
    ]
    return "\n".join(parts)

def _extract_tasks(spec_content: str) -> list[Dict[str, Any]]:
    """Extract implementation tasks from spec"""
    tasks = []
    in_tasks_section = False
    order = 1

    for line in spec_content.split('\n'):
        if "implementation task" in line.lower() or "tasks" in line.lower():
            in_tasks_section = True
            continue

        if in_tasks_section:
            if line.strip().startswith(('#', '##', '###')):
                break

            if line.strip().startswith(('- ', '* ', str(order))):
                task_text = line.strip().lstrip('-*0123456789. []')
                if task_text:
                    tasks.append({"task": task_text, "order": order})
                    order += 1

    return tasks

def _extract_file_lists(spec_content: str) -> tuple[list[str], list[str]]:
    """Extract file modification and creation lists"""
    to_modify = []
    to_create = []

    current_section = None
    for line in spec_content.split('\n'):
        line_lower = line.lower()

        if "files to modify" in line_lower:
            current_section = "modify"
        elif "files to create" in line_lower:
            current_section = "create"
        elif line.strip().startswith('#'):
            current_section = None

        if current_section and line.strip().startswith(('- ', '* ')):
            file_path = line.strip().lstrip('-* ').split(':')[0].strip()
            if file_path:
                if current_section == "modify":
                    to_modify.append(file_path)
                else:
                    to_create.append(file_path)

    return to_modify, to_create

def _estimate_complexity(spec_content: str) -> str:
    """Estimate implementation complexity"""
    # Simple heuristic based on content length and task count
    task_count = len(_extract_tasks(spec_content))
    content_length = len(spec_content)

    if task_count > 15 or content_length > 10000:
        return "high"
    elif task_count > 7 or content_length > 5000:
        return "medium"
    else:
        return "low"
```

---

#### Step 2.5: Implement review_code_with_gemini Tool

**File:** `src/gemini_workflow_bridge/tools/review_code.py`

```python
"""Code review tool"""
from typing import Any, Dict, Optional, List
from ..gemini_client import GeminiClient
from pathlib import Path
import subprocess
import os
import json

async def review_code_with_gemini(
    files: Optional[List[str]] = None,
    review_focus: Optional[List[str]] = None,
    spec_path: Optional[str] = None,
    output_path: Optional[str] = None,
    gemini_client: GeminiClient = None
) -> Dict[str, Any]:
    """Comprehensive code review using Gemini"""

    if gemini_client is None:
        gemini_client = GeminiClient()

    # Get files to review
    if not files:
        # Use git diff if no files specified
        files_content = _get_git_diff()
    else:
        files_content = _load_files(files)

    # Load spec if provided
    spec_content = ""
    if spec_path and Path(spec_path).exists():
        spec_content = Path(spec_path).read_text()

    # Default review focus
    if not review_focus:
        review_focus = ["security", "performance", "best-practices", "testing"]

    # Build review prompt
    focus_text = ", ".join(review_focus)

    prompt = f"""Conduct a comprehensive code review focusing on: {focus_text}

{"Specification to review against:" if spec_content else ""}
{spec_content if spec_content else ""}

Code to review:
{files_content}

Please provide:
1. **Issues Found**: List all issues with severity (high/medium/low), category, file, line number, description, and suggested fix
2. **Summary**: Overall assessment
3. **Blocking Issues**: Whether there are issues that must be fixed before merging
4. **Recommendations**: General recommendations for improvement

Format your response as JSON with this structure:
{{
  "issues_found": [
    {{
      "severity": "high|medium|low",
      "category": "security|performance|best-practices|testing|documentation",
      "file": "path/to/file.py",
      "line": 42,
      "issue": "Description of the issue",
      "suggestion": "How to fix it"
    }}
  ],
  "summary": "Overall assessment...",
  "has_blocking_issues": true|false,
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}}
"""

    # Generate review
    response = await gemini_client.generate_content(
        prompt=prompt,
        temperature=0.3  # Lower temperature for more consistent reviews
    )

    # Parse response
    try:
        review_data = json.loads(response)
    except json.JSONDecodeError:
        # If not JSON, create structured output
        review_data = {
            "issues_found": [],
            "summary": response,
            "has_blocking_issues": False,
            "recommendations": []
        }

    # Format as markdown
    review_markdown = _format_review_markdown(review_data, files_content)

    # Determine output path
    if not output_path:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        reviews_dir = Path(os.getenv("DEFAULT_REVIEW_DIR", "./reviews"))
        reviews_dir.mkdir(exist_ok=True)
        output_path = str(reviews_dir / f"{timestamp}-review.md")

    # Save review
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(review_markdown)

    return {
        "review_path": str(output_path),
        "review_content": review_markdown,
        "issues_found": review_data.get("issues_found", []),
        "has_blocking_issues": review_data.get("has_blocking_issues", False),
        "summary": review_data.get("summary", "")
    }

def _get_git_diff() -> str:
    """Get git diff for review"""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return "No git changes found"

def _load_files(file_paths: List[str]) -> str:
    """Load specified files"""
    content_parts = []

    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            content_parts.append(f"## File: {file_path}")
            content_parts.append("```")
            content_parts.append(path.read_text())
            content_parts.append("```")
            content_parts.append("")

    return "\n".join(content_parts)

def _format_review_markdown(review_data: Dict[str, Any], code: str) -> str:
    """Format review as markdown"""
    lines = [
        "# Code Review",
        "",
        f"**Review Date:** {_get_current_date()}",
        "",
        "## Summary",
        "",
        review_data.get("summary", ""),
        "",
        "## Issues Found",
        ""
    ]

    # Group issues by severity
    issues = review_data.get("issues_found", [])
    for severity in ["high", "medium", "low"]:
        severity_issues = [i for i in issues if i.get("severity") == severity]
        if severity_issues:
            lines.append(f"### {severity.upper()} Severity")
            lines.append("")

            for issue in severity_issues:
                lines.append(f"**{issue.get('file')}:{issue.get('line', '?')}** - {issue.get('category', 'general')}")
                lines.append(f"- **Issue:** {issue.get('issue', '')}")
                lines.append(f"- **Suggestion:** {issue.get('suggestion', '')}")
                lines.append("")

    # Blocking status
    if review_data.get("has_blocking_issues"):
        lines.append("## ⚠️ BLOCKING ISSUES FOUND")
        lines.append("This code has blocking issues that must be addressed before merging.")
        lines.append("")
    else:
        lines.append("## ✅ No Blocking Issues")
        lines.append("")

    # Recommendations
    if review_data.get("recommendations"):
        lines.append("## Recommendations")
        lines.append("")
        for rec in review_data["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)

def _get_current_date() -> str:
    """Get current date formatted"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

---

#### Step 2.6: Implement Remaining Tools

Similar implementations for:
- `generate_documentation_with_gemini` (follow pattern from above)
- `ask_gemini` (simpler, general-purpose query)

---

#### Step 2.7: Implement MCP Resources

**File:** `src/gemini_workflow_bridge/resources.py`

```python
"""MCP resource handlers for workflow artifacts"""
from pathlib import Path
from typing import Dict, Any
import os
import json

class WorkflowResources:
    """Manage workflow resources (specs, reviews, context)"""

    def __init__(self):
        self.specs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./specs"))
        self.reviews_dir = Path(os.getenv("DEFAULT_REVIEW_DIR", "./reviews"))
        self.context_dir = Path(os.getenv("DEFAULT_CONTEXT_DIR", "./.workflow-context"))

        # Ensure directories exist
        self.specs_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)

    def list_resources(self) -> list[str]:
        """List all available resources"""
        resources = []

        # Specs
        for spec_file in self.specs_dir.glob("*.md"):
            uri = f"workflow://specs/{spec_file.stem}"
            resources.append(uri)

        # Reviews
        for review_file in self.reviews_dir.glob("*.md"):
            uri = f"workflow://reviews/{review_file.stem}"
            resources.append(uri)

        # Cached contexts
        for context_file in self.context_dir.glob("*.json"):
            uri = f"workflow://context/{context_file.stem}"
            resources.append(uri)

        return resources

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource by URI"""
        if uri.startswith("workflow://specs/"):
            name = uri.replace("workflow://specs/", "")
            file_path = self.specs_dir / f"{name}.md"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": file_path.read_text()
                }

        elif uri.startswith("workflow://reviews/"):
            name = uri.replace("workflow://reviews/", "")
            file_path = self.reviews_dir / f"{name}.md"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": file_path.read_text()
                }

        elif uri.startswith("workflow://context/"):
            name = uri.replace("workflow://context/", "")
            file_path = self.context_dir / f"{name}.json"

            if file_path.exists():
                return {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": file_path.read_text()
                }

        raise ValueError(f"Resource not found: {uri}")
```

---

### Phase 3: Testing & Validation

#### Step 3.1: Write Unit Tests

**File:** `tests/test_gemini_client.py`

```python
"""Test Gemini client"""
import pytest
from gemini_workflow_bridge.gemini_client import GeminiClient

@pytest.mark.asyncio
async def test_generate_content():
    client = GeminiClient()
    response = await client.generate_content("What is 2+2?", temperature=0.0)
    assert "4" in response

# More tests...
```

#### Step 3.2: Integration Tests

Test full workflow:
1. Analyze codebase
2. Create spec
3. Review code
4. Verify artifacts

---

### Phase 4: Documentation & Deployment

#### Step 4.1: Update README

Add usage examples:

```markdown
## Installation

```bash
# Using uv
uvx hitoshura25-gemini-workflow-bridge

# Or install globally
pip install hitoshura25-gemini-workflow-bridge
```

## Configuration

Create `.env` file:
```
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.0-flash
```

## Usage with Claude Code

1. Configure in Claude Code:
```json
// ~/.claude.json
{
  "mcpServers": {
    "gemini-workflow": {
      "command": "uvx",
      "args": ["hitoshura25-gemini-workflow-bridge"],
      "env": {
        "GEMINI_API_KEY": "your_key"
      }
    }
  }
}
```

2. Use in Claude Code:
```
> I want to add a new authentication feature

[Claude Code internally]:
Let me analyze the codebase first...
[Calls: analyze_codebase_with_gemini]

Based on the analysis, let me create a specification...
[Calls: create_specification_with_gemini]

Now I'll implement according to the spec...
[Uses Claude Code's native tools]

Let me have Gemini review the changes...
[Calls: review_code_with_gemini]
```
```

#### Step 4.2: Publish to PyPI

Use the generated GitHub Actions workflow to publish.

---

## Usage Examples

### Example 1: Complete Feature Implementation

```bash
# User starts in Claude Code
> "I want to add Redis caching to the product catalog API"

# Claude Code (internally):
# 1. Analyze codebase with Gemini
[Invokes: analyze_codebase_with_gemini {
  focus_description: "product catalog API structure and caching opportunities"
}]

# Returns: Analysis with architecture, files, patterns

# 2. Create spec with Gemini
[Invokes: create_specification_with_gemini {
  feature_description: "Redis caching for product catalog",
  context_id: "ctx_abc123"
}]

# Returns: Detailed spec at ./specs/redis-caching-spec.md

# 3. Claude Code implements
# Uses native tools: Read, Edit, Write, Bash
# Follows spec step-by-step with task tracking

# 4. Run tests
# Uses native Bash tool

# 5. Code review with Gemini
[Invokes: review_code_with_gemini {
  spec_path: "./specs/redis-caching-spec.md"
}]

# Returns: Review with issues at ./reviews/2025-01-10-review.md

# 6. Address review issues
# Uses native Edit tool to fix issues

# 7. Create PR
# Uses native git tools
```

---

### Example 2: Just Codebase Analysis

```bash
# In Claude Code
> "Before I start, can you analyze our authentication system using Gemini?"

[Claude Code invokes: analyze_codebase_with_gemini {
  focus_description: "authentication system architecture and implementation",
  file_patterns: ["*.py", "*.js"],
  directories: ["src/auth", "src/middleware"]
}]

# Returns comprehensive analysis that Claude Code can use for follow-up work
```

---

### Example 3: Code Review Only

```bash
# After implementing locally
> "Can you have Gemini review my recent changes?"

[Claude Code invokes: review_code_with_gemini {
  review_focus: ["security", "performance"]
}]

# Returns detailed review
```

---

## Integration with mcp-server-generator

### Using mcp-server-generator for Initial Setup

1. **Invoke mcp-server-generator MCP tool** from Claude Code:
   ```
   > "Use the mcp-server-generator to create the gemini-workflow-bridge project with the tools I've specified"
   ```

2. **Claude Code will call the MCP tool** with the tool specifications from this document

3. **Generated structure:**
   ```
   gemini-workflow-bridge/
   ├── src/
   │   └── gemini_workflow_bridge/
   │       ├── __init__.py
   │       ├── server.py              # MCP server entry
   │       ├── gemini_client.py       # Add manually
   │       ├── codebase_loader.py     # Add manually
   │       ├── resources.py           # Add manually
   │       └── tools/
   │           ├── __init__.py
   │           ├── analyze_codebase.py
   │           ├── create_specification.py
   │           ├── review_code.py
   │           ├── generate_documentation.py
   │           └── ask_gemini.py
   ├── tests/
   ├── pyproject.toml
   ├── README.md
   ├── .env.example
   └── .github/
       └── workflows/
           ├── test.yml
           └── publish.yml
   ```

4. **Fill in implementation** using code from this document

5. **Use GitHub workflows** for CI/CD:
   - Test workflow runs on every push
   - Publish workflow publishes to PyPI on release

---

## Development Workflow

### Iterative Development with GitHub Actions

The mcp-server-generator creates workflows for:

1. **Testing** (`.github/workflows/test.yml`):
   - Runs on every push
   - Tests against multiple Python versions
   - Runs linting and type checking

2. **Publishing** (`.github/workflows/publish.yml`):
   - Triggered on version tags (v*)
   - Builds and publishes to PyPI
   - Creates GitHub release

### Local Development

```bash
# Clone generated repo
git clone https://github.com/yourusername/gemini-workflow-bridge
cd gemini-workflow-bridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Create .env file
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Run tests
pytest

# Run MCP server locally for testing
python -m gemini_workflow_bridge.server
```

---

## Deployment Strategy

### Phase 1: Alpha Testing (Week 1-2)
- Deploy to TestPyPI
- Test with small projects
- Gather feedback
- Fix critical bugs

### Phase 2: Beta Release (Week 3-4)
- Deploy to PyPI
- Documentation complete
- Example projects
- Community feedback

### Phase 3: Stable Release (Week 5+)
- Production-ready
- Performance optimization
- Advanced features
- Integration examples

---

## Success Metrics

### Technical Metrics
- ✅ All 5 tools implemented and working
- ✅ MCP resources accessible
- ✅ Test coverage > 80%
- ✅ Type hints complete
- ✅ Documentation complete

### Usage Metrics
- Token efficiency: Gemini handles 80%+ of context loading
- Speed: Analysis < 30s for medium codebases
- Quality: Specs have all required sections
- Accuracy: Reviews catch real issues

### Integration Metrics
- Works seamlessly with Claude Code
- No manual intervention needed
- Proper error handling
- Clear progress feedback

---

## Future Enhancements

### v2 Features
1. **Caching Optimization**
   - Persist context across sessions
   - Smart cache invalidation
   - Incremental updates

2. **Multi-Model Support**
   - Support different Gemini models
   - Model selection per task
   - Cost optimization

3. **Advanced Analysis**
   - Dependency graph analysis
   - Performance profiling integration
   - Security scan integration

4. **Workflow Automation**
   - Pre-defined workflow templates
   - Custom workflow definitions
   - Workflow state persistence

### v3 Features
1. **Team Collaboration**
   - Shared specs and reviews
   - Review commenting
   - Approval workflows

2. **IDE Integration**
   - VS Code extension
   - JetBrains plugin
   - Real-time analysis

---

## Conclusion

This MCP server is **highly feasible** and provides a clean way to extend Claude Code's capabilities with Gemini's strengths. The architecture is sound, the implementation is straightforward, and the integration with mcp-server-generator streamlines development.

**Key Success Factors:**
1. MCP is perfect for this use case
2. Clear separation of concerns
3. Leverages each tool's strengths
4. Simple, maintainable codebase
5. Good development workflow

**Next Steps:**
1. Use mcp-server-generator to scaffold project
2. Implement core tools (Phase 2)
3. Test with real projects
4. Iterate based on feedback
5. Deploy and share

This approach transforms the theoretical workflow into a practical, usable tool that makes the Claude Code + Gemini combination truly powerful.
