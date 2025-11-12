"""
Core business logic for hitoshura25-gemini-workflow-bridge.

MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review
"""

from typing import Any, Dict, List
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from .gemini_client import GeminiClient
from .codebase_loader import CodebaseLoader

# Set up logger for info messages
logger = logging.getLogger(__name__)

# Initialize global clients
_gemini_client = None
_codebase_loader = None


def _get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def _get_codebase_loader() -> CodebaseLoader:
    """Get or create codebase loader singleton"""
    global _codebase_loader
    if _codebase_loader is None:
        _codebase_loader = CodebaseLoader()
    return _codebase_loader


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


async def analyze_codebase_with_gemini(
    focus_description: str,
    directories: Any = None,
    file_patterns: Any = None,
    exclude_patterns: Any = None
) -> Dict[str, Any]:
    """
    Analyze codebase using Gemini's 2M token context window

    Args:
        focus_description: What to focus on in the analysis
        directories: Directories to analyze
        file_patterns: File patterns to include
        exclude_patterns: Patterns to exclude

    Returns:
        Result dictionary with analysis
    """
    try:
        # Get clients
        gemini_client = _get_gemini_client()
        codebase_loader = _get_codebase_loader()

        # Parse array parameters
        dirs = directories if isinstance(directories, list) else None
        patterns = file_patterns if isinstance(file_patterns, list) else ["*.py", "*.js", "*.ts", "*.java", "*.go"]
        excludes = exclude_patterns if isinstance(exclude_patterns, list) else ["node_modules/", "dist/", "build/", "__pycache__/"]

        # Load codebase
        files_content = codebase_loader.load_files(
            file_patterns=patterns,
            exclude_patterns=excludes,
            directories=dirs
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

        # Run async function
        response = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.7
        )

        # Defensive validation
        if not response or not response.strip():
            raise RuntimeError(
                "Received empty response from Gemini CLI for codebase analysis. "
                "Check authentication: gemini --version"
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

    except Exception as e:
        return {
            "error": str(e),
            "analysis": f"Error analyzing codebase: {str(e)}"
        }


async def _auto_load_context(
    focus_description: str = "general codebase understanding"
) -> tuple[str, str]:
    """
    Automatically load codebase and create analysis context.

    This is the core logic extracted from analyze_codebase_with_gemini
    to enable auto-loading in other tools when no context_id is provided.

    Args:
        focus_description: What to focus the analysis on

    Returns:
        Tuple of (formatted_context_string, context_id)
        On error, returns minimal fallback context with error details
    """
    try:
        # Get clients
        gemini_client = _get_gemini_client()
        codebase_loader = _get_codebase_loader()

        # Use default patterns for auto-loading
        patterns = ["*.py", "*.js", "*.ts", "*.java", "*.go"]
        excludes = ["node_modules/", "dist/", "build/", "__pycache__/"]

        # Load codebase
        files_content = codebase_loader.load_files(
            file_patterns=patterns,
            exclude_patterns=excludes,
            directories=None  # Load from current directory
        )

        # Get project structure
        project_structure = codebase_loader.get_project_structure()

        # Build context
        context = _build_codebase_context(files_content, project_structure)

        # Perform quick inline analysis
        prompt = f"""Analyze this codebase briefly with focus on: {focus_description}

Provide:
1. Architecture summary
2. Key files relevant to the focus area
3. Patterns and conventions used

Format as JSON with keys: architecture_summary, relevant_files, patterns_identified"""

        analysis_response = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.7
        )

        # Defensive validation
        if not analysis_response or not analysis_response.strip():
            raise RuntimeError(
                "Received empty analysis response from Gemini CLI. "
                "Check authentication: gemini --version"
            )

        # Parse analysis
        try:
            analysis = json.loads(analysis_response)
        except json.JSONDecodeError:
            analysis = {
                "architecture_summary": analysis_response[:500],
                "relevant_files": [],
                "patterns_identified": []
            }

        # Generate context ID
        context_id = _generate_context_id(focus_description, files_content)

        # Cache the context
        gemini_client.cache_context(context_id, {
            "files_content": files_content,
            "project_structure": project_structure,
            "analysis": analysis
        })

        # Return formatted context string and ID
        formatted_context = _format_cached_context({
            "files_content": files_content,
            "project_structure": project_structure,
            "analysis": analysis
        })

        return formatted_context, context_id

    except Exception as e:
        # Log the error (will be visible in MCP server logs)
        print(f"Warning: Auto-context loading failed: {str(e)}. Proceeding with minimal context.")

        # Return minimal fallback context
        fallback_context = f"""# Context Loading Failed

**Error:** {str(e)}

**Note:** Proceeding with limited context. The analysis may be less accurate than usual.
Consider running `analyze_codebase_with_gemini` explicitly if you need detailed context."""

        # Generate error-based context ID
        fallback_id = f"ctx_error_{hashlib.sha256(str(e).encode()).hexdigest()[:8]}"

        return fallback_context, fallback_id


async def _get_or_load_context(
    focus_description: str = "general analysis"
) -> tuple[str, str]:
    """
    Get current context or auto-load if expired/missing.

    This function automatically:
    1. Checks for current cached context
    2. Validates TTL expiration
    3. Auto-loads fresh context if needed

    No manual context ID management required! The cache manager automatically
    tracks the "current" context and reuses it across tool calls within the
    session (default TTL: 30 minutes).

    Args:
        focus_description: Focus for analysis if reloading context

    Returns:
        Tuple of (context_string, context_id)
    """
    gemini_client = _get_gemini_client()

    # Try to get current context (automatically checks TTL)
    current = gemini_client.get_current_context()

    if current:
        context_data, context_id = current
        # Safe truncation for display
        context_id_display = context_id[:12] + "..." if len(context_id) > 12 else context_id
        logger.info(f"Reusing cached context (ID: {context_id_display}, TTL: {gemini_client.cache_manager.ttl_minutes}min)")
        return _format_cached_context(context_data), context_id

    # No current context or expired - auto-load
    logger.info(f"Loading fresh codebase context (TTL: {gemini_client.cache_manager.ttl_minutes}min)...")
    return await _auto_load_context(focus_description)


# Specification templates
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
    "comprehensive": """# {feature_name} - Comprehensive Technical Specification

[Extended version with additional sections for API docs, examples, rollback, monitoring, etc.]"""
}


def _extract_tasks(spec_content: str) -> List[Dict[str, Any]]:
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


def _extract_file_lists(spec_content: str) -> tuple:
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
    task_count = len(_extract_tasks(spec_content))
    content_length = len(spec_content)

    if task_count > 15 or content_length > 10000:
        return "high"
    elif task_count > 7 or content_length > 5000:
        return "medium"
    else:
        return "low"


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


async def create_specification_with_gemini(
    feature_description: str,
    spec_template: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Generate detailed technical specification using full codebase context.

    This tool automatically loads and analyzes your codebase (or reuses
    recently cached context within the session). No manual context management
    required! Context is cached for 30 minutes by default (configurable).

    Args:
        feature_description: What feature to specify
        spec_template: Specification template to use
        output_path: Where to save the spec

    Returns:
        Result dictionary with specification (no context_id needed)
    """
    try:
        gemini_client = _get_gemini_client()

        # Get or auto-load context (automatically reuses current context!)
        context, _ = await _get_or_load_context(
            focus_description=f"creating specification for: {feature_description}"
        )

        # Get template
        template = SPEC_TEMPLATES.get(spec_template or "standard", SPEC_TEMPLATES["standard"])

        # Build prompt (context is always available now)
        prompt = f"""Create a detailed technical specification for the following feature:

Feature: {feature_description}

Use this template structure:
{template}

Based on the codebase context provided, ensure the specification:
1. Aligns with existing architecture and patterns
2. Lists specific files to create/modify
3. Provides ordered implementation tasks
4. Includes comprehensive testing strategy
5. Addresses security and performance
6. Lists all dependencies

Provide the complete specification in markdown format."""

        # Generate specification (always use analyze_with_context now)
        spec_content = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.7
        )

        # Defensive validation
        if not spec_content or not spec_content.strip():
            raise RuntimeError(
                "Received empty specification response from Gemini CLI. "
                "Check authentication: gemini --version"
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

        # Extract tasks
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

    except Exception as e:
        return {
            "error": str(e),
            "spec_content": f"Error creating specification: {str(e)}"
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
        f"**Review Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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


async def review_code_with_gemini(
    files: Any = None,
    review_focus: Any = None,
    spec_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Comprehensive code review using Gemini.

    This tool automatically loads and analyzes your codebase (or reuses
    recently cached context within the session). It reviews git changes
    by default, or specific files if provided.

    Args:
        files: Files to review (defaults to git diff if not provided)
        review_focus: Areas to focus on (e.g., security, performance)
        spec_path: Path to spec to review against
        output_path: Where to save review

    Returns:
        Result dictionary with review (no context_id needed)
    """
    try:
        gemini_client = _get_gemini_client()

        # Get or auto-load codebase context (automatically reuses current context!)
        context, _ = await _get_or_load_context(
            focus_description="code review and quality analysis"
        )

        # Parse array parameters
        file_list = files if isinstance(files, list) else None
        focus_list = review_focus if isinstance(review_focus, list) else ["security", "performance", "best-practices", "testing"]

        # Get files to review
        if not file_list:
            files_content = _get_git_diff()
        else:
            files_content = _load_files(file_list)

        # Load spec if provided
        spec_content = ""
        if spec_path and Path(spec_path).exists():
            spec_content = Path(spec_path).read_text()

        # Build review prompt
        focus_text = ", ".join(focus_list)

        prompt = f"""Conduct a comprehensive code review focusing on: {focus_text}

{"Specification to review against:" if spec_content else ""}
{spec_content if spec_content else ""}

Code to review:
{files_content}

Considering the codebase context provided, please provide:
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

        # Generate review (always use analyze_with_context now)
        response = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.3
        )

        # Defensive validation
        if not response or not response.strip():
            raise RuntimeError(
                "Received empty review response from Gemini CLI. "
                "Check authentication: gemini --version"
            )

        # Parse response
        try:
            review_data = json.loads(response)
        except json.JSONDecodeError:
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
            "summary": review_data.get("summary", ""),
            "recommendations": review_data.get("recommendations", [])
        }

    except Exception as e:
        return {
            "error": str(e),
            "review_content": f"Error reviewing code: {str(e)}"
        }


async def generate_documentation_with_gemini(
    documentation_type: str,
    scope: str,
    output_path: str = None,
    include_examples: bool = None
) -> Dict[str, Any]:
    """
    Generate comprehensive documentation with full codebase context.

    This tool automatically loads and analyzes your codebase (or reuses
    recently cached context within the session) to generate context-aware
    documentation with examples from your actual code.

    Args:
        documentation_type: Type of documentation (api, architecture, user-guide, etc.)
        scope: What to document (e.g., "authentication system", "REST API")
        output_path: Where to save documentation
        include_examples: Include code examples from the codebase

    Returns:
        Result dictionary with documentation (no context_id needed)
    """
    try:
        gemini_client = _get_gemini_client()

        # Get or auto-load context (automatically reuses current context!)
        context, _ = await _get_or_load_context(
            focus_description=f"generating {documentation_type} documentation for: {scope}"
        )

        # Build prompt
        prompt = f"""Generate {documentation_type} documentation for: {scope}

{"Include code examples and usage patterns." if include_examples else ""}

Ensure the documentation:
1. Is clear and well-structured
2. Covers all relevant aspects
3. Includes examples where appropriate
4. Follows best practices for {documentation_type} documentation

Provide the complete documentation in markdown format."""

        # Generate documentation
        doc_content = await gemini_client.analyze_with_context(
            prompt=prompt,
            context=context,
            temperature=0.7
        )

        # Defensive validation
        if not doc_content or not doc_content.strip():
            raise RuntimeError(
                "Received empty documentation response from Gemini CLI. "
                "Check authentication: gemini --version"
            )

        # Determine output path
        if not output_path:
            doc_type_slug = documentation_type.lower().replace(' ', '-')
            docs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./docs"))
            docs_dir.mkdir(exist_ok=True)
            output_path = str(docs_dir / f"{doc_type_slug}-documentation.md")

        # Save documentation
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(doc_content)

        return {
            "doc_path": str(output_path),
            "doc_content": doc_content,
            "sections": ["overview", "details", "examples"] if include_examples else ["overview", "details"],
            "word_count": len(doc_content.split())
        }

    except Exception as e:
        return {
            "error": str(e),
            "doc_content": f"Error generating documentation: {str(e)}"
        }


async def ask_gemini(
    prompt: str,
    include_codebase_context: bool = None,
    temperature: float = None
) -> Dict[str, Any]:
    """
    General-purpose Gemini query with optional codebase context.

    By default, queries are answered without codebase context. Set
    include_codebase_context=True to automatically load and use your
    codebase (or reuse recently cached context).

    Args:
        prompt: Question or task for Gemini
        include_codebase_context: Load full codebase context (default: False)
        temperature: Temperature for generation 0.0-1.0 (default: 0.7)

    Returns:
        Result dictionary with response (no context_id needed)
    """
    try:
        gemini_client = _get_gemini_client()

        if include_codebase_context:
            # Get or auto-load context (automatically reuses current context!)
            context, _ = await _get_or_load_context(
                focus_description=f"answering question: {prompt[:100]}"
            )

            # Generate with context
            temp = float(temperature) if temperature is not None else 0.7
            response = await gemini_client.analyze_with_context(
                prompt=prompt,
                context=context,
                temperature=temp
            )

            # Defensive validation
            if not response or not response.strip():
                raise RuntimeError(
                    "Received empty response from Gemini CLI. "
                    "Check authentication: gemini --version"
                )

            return {
                "response": response,
                "context_used": True,
                "token_count": len(response.split())
            }
        else:
            # No context needed - simple query
            temp = float(temperature) if temperature is not None else 0.7
            response = await gemini_client.generate_content(
                prompt=prompt,
                temperature=temp
            )

            # Defensive validation
            if not response or not response.strip():
                raise RuntimeError(
                    "Received empty response from Gemini CLI. "
                    "Check authentication: gemini --version"
                )

            return {
                "response": response,
                "context_used": False,
                "token_count": len(response.split())
            }

    except Exception as e:
        return {
            "error": str(e),
            "response": f"Error querying Gemini: {str(e)}"
        }
