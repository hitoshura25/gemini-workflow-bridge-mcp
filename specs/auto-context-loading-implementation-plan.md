# Auto-Context Loading Implementation Plan

**Created**: 2025-01-11
**Status**: Ready for Implementation
**Complexity**: Medium (2-3 hours)

## Problem Statement

### User Experience Issue

When users call MCP tools like `create_specification_with_gemini` directly without first running `analyze_codebase_with_gemini`, the tools generate output with **zero codebase context**, resulting in:

- Generic, template-like specifications
- No awareness of existing architecture
- Missing technical details specific to the codebase
- Significantly lower quality compared to Claude's direct output

### Root Cause

The current implementation has a critical workflow gap:

**Current Two-Path Design:**

```
Path 1 (High Quality):
analyze_codebase_with_gemini → Returns context_id
                              ↓
create_specification_with_gemini(context_id=<id>) → Loads cached context → HIGH QUALITY OUTPUT

Path 2 (Low Quality - What Users Actually Do):
create_specification_with_gemini() → No context_id → NO CODEBASE LOADED → LOW QUALITY OUTPUT
```

**The Core Issue:**

Looking at `generator.py:335-373`:

```python
async def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,  # Optional - users skip this
    ...
):
    # Get cached context if available
    context = ""
    if context_id:
        cached = gemini_client.get_cached_context(context_id)
        if cached:
            context = _format_cached_context(cached)

    # If no context_id, context stays empty!

    if context:
        # Uses analyze_with_context - includes codebase
        spec_content = await gemini_client.analyze_with_context(...)
    else:
        # Just generate_content - NO CODEBASE!
        spec_content = await gemini_client.generate_content(...)
```

**When `context_id=None` (the default):**
1. `context = ""` (line 335)
2. Tool falls through to `generate_content()` (line 370)
3. NO codebase is loaded
4. Specification generated in vacuum

### Why This Happens

1. **No automatic context passing** - Users must manually track and pass `context_id` between tool calls
2. **No intelligent fallback** - Tools don't auto-load codebase when context is missing
3. **Non-obvious workflow** - Tool interface doesn't indicate the two-step requirement
4. **Error-prone manual process** - Easy to forget the first step

## Solution: Hybrid Auto-Loading Approach

### Design Philosophy

**Make each tool self-sufficient while enabling optimization when context is reused.**

### Key Principles

1. **Smart Defaults**: If no `context_id` provided, automatically load and analyze codebase
2. **Optimization Path**: Return `context_id` in results so Claude Code can reuse it
3. **Backward Compatibility**: Existing code with manual `context_id` passing still works
4. **Zero Configuration**: No session management or complex state

### User Experience After Implementation

**Single Call (Auto-Loading):**
```
User: "Create a spec for user authentication"
      ↓
create_specification_with_gemini(feature_description="user authentication")
      ↓
Tool internally:
  1. Detects no context_id
  2. Loads codebase automatically
  3. Performs inline analysis
  4. Generates spec WITH full context
  5. Returns spec + context_id
      ↓
Result: HIGH QUALITY SPEC (same as 2-step manual process)
```

**Optimized Multi-Call (Context Reuse):**
```
Call 1: create_specification_with_gemini(...)
        → Returns {spec, context_id: "ctx_abc123"}

Claude Code detects context_id in response

Call 2: review_code_with_gemini(context_id="ctx_abc123")
        → Reuses cached context, skips reload

Result: Fast subsequent calls, shared context
```

## Architecture Changes

### New Helper Function: `_auto_load_context()`

Extract the codebase loading logic from `analyze_codebase_with_gemini` into a reusable helper:

**Location**: `generator.py` (insert after line 169, before `create_specification_with_gemini`)

```python
async def _auto_load_context(
    focus_description: str = "general codebase understanding"
) -> tuple[str, str]:
    """
    Automatically load codebase and create analysis context.

    This is the core logic extracted from analyze_codebase_with_gemini
    to enable auto-loading in other tools.

    Args:
        focus_description: What to focus the analysis on

    Returns:
        Tuple of (formatted_context_string, context_id)
    """
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
```

### Modified Functions (4 total)

#### 1. `create_specification_with_gemini`

**Location**: `generator.py:313-406`

**Current Logic** (lines 335-373):
```python
# Get cached context if available
context = ""
if context_id:
    cached = gemini_client.get_cached_context(context_id)
    if cached:
        context = _format_cached_context(cached)

# Generate specification
if context:
    spec_content = await gemini_client.analyze_with_context(...)
else:
    spec_content = await gemini_client.generate_content(...)
```

**New Logic** (replace lines 334-373):
```python
# Get or auto-load context
context = ""
auto_loaded_context_id = None

if context_id:
    # Use provided context_id
    cached = gemini_client.get_cached_context(context_id)
    if cached:
        context = _format_cached_context(cached)
else:
    # AUTO-LOAD: No context_id provided, load codebase automatically
    context, auto_loaded_context_id = await _auto_load_context(
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

# ... rest of function remains same ...

# UPDATE RETURN to include context_id
return {
    "spec_path": str(output_path),
    "spec_content": spec_content,
    "implementation_tasks": tasks,
    "estimated_complexity": _estimate_complexity(spec_content),
    "files_to_modify": files_to_modify,
    "files_to_create": files_to_create,
    "context_id": context_id or auto_loaded_context_id  # NEW: Return for reuse
}
```

**Key Changes:**
1. Lines 335-340: Add auto-loading branch
2. Lines 352-373: Remove conditional - always use `analyze_with_context`
3. Return dict: Add `context_id` field

#### 2. `review_code_with_gemini`

**Location**: `generator.py:488-584`

**Current Context Logic** (there is none - no context support at all):
```python
# Currently loads git diff or files, but no codebase context!
```

**New Logic** (insert after line 507, before prompt building):
```python
# Get or auto-load codebase context
context = ""
auto_loaded_context_id = None

if context_id:
    cached = gemini_client.get_cached_context(context_id)
    if cached:
        context = _format_cached_context(cached)
else:
    # AUTO-LOAD: Load codebase for better review context
    context, auto_loaded_context_id = await _auto_load_context(
        focus_description="code review and quality analysis"
    )

# Update prompt to include context reference
prompt = f"""Review the following code changes:

{code_to_review}

{"Considering the codebase context provided, " if context else ""}please provide:
1. Issues found (categorized by severity)
2. Summary of overall quality
3. Recommendations for improvement
4. Whether there are blocking issues

{spec_context}

Format as JSON with keys: issues_found (array), summary, has_blocking_issues (bool), recommendations (array)"""

# Use analyze_with_context instead of generate_content
response = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.3
)

# ... rest of function ...

# UPDATE RETURN
return {
    "review_path": str(output_path),
    "review_content": review_markdown,
    "issues_found": review_data.get("issues_found", []),
    "has_blocking_issues": review_data.get("has_blocking_issues", False),
    "summary": review_data.get("summary", ""),
    "recommendations": review_data.get("recommendations", []),
    "context_id": context_id or auto_loaded_context_id  # NEW
}
```

#### 3. `generate_documentation_with_gemini`

**Location**: `generator.py:606-670`

**Current Logic**: Uses context, but only if provided

**New Logic** (replace lines 619-646):
```python
# Get or auto-load context
context = ""
auto_loaded_context_id = None

if context_id:
    cached = gemini_client.get_cached_context(context_id)
    if cached:
        context = _format_cached_context(cached)
else:
    # AUTO-LOAD: Documentation needs full codebase context
    context, auto_loaded_context_id = await _auto_load_context(
        focus_description=f"generating {documentation_type} documentation for: {scope}"
    )

# ... build prompt (context always available) ...

# Always use analyze_with_context
doc_content = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.7
)

# ... save and process ...

# UPDATE RETURN
return {
    "doc_path": str(output_path),
    "doc_content": doc_content,
    "sections": sections,
    "word_count": word_count,
    "context_id": context_id or auto_loaded_context_id  # NEW
}
```

#### 4. `ask_gemini`

**Location**: `generator.py:678-738`

**Current Logic**: Has context support but optional

**New Logic** (replace lines 690-724):
```python
# Determine if context is needed
needs_context = include_codebase_context or context_id

if needs_context:
    context = ""
    auto_loaded_context_id = None

    if context_id:
        # Use cached context
        cached = gemini_client.get_cached_context(context_id)
        if cached:
            context = _format_cached_context(cached)
    elif include_codebase_context:
        # AUTO-LOAD: User explicitly wants codebase context
        context, auto_loaded_context_id = await _auto_load_context(
            focus_description=f"answering question: {prompt[:100]}"
        )

    # Use analyze_with_context
    response = await gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=temp
    )

    return {
        "response": response,
        "context_used": True,
        "token_count": len(response.split()),
        "context_id": context_id or auto_loaded_context_id  # NEW
    }
else:
    # No context needed - simple query
    response = await gemini_client.generate_content(
        prompt=prompt,
        temperature=temp
    )

    return {
        "response": response,
        "context_used": False,
        "token_count": len(response.split())
    }
```

**Key Insight**: For `ask_gemini`, only auto-load if `include_codebase_context=True`. Otherwise, respect the user's intent for a simple query.

### Update `analyze_codebase_with_gemini`

**Refactor to use the helper** (lines 100-130):

```python
# Instead of duplicating logic, call the helper
context, context_id = await _auto_load_context(focus_description)

# Still return the detailed analysis format
# (helper does lighter analysis, this does full analysis)
response = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.7
)

# Parse and return as before
# ... existing parsing logic ...

return {
    "analysis": analysis.get("analysis", response),
    "architecture_summary": analysis.get("architecture_summary", ""),
    "relevant_files": analysis.get("relevant_files", []),
    "patterns_identified": analysis.get("patterns_identified", []),
    "integration_points": analysis.get("integration_points", []),
    "cached_context_id": context_id  # Same as before
}
```

## Implementation Checklist

### Phase 1: Core Implementation

- [ ] Create `_auto_load_context()` helper function
  - [ ] Extract loading logic from `analyze_codebase_with_gemini`
  - [ ] Add inline analysis
  - [ ] Return (context, context_id) tuple
  - [ ] Add error handling

- [ ] Update `create_specification_with_gemini`
  - [ ] Add auto-loading branch
  - [ ] Remove conditional context usage (always use context)
  - [ ] Add `context_id` to return dict
  - [ ] Update docstring

- [ ] Update `review_code_with_gemini`
  - [ ] Add context support with auto-loading
  - [ ] Switch to `analyze_with_context`
  - [ ] Add `context_id` to return dict
  - [ ] Update docstring

- [ ] Update `generate_documentation_with_gemini`
  - [ ] Add auto-loading branch
  - [ ] Ensure always uses context
  - [ ] Add `context_id` to return dict
  - [ ] Update docstring

- [ ] Update `ask_gemini`
  - [ ] Add auto-loading for `include_codebase_context=True`
  - [ ] Add `context_id` to return dict (conditional)
  - [ ] Update docstring

### Phase 2: Documentation

- [ ] Update tool docstrings in `server.py`
  - [ ] Explain auto-loading behavior
  - [ ] Document `context_id` in returns
  - [ ] Provide usage examples

- [ ] Update README.md
  - [ ] Add "How It Works" section
  - [ ] Show single-call vs optimized workflow
  - [ ] Explain context reuse optimization

### Phase 3: Testing

- [ ] Add unit tests for `_auto_load_context()`
  - [ ] Test successful loading
  - [ ] Test caching
  - [ ] Test error handling

- [ ] Update existing tests
  - [ ] Modify mocks to handle auto-loading
  - [ ] Verify backward compatibility (context_id provided)
  - [ ] Test new auto-loading path (no context_id)

- [ ] Add integration tests
  - [ ] Test single call workflow
  - [ ] Test context reuse between calls
  - [ ] Test mixed workflows

- [ ] Run full test suite
  - [ ] All 23 existing tests must pass
  - [ ] New tests for auto-loading must pass

### Phase 4: Validation

- [ ] Manual testing with real MCP server
  - [ ] Call `gemini-spec` directly (should work now!)
  - [ ] Verify output quality matches 2-step workflow
  - [ ] Test context reuse in multi-call scenarios

- [ ] Performance testing
  - [ ] Measure auto-loading overhead (expected: ~2-5s)
  - [ ] Verify caching reduces subsequent calls

## Success Criteria

### Functional Requirements

✅ **Single-call workflow produces high-quality output**
- Calling any tool without `context_id` automatically loads codebase
- Output quality matches manual 2-step workflow
- No user action required

✅ **Context reuse optimization works**
- Tools return `context_id` in results
- Providing `context_id` skips auto-loading
- Subsequent calls are fast (no reload)

✅ **Backward compatibility maintained**
- Existing code with manual `context_id` passing works unchanged
- `analyze_codebase_with_gemini` still works as before
- No breaking API changes

### Non-Functional Requirements

✅ **Performance acceptable**
- Auto-loading adds ~2-5 seconds on first call
- Cached calls remain fast (<1 second)
- No memory leaks from caching

✅ **Error handling robust**
- Gracefully handle loading failures
- Provide clear error messages
- Fallback to no-context if load fails

## Migration Guide

### For Existing Users

**No changes required!** Existing workflows continue to work:

```python
# Old workflow (still works)
result1 = analyze_codebase_with_gemini(focus="auth system")
context_id = result1["cached_context_id"]

result2 = create_specification_with_gemini(
    feature="2FA authentication",
    context_id=context_id
)
```

### For New Users

**Simplified workflow** - just call the tool:

```python
# New workflow (auto-loading)
result = create_specification_with_gemini(
    feature="2FA authentication"
)
# Automatically loads codebase, generates high-quality spec
```

### For Optimization

**Reuse context between calls:**

```python
# First call - auto-loads
spec_result = create_specification_with_gemini(feature="user auth")
context_id = spec_result["context_id"]  # NEW field

# Second call - reuses context (fast!)
review_result = review_code_with_gemini(
    files=["auth.py"],
    context_id=context_id
)
```

## Potential Issues and Mitigations

### Issue 1: Auto-loading Overhead

**Problem**: First call takes 2-5 seconds longer due to codebase loading

**Mitigation**:
- Document the behavior clearly
- Show loading happens once per session
- Explain optimization via context reuse
- Consider adding progress feedback in future

### Issue 2: Large Codebases

**Problem**: Very large codebases (>100k LOC) might exceed token limits

**Mitigation**:
- Current `_build_codebase_context()` already handles this
- Uses smart truncation and summarization
- No changes needed to existing logic

### Issue 3: Cache Invalidation

**Problem**: Cached context becomes stale if code changes during session

**Mitigation**:
- Document cache lifetime (MCP server process lifetime)
- Users can restart server to clear cache
- Future: Add cache TTL or change detection

### Issue 4: Context Relevance

**Problem**: Auto-loaded context uses default patterns, might miss relevant files

**Mitigation**:
- Default patterns cover 90% of use cases
- Users can still call `analyze_codebase_with_gemini` first with custom patterns
- Future: Add pattern hints to tool parameters

## Timeline Estimate

**Total: 2-3 hours**

- **Phase 1 (Core)**: 60-90 minutes
  - Helper function: 15 minutes
  - 4 function updates: 10 minutes each = 40 minutes
  - Testing changes: 15 minutes

- **Phase 2 (Docs)**: 30 minutes
  - Docstrings: 15 minutes
  - README: 15 minutes

- **Phase 3 (Testing)**: 45-60 minutes
  - New tests: 20 minutes
  - Update existing tests: 15 minutes
  - Integration testing: 20 minutes

- **Phase 4 (Validation)**: 15-30 minutes
  - Manual testing: 15 minutes
  - Performance check: 15 minutes

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 implementation
3. Test incrementally after each function update
4. Complete all phases in order
5. Validate with real-world usage

---

**Questions or Concerns?**

Before proceeding with implementation, review this plan and raise any:
- Architectural concerns
- Missing edge cases
- Alternative approaches
- Testing gaps
