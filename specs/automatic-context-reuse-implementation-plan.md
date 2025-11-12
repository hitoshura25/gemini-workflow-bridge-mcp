# Implementation Plan: Automatic Context Reuse with Time-Based Expiration

**Date**: 2025-01-12
**Status**: Planning

## Overview

Remove explicit `context_id` parameters from all tools and implement automatic session-level context reuse with time-based expiration (TTL). This simplifies the API and eliminates redundant codebase reloading within sessions.

## Problem Statement

### Current Behavior (Inefficient)

Every tool call without `context_id` triggers a full codebase load + Gemini analysis:

```python
# Call 1 - loads and analyzes codebase
create_specification_with_gemini(feature_description="2FA")
# Returns context_id: "ctx_abc123"

# Call 2 - LOADS AND RE-ANALYZES AGAIN (wasteful!)
review_code_with_gemini(files=["auth.py"])
# Returns NEW context_id: "ctx_def456"

# Call 3 - LOADS AND RE-ANALYZES AGAIN (wasteful!)
generate_documentation_with_gemini(documentation_type="api", scope="auth")
# Returns NEW context_id: "ctx_ghi789"
```

**Issues:**
- Expensive (multiple Gemini API calls)
- Slow (re-scanning filesystem each time)
- Redundant (analyzing same codebase repeatedly)
- Complex API (users must track and pass `context_id`)

### Desired Behavior (Efficient)

Session-level "current context" that tools automatically reuse:

```python
# Call 1 - loads and caches as "current context" (30 min TTL)
create_specification_with_gemini(feature_description="2FA")

# Call 2 - AUTOMATICALLY reuses "current context" (fast!)
review_code_with_gemini(files=["auth.py"])

# Call 3 - AUTOMATICALLY reuses "current context" (fast!)
generate_documentation_with_gemini(documentation_type="api", scope="auth")

# Call 4 (31 minutes later) - Auto-refreshes expired context
ask_gemini(prompt="Explain auth flow", include_codebase_context=True)
```

**Benefits:**
- No manual context ID tracking
- Automatic reuse within session
- Auto-refresh when stale
- Simpler API

## Goals

1. **Simplify API**: Remove `context_id` from all 5 tool signatures
2. **Automatic Reuse**: Tools automatically use the "current" cached context
3. **Time-Based Expiration**: Context expires after N minutes (configurable, default 30)
4. **Performance**: Eliminate redundant codebase reloading within sessions
5. **Smart Refresh**: Auto-detect and reload expired contexts

## Current Implementation Analysis

### Cache Structure (`gemini_client.py`)

```python
class GeminiClient:
    def __init__(self, model: str = "auto"):
        self.context_cache: Dict[str, Any] = {}  # Simple dict, no TTL

    def cache_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Cache context for reuse"""
        self.context_cache[context_id] = context

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context"""
        return self.context_cache.get(context_id)
```

**Cached Context Structure:**
```python
{
    "files_content": Dict[str, str],      # File paths -> content mapping
    "project_structure": str,              # ASCII tree structure
    "analysis": {                          # Gemini analysis results
        "architecture_summary": str,
        "relevant_files": List[str],
        "patterns_identified": List[str],
        "integration_points": List[str]
    }
}
```

### Issues with Current Implementation

1. **No TTL/Expiration**: Cached contexts remain forever (or until server restart)
2. **No Timestamps**: Can't determine when context was cached
3. **No "Current Context" Tracking**: No way to automatically reuse last context
4. **Manual ID Management**: Users must track and pass `context_id` between calls
5. **No Automatic Invalidation**: Stale cached contexts can be reused even if source files change

## Proposed Solution

### Architecture Changes

```
┌─────────────────────────────────────────────────────────┐
│                  Tool Functions                          │
│  (create_spec, review_code, generate_docs, ask_gemini)  │
│                                                          │
│  NO MORE context_id parameters!                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│            _get_or_load_context()                        │
│                                                          │
│  1. Check if "current context" exists                   │
│  2. Check if current context is expired (TTL)           │
│  3. Return cached context OR auto-reload                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│            ContextCacheManager                           │
│                                                          │
│  • get_current_context() -> checks TTL                  │
│  • set_current_context(id) -> marks as active           │
│  • is_expired(id) -> checks timestamp vs TTL            │
│  • cleanup_expired() -> removes old entries             │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Create Enhanced Cache Manager (30 min)

**File**: `hitoshura25_gemini_workflow_bridge/cache_manager.py` (NEW)

```python
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import os

class ContextCacheManager:
    """Manages context caching with TTL and automatic reuse."""

    def __init__(self, ttl_minutes: int = 30):
        """Initialize cache manager.

        Args:
            ttl_minutes: Time-to-live for cached contexts in minutes
        """
        self.ttl_minutes = ttl_minutes
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.current_context_id: Optional[str] = None
        self.stats = {"hits": 0, "misses": 0, "expirations": 0}

    def cache_context(
        self,
        context_id: str,
        context: Dict[str, Any],
        set_as_current: bool = True
    ) -> None:
        """Cache context with timestamp.

        Args:
            context_id: Unique identifier for context
            context: Context data to cache
            set_as_current: Whether to set as current context
        """
        self.cache[context_id] = {
            "data": context,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=self.ttl_minutes),
            "access_count": 0
        }

        if set_as_current:
            self.current_context_id = context_id

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context if not expired.

        Args:
            context_id: Context ID to retrieve

        Returns:
            Context data or None if not found/expired
        """
        if context_id not in self.cache:
            self.stats["misses"] += 1
            return None

        entry = self.cache[context_id]

        # Check expiration
        if datetime.now() > entry["expires_at"]:
            self.stats["expirations"] += 1
            del self.cache[context_id]
            if self.current_context_id == context_id:
                self.current_context_id = None
            return None

        # Update access stats
        entry["access_count"] += 1
        self.stats["hits"] += 1

        return entry["data"]

    def get_current_context(self) -> Optional[tuple[Dict[str, Any], str]]:
        """Get the current active context.

        Returns:
            Tuple of (context_data, context_id) or None if no current context
        """
        if not self.current_context_id:
            return None

        context = self.get_cached_context(self.current_context_id)
        if context:
            return context, self.current_context_id

        # Current context expired
        return None

    def set_current_context(self, context_id: str) -> bool:
        """Set a context as the current context.

        Args:
            context_id: Context ID to set as current

        Returns:
            True if successful, False if context not found/expired
        """
        if self.get_cached_context(context_id):
            self.current_context_id = context_id
            return True
        return False

    def is_expired(self, context_id: str) -> bool:
        """Check if context is expired.

        Args:
            context_id: Context ID to check

        Returns:
            True if expired or not found, False otherwise
        """
        if context_id not in self.cache:
            return True

        return datetime.now() > self.cache[context_id]["expires_at"]

    def cleanup_expired(self) -> int:
        """Remove all expired contexts.

        Returns:
            Number of contexts removed
        """
        expired = [
            cid for cid, entry in self.cache.items()
            if datetime.now() > entry["expires_at"]
        ]

        for cid in expired:
            del self.cache[cid]
            if self.current_context_id == cid:
                self.current_context_id = None

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            **self.stats,
            "cached_contexts": len(self.cache),
            "current_context_id": self.current_context_id,
            "hit_rate": (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0.0
            )
        }

    def clear(self) -> None:
        """Clear all cached contexts."""
        self.cache.clear()
        self.current_context_id = None
```

**Tests**: `hitoshura25_gemini_workflow_bridge/tests/test_cache_manager.py` (NEW)

### Phase 2: Update GeminiClient (20 min)

**File**: `hitoshura25_gemini_workflow_bridge/gemini_client.py`

**Changes:**

```python
from .cache_manager import ContextCacheManager

class GeminiClient:
    def __init__(self, model: str = "auto"):
        # ... existing initialization ...

        # REPLACE: self.context_cache: Dict[str, Any] = {}
        # WITH:
        ttl_minutes = int(os.getenv("CONTEXT_CACHE_TTL_MINUTES", "30"))
        self.cache_manager = ContextCacheManager(ttl_minutes=ttl_minutes)

    def cache_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Cache context for reuse (auto-sets as current)."""
        self.cache_manager.cache_context(
            context_id,
            context,
            set_as_current=True  # Always set as current
        )

    def get_cached_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context (checks TTL)."""
        return self.cache_manager.get_cached_context(context_id)

    def get_current_context(self) -> Optional[tuple[Dict[str, Any], str]]:
        """Get the current active context."""
        return self.cache_manager.get_current_context()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_stats()
```

### Phase 3: Update Generator Functions (40 min)

**File**: `hitoshura25_gemini_workflow_bridge/generator.py`

#### 3.1 Update `_get_or_load_context()` Helper

**BEFORE:**
```python
async def _get_or_load_context(
    context_id: str = None,
    focus_description: str = "general analysis"
) -> tuple[str, str]:
    """Get cached context or auto-load if not available."""
    gemini_client = _get_gemini_client()

    # Try cached context first
    if context_id:
        cached = gemini_client.get_cached_context(context_id)
        if cached:
            return _format_cached_context(cached), context_id
        print(f"Warning: Context ID '{context_id}' not found in cache. Auto-loading fresh context.")

    # Auto-load codebase
    return await _auto_load_context(focus_description)
```

**AFTER:**
```python
async def _get_or_load_context(
    focus_description: str = "general analysis"
) -> tuple[str, str]:
    """Get current context or auto-load if expired/missing.

    This function automatically:
    1. Checks for current cached context
    2. Validates TTL expiration
    3. Auto-loads fresh context if needed

    Args:
        focus_description: What to focus on if reloading

    Returns:
        Tuple of (formatted_context_string, context_id)
    """
    gemini_client = _get_gemini_client()

    # Try to get current context
    current = gemini_client.get_current_context()

    if current:
        context_data, context_id = current
        print(f"ℹ️  Reusing cached context (ID: {context_id[:12]}...)")
        return _format_cached_context(context_data), context_id

    # No current context or expired - auto-load
    print(f"ℹ️  Loading fresh codebase context (TTL: {gemini_client.cache_manager.ttl_minutes} minutes)...")
    return await _auto_load_context(focus_description)
```

#### 3.2 Remove `context_id` Parameter from Tools

**Tool 1**: `create_specification_with_gemini`

**BEFORE:**
```python
async def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,  # REMOVE THIS
    spec_template: str = None,
    output_path: str = None
) -> Dict[str, Any]:
```

**AFTER:**
```python
async def create_specification_with_gemini(
    feature_description: str,
    spec_template: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """Generate detailed technical specification using full codebase context.

    This tool automatically loads and analyzes your codebase (or reuses
    recently cached context). Context is cached for 30 minutes by default.

    Args:
        feature_description: What feature to specify
        spec_template: Specification template to use
        output_path: Where to save the spec

    Returns:
        Result dictionary with specification (no context_id)
    """
    try:
        gemini_client = _get_gemini_client()

        # Get or auto-load context (no context_id parameter!)
        context, _ = await _get_or_load_context(
            focus_description=f"creating specification for: {feature_description}"
        )

        # ... rest of function ...

        return {
            "spec_path": str(output_path),
            "spec_content": spec_content,
            "implementation_tasks": tasks,
            "estimated_complexity": _estimate_complexity(spec_content),
            "files_to_modify": files_to_modify,
            "files_to_create": files_to_create
            # REMOVED: "context_id": resolved_context_id
        }
```

**Repeat similar changes for:**
- `review_code_with_gemini()` - remove `context_id` parameter
- `generate_documentation_with_gemini()` - remove `context_id` parameter
- `ask_gemini()` - keep `include_codebase_context` but remove `context_id`

**Keep `context_id` in `analyze_codebase_with_gemini()`** for explicit analysis requests.

### Phase 4: Update MCP Server Tool Signatures (10 min)

**File**: `hitoshura25_gemini_workflow_bridge/server.py`

**Tool 1**: `create_specification_with_gemini`

**BEFORE:**
```python
@mcp.tool()
async def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,  # REMOVE
    spec_template: str = None,
    output_path: str = None
) -> str:
```

**AFTER:**
```python
@mcp.tool()
async def create_specification_with_gemini(
    feature_description: str,
    spec_template: str = None,
    output_path: str = None
) -> str:
    """Generate detailed technical specification using full codebase context

    This tool automatically loads and analyzes your codebase (or reuses
    recently cached context within the session). No manual context management
    required!

    Args:
        feature_description: What feature to specify
        spec_template: Specification template to use (standard/minimal)
        output_path: Where to save the spec

    Returns:
        JSON string containing spec_path, spec_content, implementation_tasks,
        estimated_complexity, files_to_modify, and files_to_create
    """
```

**Repeat for:**
- `review_code_with_gemini`
- `generate_documentation_with_gemini`
- `ask_gemini`

### Phase 5: Update Tests (30 min)

**File**: `hitoshura25_gemini_workflow_bridge/tests/test_generator.py`

#### 5.1 Remove `context_id` from Existing Tests

Update all test calls to remove `context_id` parameter:

```python
# BEFORE
result = await create_specification_with_gemini(
    feature_description="test feature",
    context_id="ctx_test123",  # REMOVE
    output_path=str(output_file)
)

# AFTER
result = await create_specification_with_gemini(
    feature_description="test feature",
    output_path=str(output_file)
)
```

#### 5.2 Add New Tests for Automatic Context Reuse

```python
@pytest.mark.asyncio
async def test_automatic_context_reuse(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that multiple tool calls automatically reuse cached context."""
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\\n\\n## Tasks\\n- Task 1"
    )

    # First call - should load codebase
    spec_result = await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(tmp_path / "spec.md")
    )

    load_call_count = mock_codebase_loader.load_files.call_count
    assert load_call_count == 1  # Should have loaded once

    # Second call - should reuse context (NOT reload)
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value='{"issues_found": [], "summary": "Good", "has_blocking_issues": false, "recommendations": []}'
    )

    review_result = await review_code_with_gemini(
        files=["test.py"],
        output_path=str(tmp_path / "review.md")
    )

    # Should NOT have reloaded codebase
    assert mock_codebase_loader.load_files.call_count == load_call_count


@pytest.mark.asyncio
async def test_context_expiration_triggers_reload(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that expired context triggers automatic reload."""
    from unittest.mock import patch
    from datetime import datetime, timedelta

    # Create cache manager with very short TTL for testing
    mock_gemini_client.cache_manager = ContextCacheManager(ttl_minutes=0.01)  # 0.6 seconds

    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\\n\\n## Tasks\\n- Task 1"
    )

    # First call - loads context
    await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(tmp_path / "spec.md")
    )

    load_count_before = mock_codebase_loader.load_files.call_count

    # Wait for expiration
    import time
    time.sleep(1)

    # Second call - should reload because context expired
    await create_specification_with_gemini(
        feature_description="another feature",
        output_path=str(tmp_path / "spec2.md")
    )

    # Should have reloaded
    assert mock_codebase_loader.load_files.call_count > load_count_before


@pytest.mark.asyncio
async def test_cache_stats_tracking(mock_gemini_client, mock_codebase_loader, tmp_path):
    """Test that cache statistics are tracked correctly."""
    mock_gemini_client.analyze_with_context = AsyncMock(
        return_value="# Test Spec\\n\\n## Tasks\\n- Task 1"
    )

    # First call - cache miss
    await create_specification_with_gemini(
        feature_description="test feature",
        output_path=str(tmp_path / "spec.md")
    )

    # Second call - cache hit
    await review_code_with_gemini(
        files=["test.py"],
        output_path=str(tmp_path / "review.md")
    )

    stats = mock_gemini_client.get_cache_stats()
    assert stats["hits"] >= 1
    assert stats["cached_contexts"] >= 1
```

#### 5.3 Update Cache Miss Tests

Remove or update tests that explicitly test `context_id` parameter:
- `test_cache_miss_fallback_in_create_spec` - UPDATE to test automatic reuse
- `test_cache_miss_fallback_in_review` - UPDATE to test automatic reuse
- `test_context_reuse_across_calls` - UPDATE to remove explicit context_id passing

### Phase 6: Update Documentation (20 min)

**File**: `README.md`

#### 6.1 Update "How It Works" Section

**BEFORE:**
```markdown
**Optimized Multi-Call Workflow** (For multiple related operations):
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
```

**AFTER:**
```markdown
**Automatic Context Reuse** (Default behavior - no manual management!):

# First call - loads and caches context (30 min TTL)
create_specification_with_gemini({
  feature_description: "2FA authentication"
})

# Second call - AUTOMATICALLY reuses cached context (fast!)
review_code_with_gemini({
  files: ["auth.py", "middleware.py"]
})

# Third call - AUTOMATICALLY reuses cached context (fast!)
generate_documentation_with_gemini({
  documentation_type: "api",
  scope: "authentication system"
})

# After 30 minutes - automatically refreshes context
ask_gemini({
  prompt: "Explain the auth flow",
  include_codebase_context: true
})
```

#### 6.2 Update Tool Reference Section

Remove all references to `context_id` parameter from:
- `create_specification_with_gemini`
- `review_code_with_gemini`
- `generate_documentation_with_gemini`
- `ask_gemini`

Add new "Configuration" section:

```markdown
## Configuration

### Context Cache TTL

Control how long codebase context is cached before auto-refresh:

```env
# .env file
CONTEXT_CACHE_TTL_MINUTES=30  # Default: 30 minutes
```

**Recommended values:**
- **Fast iteration** (10-15 minutes): For active development with frequent file changes
- **Standard** (30 minutes): Good balance for most workflows
- **Long sessions** (60-120 minutes): For large codebases where analysis is expensive

### Force Context Refresh

To force a fresh analysis at any time, call:

```
analyze_codebase_with_gemini({
  focus_description: "full refresh"
})
```

This loads fresh context and sets it as the current context for subsequent calls.
```

#### 6.3 Update Examples

Update all usage examples in README to remove `context_id` references.

### Phase 7: Add Configuration Support

**File**: `.env.example`

```env
# Gemini Model Selection
GEMINI_MODEL=auto

# Context Cache Configuration
CONTEXT_CACHE_TTL_MINUTES=30

# Enable cache statistics logging (for debugging)
ENABLE_CACHE_STATS=false

# Output Directories
DEFAULT_SPEC_DIR=./specs
DEFAULT_REVIEW_DIR=./reviews
DEFAULT_CONTEXT_DIR=./.workflow-context
```

## Testing Strategy

### Unit Tests

1. **Cache Manager Tests** (`test_cache_manager.py`)
   - TTL expiration
   - Current context tracking
   - Statistics tracking
   - Cleanup of expired entries

2. **Generator Tests** (`test_generator.py`)
   - Automatic context reuse
   - TTL-based refresh
   - Cache hits/misses
   - Multiple tool calls in sequence

3. **Integration Tests**
   - Full workflow: spec → review → docs (should reuse context)
   - Expiration workflow: wait for TTL → automatic refresh
   - Cache statistics accuracy

### Manual Testing

1. Start MCP server
2. Call `create_specification_with_gemini` - should load context
3. Immediately call `review_code_with_gemini` - should reuse (fast)
4. Check server logs for "Reusing cached context" message
5. Wait 31+ minutes
6. Call another tool - should see "Loading fresh codebase context" message

## Migration Guide for Users

### Breaking Changes

**Removed Parameters:**
- `context_id` parameter removed from 4 tools:
  - `create_specification_with_gemini`
  - `review_code_with_gemini`
  - `generate_documentation_with_gemini`
  - `ask_gemini`

### Migration Steps

**Before**
```python
# Manual context management
analysis = analyze_codebase_with_gemini({
  focus_description: "auth system"
})
context_id = analysis["cached_context_id"]

spec = create_specification_with_gemini({
  feature_description: "2FA",
  context_id: context_id  # Manual passing
})

review = review_code_with_gemini({
  files: ["auth.py"],
  context_id: context_id  # Manual passing
})
```

**After**
```python
# Automatic context management
create_specification_with_gemini({
  feature_description: "2FA"
})
# Context automatically cached

review_code_with_gemini({
  files: ["auth.py"]
})
# Context automatically reused

# Optional: Force refresh if needed
analyze_codebase_with_gemini({
  focus_description: "auth system"
})
```

## Rollout Plan

1. **Version Bump**: Breaking change
2. **CHANGELOG**: Document breaking changes
3. **Release Notes**: Highlight automatic context reuse feature
4. **Migration Guide**: Add to README
5. **GitHub Issues**: Close related issues about context management

## Success Metrics

- ✅ All 30+ tests passing
- ✅ No manual `context_id` management required
- ✅ Cache hit rate > 60% in typical workflows
- ✅ README updated with clear examples
- ✅ Zero breaking issues for existing users (with migration guide)

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking change for existing users | High | clear migration guide, CHANGELOG |
| TTL too short = frequent reloads | Medium | Default 30 min, configurable via env |
| TTL too long = stale context | Medium | Can call `analyze_codebase_with_gemini` to force refresh |
| Memory usage with long-lived contexts | Low | Implement max cache size limit (future) |
| Server restart loses context | Low | Expected behavior, context reloads on first call |

## Future Enhancements

1. **File System Watching**: Auto-detect file changes and invalidate context
2. **Persistent Cache**: Save context to disk for cross-session reuse
3. **Multi-Project Support**: Track different contexts for different projects
4. **Cache Size Limits**: LRU eviction when cache grows too large
5. **Smart Partial Refresh**: Only reload changed files instead of full rescan

## Estimated Timeline

- **Total Time**: 2.5 hours
- **Phase 1** (Cache Manager): 30 min
- **Phase 2** (Client Updates): 20 min
- **Phase 3** (Generator Updates): 40 min
- **Phase 4** (Server Updates): 10 min
- **Phase 5** (Testing): 30 min
- **Phase 6** (Documentation): 20 min

## Approval Required

- [ ] User approves breaking change
- [ ] User approves default TTL of 30 minutes
- [ ] User approves removal of `context_id` from all tool returns
- [ ] Ready to proceed with implementation

---

**Next Steps**: Once approved, begin Phase 1 implementation.
