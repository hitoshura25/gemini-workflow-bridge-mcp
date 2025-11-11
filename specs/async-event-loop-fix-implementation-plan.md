# Async Event Loop Fix - Implementation Plan

**Date:** 2025-11-11
**Issue:** `asyncio.run() cannot be called from a running event loop`
**Priority:** Critical - Blocking MCP server usage

## 1. Problem Statement

### Error Description

When using the MCP server from Claude Code, all tool calls fail with:

```
Error: asyncio.run() cannot be called from a running event loop
```

### Root Cause

The MCP server (FastMCP) runs in an async event loop. When MCP tool handlers are invoked, they're already running within that event loop. However, our business logic functions (`generator.py`) are synchronous functions that call `asyncio.run()` to execute async operations, which attempts to create a nested event loop.

**This is not allowed in Python's asyncio** - you cannot call `asyncio.run()` from within an already-running event loop.

### Why It Happens

```
MCP Server Event Loop (running)
  ↓
MCP Tool Handler (sync) in server.py
  ↓
Business Logic Function (sync) in generator.py
  ↓
asyncio.run() ← ❌ FAILS: tries to create new event loop
  ↓
GeminiClient async method (async)
```

### The Solution

Convert the entire call chain to async, using `await` instead of `asyncio.run()`:

```
MCP Server Event Loop (running)
  ↓
MCP Tool Handler (async) in server.py
  ↓ await
Business Logic Function (async) in generator.py
  ↓ await
GeminiClient async method (async)
```

---

## 2. Current Architecture Analysis

### Layer 1: gemini_client.py (Client Layer) ✅ Already Async

**Status:** No changes needed - already fully async

```python
class GeminiClient:
    async def generate_content(self, prompt: str, ...) -> str:
        # Uses asyncio.create_subprocess_exec

    async def analyze_with_context(self, prompt: str, context: str, ...) -> str:
        # Uses asyncio.create_subprocess_exec
```

### Layer 2: generator.py (Business Logic Layer) ❌ Currently Sync

**Status:** NEEDS CONVERSION - 5 functions, 7 asyncio.run() calls

All main functions are currently synchronous:

1. `def analyze_codebase_with_gemini(...)` - Line 72
2. `def create_specification_with_gemini(...)` - Line 315
3. `def review_code_with_gemini(...)` - Line 494
4. `def generate_documentation_with_gemini(...)` - Line 614
5. `def ask_gemini(...)` - Line 688

Each calls `asyncio.run()` to execute async operations.

### Layer 3: server.py (MCP Tool Layer) ❌ Currently Sync

**Status:** NEEDS CONVERSION - 5 handlers

All MCP tool handlers are currently synchronous:

1. `@mcp.tool() def analyze_codebase_with_gemini(...)` - Line 21
2. `@mcp.tool() def create_specification_with_gemini(...)` - Line 65
3. `@mcp.tool() def review_code_with_gemini(...)` - Line 109
4. `@mcp.tool() def generate_documentation_with_gemini(...)` - Line 153
5. `@mcp.tool() def ask_gemini(...)` - Line 197

Each calls the corresponding generator function synchronously.

---

## 3. Complete List of Changes Required

### File 1: `generator.py` - 5 Functions, 7 Changes

#### Change 1.1: `analyze_codebase_with_gemini()` (Line 72)

**Function Signature:**
```python
# BEFORE (Line 72)
def analyze_codebase_with_gemini(
    focus_description: str,
    directories: List[str] = None,
    file_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> Dict[str, Any]:

# AFTER
async def analyze_codebase_with_gemini(
    focus_description: str,
    directories: List[str] = None,
    file_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> Dict[str, Any]:
```

**asyncio.run() Replacement (Line 126):**
```python
# BEFORE
response = asyncio.run(
    gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=0.7
    )
)

# AFTER
response = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.7
)
```

#### Change 1.2: `create_specification_with_gemini()` (Line 315)

**Function Signature:**
```python
# BEFORE (Line 315)
def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,
    spec_template: str = "standard",
    output_path: str = None
) -> Dict[str, Any]:

# AFTER
async def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,
    spec_template: str = "standard",
    output_path: str = None
) -> Dict[str, Any]:
```

**asyncio.run() Replacement #1 (Line 366):**
```python
# BEFORE
spec_content = asyncio.run(
    gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=0.7
    )
)

# AFTER
spec_content = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.7
)
```

**asyncio.run() Replacement #2 (Line 374):**
```python
# BEFORE
spec_content = asyncio.run(
    gemini_client.generate_content(
        prompt=prompt,
        temperature=0.7
    )
)

# AFTER
spec_content = await gemini_client.generate_content(
    prompt=prompt,
    temperature=0.7
)
```

#### Change 1.3: `review_code_with_gemini()` (Line 494)

**Function Signature:**
```python
# BEFORE (Line 494)
def review_code_with_gemini(
    files: List[str] = None,
    review_focus: List[str] = None,
    spec_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:

# AFTER
async def review_code_with_gemini(
    files: List[str] = None,
    review_focus: List[str] = None,
    spec_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:
```

**asyncio.run() Replacement (Line 566):**
```python
# BEFORE
response = asyncio.run(
    gemini_client.generate_content(
        prompt=prompt,
        temperature=0.3
    )
)

# AFTER
response = await gemini_client.generate_content(
    prompt=prompt,
    temperature=0.3
)
```

#### Change 1.4: `generate_documentation_with_gemini()` (Line 614)

**Function Signature:**
```python
# BEFORE (Line 614)
def generate_documentation_with_gemini(
    documentation_type: str = "api",
    scope: str = None,
    output_path: str = None,
    include_examples: bool = True
) -> Dict[str, Any]:

# AFTER
async def generate_documentation_with_gemini(
    documentation_type: str = "api",
    scope: str = None,
    output_path: str = None,
    include_examples: bool = True
) -> Dict[str, Any]:
```

**asyncio.run() Replacement (Line 654):**
```python
# BEFORE
doc_content = asyncio.run(
    gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=0.7
    )
)

# AFTER
doc_content = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=0.7
)
```

#### Change 1.5: `ask_gemini()` (Line 688)

**Function Signature:**
```python
# BEFORE (Line 688)
def ask_gemini(
    prompt: str,
    include_codebase_context: bool = None,
    context_id: str = None,
    temperature: float = None
) -> Dict[str, Any]:

# AFTER
async def ask_gemini(
    prompt: str,
    include_codebase_context: bool = None,
    context_id: str = None,
    temperature: float = None
) -> Dict[str, Any]:
```

**asyncio.run() Replacement #1 (Line 728):**
```python
# BEFORE
response = asyncio.run(
    gemini_client.analyze_with_context(
        prompt=prompt,
        context=context,
        temperature=temp
    )
)

# AFTER
response = await gemini_client.analyze_with_context(
    prompt=prompt,
    context=context,
    temperature=temp
)
```

**asyncio.run() Replacement #2 (Line 736):**
```python
# BEFORE
response = asyncio.run(
    gemini_client.generate_content(
        prompt=prompt,
        temperature=temp
    )
)

# AFTER
response = await gemini_client.generate_content(
    prompt=prompt,
    temperature=temp
)
```

### File 2: `server.py` - 5 Handlers, 5 Changes

#### Change 2.1: `analyze_codebase_with_gemini()` Handler (Line 21)

```python
# BEFORE (Line 21)
@mcp.tool()
def analyze_codebase_with_gemini(
    focus_description: str,
    directories: List[str] = None,
    file_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> str:
    result = analyze_codebase_with_gemini(
        focus_description=focus_description,
        directories=directories,
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns
    )
    return str(result)

# AFTER
@mcp.tool()
async def analyze_codebase_with_gemini(
    focus_description: str,
    directories: List[str] = None,
    file_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> str:
    result = await analyze_codebase_with_gemini(
        focus_description=focus_description,
        directories=directories,
        file_patterns=file_patterns,
        exclude_patterns=exclude_patterns
    )
    return str(result)
```

#### Change 2.2: `create_specification_with_gemini()` Handler (Line 65)

```python
# BEFORE (Line 65)
@mcp.tool()
def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,
    spec_template: str = "standard",
    output_path: str = None
) -> str:
    result = create_specification_with_gemini(
        feature_description=feature_description,
        context_id=context_id,
        spec_template=spec_template,
        output_path=output_path
    )
    return str(result)

# AFTER
@mcp.tool()
async def create_specification_with_gemini(
    feature_description: str,
    context_id: str = None,
    spec_template: str = "standard",
    output_path: str = None
) -> str:
    result = await create_specification_with_gemini(
        feature_description=feature_description,
        context_id=context_id,
        spec_template=spec_template,
        output_path=output_path
    )
    return str(result)
```

#### Change 2.3: `review_code_with_gemini()` Handler (Line 109)

```python
# BEFORE (Line 109)
@mcp.tool()
def review_code_with_gemini(
    files: List[str] = None,
    review_focus: List[str] = None,
    spec_path: str = None,
    output_path: str = None
) -> str:
    result = review_code_with_gemini(
        files=files,
        review_focus=review_focus,
        spec_path=spec_path,
        output_path=output_path
    )
    return str(result)

# AFTER
@mcp.tool()
async def review_code_with_gemini(
    files: List[str] = None,
    review_focus: List[str] = None,
    spec_path: str = None,
    output_path: str = None
) -> str:
    result = await review_code_with_gemini(
        files=files,
        review_focus=review_focus,
        spec_path=spec_path,
        output_path=output_path
    )
    return str(result)
```

#### Change 2.4: `generate_documentation_with_gemini()` Handler (Line 153)

```python
# BEFORE (Line 153)
@mcp.tool()
def generate_documentation_with_gemini(
    documentation_type: str = "api",
    scope: str = None,
    output_path: str = None,
    include_examples: bool = True
) -> str:
    result = generate_documentation_with_gemini(
        documentation_type=documentation_type,
        scope=scope,
        output_path=output_path,
        include_examples=include_examples
    )
    return str(result)

# AFTER
@mcp.tool()
async def generate_documentation_with_gemini(
    documentation_type: str = "api",
    scope: str = None,
    output_path: str = None,
    include_examples: bool = True
) -> str:
    result = await generate_documentation_with_gemini(
        documentation_type=documentation_type,
        scope=scope,
        output_path=output_path,
        include_examples=include_examples
    )
    return str(result)
```

#### Change 2.5: `ask_gemini()` Handler (Line 197)

```python
# BEFORE (Line 197)
@mcp.tool()
def ask_gemini(
    prompt: str,
    include_codebase_context: bool = None,
    context_id: str = None,
    temperature: float = None
) -> str:
    result = ask_gemini(
        prompt=prompt,
        include_codebase_context=include_codebase_context,
        context_id=context_id,
        temperature=temperature
    )
    return str(result)

# AFTER
@mcp.tool()
async def ask_gemini(
    prompt: str,
    include_codebase_context: bool = None,
    context_id: str = None,
    temperature: float = None
) -> str:
    result = await ask_gemini(
        prompt=prompt,
        include_codebase_context=include_codebase_context,
        context_id=context_id,
        temperature=temperature
    )
    return str(result)
```

---

## 4. Implementation Checklist

### Phase 1: Update generator.py (Business Logic Layer)

- [ ] **Task 1.1:** Convert `analyze_codebase_with_gemini()` to async (Line 72)
  - [ ] Change `def` to `async def`
  - [ ] Replace `asyncio.run()` at line 126 with `await`

- [ ] **Task 1.2:** Convert `create_specification_with_gemini()` to async (Line 315)
  - [ ] Change `def` to `async def`
  - [ ] Replace `asyncio.run()` at line 366 with `await`
  - [ ] Replace `asyncio.run()` at line 374 with `await`

- [ ] **Task 1.3:** Convert `review_code_with_gemini()` to async (Line 494)
  - [ ] Change `def` to `async def`
  - [ ] Replace `asyncio.run()` at line 566 with `await`

- [ ] **Task 1.4:** Convert `generate_documentation_with_gemini()` to async (Line 614)
  - [ ] Change `def` to `async def`
  - [ ] Replace `asyncio.run()` at line 654 with `await`

- [ ] **Task 1.5:** Convert `ask_gemini()` to async (Line 688)
  - [ ] Change `def` to `async def`
  - [ ] Replace `asyncio.run()` at line 728 with `await`
  - [ ] Replace `asyncio.run()` at line 736 with `await`

- [ ] **Verify:** No remaining `asyncio.run()` calls in generator.py

### Phase 2: Update server.py (MCP Tool Layer)

- [ ] **Task 2.1:** Convert `analyze_codebase_with_gemini()` handler to async (Line 21)
  - [ ] Change `def` to `async def`
  - [ ] Add `await` when calling generator function

- [ ] **Task 2.2:** Convert `create_specification_with_gemini()` handler to async (Line 65)
  - [ ] Change `def` to `async def`
  - [ ] Add `await` when calling generator function

- [ ] **Task 2.3:** Convert `review_code_with_gemini()` handler to async (Line 109)
  - [ ] Change `def` to `async def`
  - [ ] Add `await` when calling generator function

- [ ] **Task 2.4:** Convert `generate_documentation_with_gemini()` handler to async (Line 153)
  - [ ] Change `def` to `async def`
  - [ ] Add `await` when calling generator function

- [ ] **Task 2.5:** Convert `ask_gemini()` handler to async (Line 197)
  - [ ] Change `def` to `async def`
  - [ ] Add `await` when calling generator function

- [ ] **Verify:** All MCP handlers are now async

### Phase 3: Update Tests

- [ ] **Task 3.1:** Check if test imports need updating
- [ ] **Task 3.2:** Run test suite: `pytest hitoshura25_gemini_workflow_bridge/tests/ -v`
- [ ] **Task 3.3:** Fix any test failures related to async changes
- [ ] **Task 3.4:** Ensure all 23 tests pass

### Phase 4: Integration Testing

- [ ] **Task 4.1:** Start MCP server in development mode
- [ ] **Task 4.2:** Connect from Claude Code
- [ ] **Task 4.3:** Test `ask_gemini` tool with simple prompt
- [ ] **Task 4.4:** Test `analyze_codebase_with_gemini` tool
- [ ] **Task 4.5:** Test `create_specification_with_gemini` tool
- [ ] **Task 4.6:** Verify no "event loop" errors occur
- [ ] **Task 4.7:** Verify all tools return correct results

---

## 5. Testing Strategy

### Unit Tests (Automated)

**Command:**
```bash
source venv/bin/activate
pytest hitoshura25_gemini_workflow_bridge/tests/ -v
```

**Expected Outcome:**
- All 23 tests should pass
- No new failures introduced
- Tests already use pytest-asyncio, so should handle async functions correctly

**If tests fail:**
- Check if test fixtures need updating
- Verify imports are correct
- Ensure mocks still work with async functions

### Integration Testing (Manual)

**Test 1: Simple Query**
```
Prompt to Claude Code: "Use the ask_gemini tool to explain what MCP servers are"
Expected: Response from Gemini, no event loop error
```

**Test 2: Codebase Analysis**
```
Prompt to Claude Code: "Analyze the authentication system in this codebase"
Expected: Detailed analysis, cached context created, no errors
```

**Test 3: Specification Creation**
```
Prompt to Claude Code: "Create a spec for adding OAuth2 authentication"
Expected: Spec file created in specs/, detailed implementation plan, no errors
```

**Test 4: Code Review**
```
Prompt to Claude Code: "Review the gemini_client.py file for security issues"
Expected: Review file created in reviews/, issues identified, no errors
```

### Error Conditions to Test

**Test for graceful failure:**
- Gemini CLI not installed
- Invalid model specified
- Network timeout
- Malformed prompts

**Expected:** Error messages, no crashes, no event loop errors

---

## 6. What Won't Change (Important!)

### Helper Functions Remain Synchronous

These internal helper functions in `generator.py` are synchronous and should NOT be changed:

- `_get_gemini_client()` - Line 35
- `_get_codebase_loader()` - Line 45
- `_generate_context_id()` - Line 51
- `_build_codebase_context()` - Line 58
- `_get_git_diff()` - Line 275
- `_extract_tasks()` - Line 299
- All `_TEMPLATE_*` constants

**Reason:** These are pure utility functions that don't interact with async operations. They're called from async functions but don't need to be async themselves.

### cli.py Unchanged

The standalone CLI in `cli.py` can remain synchronous because:
- It runs in its own process
- Not invoked through MCP server
- Uses `asyncio.run()` correctly (not from within an event loop)

### Test Structure Unchanged

Test files structure should work as-is because:
- pytest-asyncio already configured
- Mocks work with async functions
- Test fixtures handle async/sync correctly

---

## 7. Rollback Plan

If the changes cause issues:

### Quick Rollback
```bash
git checkout hitoshura25_gemini_workflow_bridge/generator.py
git checkout hitoshura25_gemini_workflow_bridge/server.py
```

### Partial Rollback

If one layer works but the other doesn't:
1. Can rollback just `generator.py` OR just `server.py`
2. But must rollback both if rolling back either (they're tightly coupled)

### Identify Specific Issues

If specific functions fail:
1. Check the exact error message
2. Verify `await` is used correctly (not `asyncio.run()`)
3. Ensure function signature is `async def`
4. Check all callers are using `await`

---

## 8. Success Criteria

The fix is complete and successful when:

✅ All 23 unit tests pass
✅ MCP server starts without errors
✅ All 5 tools can be invoked from Claude Code
✅ No "event loop" errors occur
✅ Tool responses are correct and complete
✅ Specs, reviews, and docs are generated correctly
✅ Performance is acceptable (no significant slowdown)

---

## 9. Additional Notes

### Why FastMCP Supports Async Handlers

FastMCP is built on Starlette/FastAPI, which fully supports async request handlers. When a tool is decorated with `@mcp.tool()` and defined as `async def`, FastMCP automatically:
- Awaits the handler within its event loop
- Manages async context correctly
- Returns results properly

### Import Changes Required

**generator.py:** No new imports needed (already imports asyncio)

**server.py:** No import changes needed

### Performance Impact

**Expected:** Minimal to no performance impact
- Async operations already happening (in GeminiClient)
- Just removing the overhead of `asyncio.run()` creating new event loops
- May actually be slightly faster by staying in one event loop

---

## 10. Timeline Estimate

**Phase 1 (generator.py):** 15-20 minutes
- 5 functions × 3-4 minutes each
- Straightforward find/replace pattern

**Phase 2 (server.py):** 10-15 minutes
- 5 handlers × 2-3 minutes each
- Very similar pattern for all

**Phase 3 (Testing):** 10-15 minutes
- Run tests: 1 minute
- Fix any issues: 5-10 minutes (if needed)
- Re-run tests: 1 minute

**Phase 4 (Integration):** 15-20 minutes
- Setup: 5 minutes
- Test each tool: 10-15 minutes

**Total Estimated Time:** 50-70 minutes

---

## Implementation Ready

This plan is now ready for implementation. Follow the checklist in order, and verify each phase before moving to the next.
