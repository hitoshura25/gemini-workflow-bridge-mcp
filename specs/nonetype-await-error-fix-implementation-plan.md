# Implementation Plan: Fix NoneType Await Error

## Overview

**Issue**: Production error `'NoneType' object can't be awaited` occurring in `create_specification_with_gemini` tool when called by external projects.

**Root Cause**: Missing validation for empty/None responses from Gemini CLI, allowing None values to propagate through async call chains.

**Impact**: Critical - Tool fails completely when Gemini CLI returns empty or malformed responses.

**Version**: To be implemented in v2.0.1 (patch release)

---

## Error Analysis

### Actual Error

```
Error: 'NoneType' object can't be awaited
Tool: create_specification_with_gemini
```

### Why This Error Occurs

This error happens when code tries to `await` something that is `None` instead of a coroutine:

```python
# This causes the error:
result = await None  # TypeError: object NoneType can't be used in 'await' expression

# This is correct:
result = await some_async_function()  # Returns a coroutine
```

### Current Code Vulnerability

In `hitoshura25_gemini_workflow_bridge/gemini_client.py`, the `generate_content()` method (lines 96-110) has several paths that could return empty or None values:

```python
async def generate_content(self, prompt: str, ...) -> str:
    # ...
    try:
        output = stdout.decode('utf-8', errors='replace')
        result = json.loads(output)

        if isinstance(result, dict) and "response" in result:
            return result["response"]  # ⚠️ Could be None!
        else:
            return output  # ⚠️ Could be empty string!
    except json.JSONDecodeError as e:
        output = stdout.decode('utf-8', errors='replace')
        return output  # ⚠️ Could be empty string!
```

**Problem Scenarios**:
1. Gemini CLI returns `{"response": null}` → Returns None
2. Gemini CLI returns empty stdout → Returns empty string ""
3. Gemini CLI crashes silently → Returns empty string ""

---

## Root Cause Investigation

### Files Analyzed

1. **`hitoshura25_gemini_workflow_bridge/gemini_client.py`**
   - `generate_content()` method (lines 52-110)
   - Missing validation for None/empty responses
   - No explicit error handling for malformed CLI output

2. **`hitoshura25_gemini_workflow_bridge/generator.py`**
   - All tool functions call `gemini_client.generate_content()`
   - No defensive checks for empty responses
   - Assumes responses are always valid strings

### Async Call Chain

```
MCP Server (FastMCP)
  ↓ await
create_specification_with_gemini() [generator.py]
  ↓ await
gemini_client.generate_content() [gemini_client.py]
  ↓ returns
None or "" (invalid response)
  ↓
Error propagates back
```

### Why None Reaches Await

If `gemini_client.generate_content()` returns None, and the caller doesn't validate it:

```python
# In generator.py (hypothetical vulnerable code)
response = await gemini_client.generate_content(prompt)
# If response is None, next await could fail
next_result = await process_response(response)  # Error here!
```

---

## Proposed Solution

### Strategy

1. **Add validation in `gemini_client.generate_content()`** - Primary defense
2. **Add validation in tool functions** - Secondary defense (defense in depth)
3. **Improve error messages** - Better debugging
4. **Add comprehensive tests** - Prevent regression

### Design Principles

- **Fail fast**: Detect invalid responses immediately at the source
- **Clear errors**: Provide actionable error messages for debugging
- **Defense in depth**: Multiple validation layers
- **No breaking changes**: Maintain backward compatibility

---

## Implementation Phases

### Phase 1: Add Validation in `gemini_client.generate_content()`

**File**: `hitoshura25_gemini_workflow_bridge/gemini_client.py`

**Location**: Lines 96-110 (response parsing section)

**Before**:
```python
async def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    # ... subprocess execution ...

    try:
        output = stdout.decode('utf-8', errors='replace')
        result = json.loads(output)

        if isinstance(result, dict) and "response" in result:
            return result["response"]
        else:
            return output
    except json.JSONDecodeError as e:
        output = stdout.decode('utf-8', errors='replace')
        return output
```

**After**:
```python
async def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    # ... subprocess execution ...

    try:
        output = stdout.decode('utf-8', errors='replace')

        # Validate output is not empty
        if not output or not output.strip():
            raise RuntimeError(
                "Gemini CLI returned empty response. "
                "Check authentication with: gemini --version"
            )

        result = json.loads(output)

        if isinstance(result, dict) and "response" in result:
            response = result["response"]

            # Validate response is not None
            if response is None:
                raise RuntimeError(
                    f"Gemini CLI returned None response. "
                    f"Raw output: {json.dumps(result)}"
                )

            # Validate response is not empty
            if not response.strip():
                raise RuntimeError(
                    f"Gemini CLI returned empty response string. "
                    f"Raw output: {json.dumps(result)}"
                )

            return response
        else:
            # Non-dict response or missing "response" key
            if not output.strip():
                raise RuntimeError(
                    "Gemini CLI returned non-JSON empty output. "
                    "Check CLI status with: gemini --version"
                )
            return output

    except json.JSONDecodeError as e:
        output = stdout.decode('utf-8', errors='replace')

        # Validate decoded output
        if not output or not output.strip():
            raise RuntimeError(
                f"Gemini CLI returned invalid/empty output. "
                f"JSON decode error: {str(e)}"
            )

        # Return raw output if JSON parsing fails but output exists
        return output
```

**Changes**:
1. Add empty output validation before JSON parsing
2. Add None check for response value
3. Add empty string check for response value
4. Improve error messages with debugging hints
5. Validate output after JSON decode errors

**Impact**:
- Catches all None/empty response scenarios
- Provides clear error messages for debugging
- No breaking changes (still returns str)

---

### Phase 2: Add Validation in Tool Functions (Defense in Depth)

**File**: `hitoshura25_gemini_workflow_bridge/generator.py`

**Functions to Update**:
1. `create_specification_with_gemini()` (lines ~60-120)
2. `review_code_with_gemini()` (lines ~150-220)

**Pattern to Add**:

```python
# After receiving response from gemini_client
response = await gemini_client.generate_content(prompt)

# Add defensive validation
if not response or not response.strip():
    raise RuntimeError(
        f"Received empty response from Gemini CLI for {operation_name}. "
        f"This may indicate authentication issues or service unavailability."
    )

# Continue with response processing
```

**Example - In `create_specification_with_gemini()`**:

**Before** (around line 100):
```python
# Generate specification
response = await gemini_client.generate_content(prompt)

# Parse response
spec_content = response.strip()
```

**After**:
```python
# Generate specification
response = await gemini_client.generate_content(prompt)

# Defensive validation (belt-and-suspenders approach)
if not response or not response.strip():
    raise RuntimeError(
        "Received empty specification response from Gemini CLI. "
        "Check authentication: gemini --version"
    )

# Parse response
spec_content = response.strip()
```

**Apply same pattern to**:
- `create_specification_with_gemini()` - After response generation
- `review_code_with_gemini()` - After response generation

**Why Both Layers?**:
- Layer 1 (gemini_client): Catches issues at the source
- Layer 2 (generator): Additional safety if Layer 1 bypassed in testing/mocking
- Defense in depth principle

---

### Phase 3: Improve Error Logging

**File**: `hitoshura25_gemini_workflow_bridge/gemini_client.py`

**Add Debug Logging**:

```python
import logging

logger = logging.getLogger(__name__)

async def generate_content(self, prompt: str, ...) -> str:
    # ... subprocess execution ...

    # Add debug logging
    logger.debug(f"Gemini CLI returncode: {process.returncode}")
    logger.debug(f"Gemini CLI stdout length: {len(stdout)}")
    logger.debug(f"Gemini CLI stderr: {stderr.decode('utf-8', errors='replace')}")

    try:
        output = stdout.decode('utf-8', errors='replace')

        # Log raw output for debugging (truncated)
        logger.debug(f"Gemini CLI raw output (first 500 chars): {output[:500]}")

        # ... validation and parsing ...
```

**Benefits**:
- Easier debugging in production
- Can identify patterns in failures
- Helps with CLI authentication issues

---

### Phase 4: Add Error Condition Tests

**File**: `hitoshura25_gemini_workflow_bridge/tests/test_gemini_client.py`

**New Tests to Add**:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from hitoshura25_gemini_workflow_bridge.gemini_client import GeminiClient


@pytest.mark.asyncio
async def test_generate_content_empty_stdout():
    """Test that empty stdout raises clear error."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_none_response():
    """Test that None response value raises clear error."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": null}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": null}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="None response"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_empty_string_response():
    """Test that empty string response raises clear error."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": ""}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": ""}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response string"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_whitespace_only_response():
    """Test that whitespace-only response raises clear error."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Gemini CLI returns {"response": "   \n\t   "}
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": "   \\n\\t   "}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="empty response string"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_json_decode_error_empty():
    """Test that JSON decode error with empty output raises clear error."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # Invalid JSON that's empty after decode
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="invalid/empty output"):
            await client.generate_content("test prompt")


@pytest.mark.asyncio
async def test_generate_content_valid_response():
    """Test that valid responses work correctly."""
    client = GeminiClient(model="auto")

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(b'{"response": "Valid response text"}', b"")
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await client.generate_content("test prompt")
        assert result == "Valid response text"
```

**Test Coverage**:
- Empty stdout
- None response value
- Empty string response
- Whitespace-only response
- JSON decode error with empty output
- Valid response (regression test)

**Expected Results**: All new tests should pass with Phase 1 implementation

---

### Phase 5: Integration Testing

**File**: `hitoshura25_gemini_workflow_bridge/tests/test_generator.py`

**Add End-to-End Error Tests**:

```python
@pytest.mark.asyncio
async def test_create_specification_handles_empty_response(tmp_path):
    """Test that create_specification_with_gemini handles empty CLI responses."""
    with patch('hitoshura25_gemini_workflow_bridge.generator.GeminiClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.generate_content = AsyncMock(return_value="")  # Empty response
        mock_client_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="empty"):
            await create_specification_with_gemini(
                feature_request="Test feature",
                context_path=str(tmp_path)
            )


@pytest.mark.asyncio
async def test_review_code_handles_none_response(tmp_path):
    """Test that review_code_with_gemini handles None responses."""
    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    with patch('hitoshura25_gemini_workflow_bridge.generator.GeminiClient') as mock_client_class:
        mock_client = AsyncMock()
        # Simulate None response (though should be caught earlier)
        mock_client.generate_content = AsyncMock(
            side_effect=RuntimeError("Gemini CLI returned None response")
        )
        mock_client_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="None response"):
            await review_code_with_gemini(
                files_to_review=[str(test_file)],
                context_path=str(tmp_path)
            )
```

---

## Testing Strategy

### Unit Tests
- Test `gemini_client.generate_content()` with all error conditions
- Verify error messages are clear and actionable
- Test validation logic in isolation

### Integration Tests
- Test full tool execution with mocked empty responses
- Verify error propagation through async call chain
- Test defense-in-depth (both validation layers)

### Manual Testing
1. Run tests: `pytest hitoshura25_gemini_workflow_bridge/tests/ -v`
2. Test with actual Gemini CLI (if authenticated)
3. Test error scenarios with CLI not installed
4. Verify error messages in production-like environment

### Expected Test Results
- **Before Fix**: 51 tests passing
- **After Fix**: 57 tests passing (6 new error condition tests)
- **No Regressions**: All existing tests must still pass

---

## Implementation Order

1. **Phase 1** - Add validation in `gemini_client.py` (Critical)
   - This is the primary fix
   - Catches all error scenarios
   - Clear error messages

2. **Phase 4** - Add unit tests for `gemini_client.py` (Critical)
   - Verify Phase 1 works correctly
   - Test all error conditions
   - Ensure no regressions

3. **Phase 2** - Add validation in `generator.py` (Important)
   - Defense in depth
   - Additional safety layer
   - Better error context

4. **Phase 5** - Add integration tests (Important)
   - Test end-to-end behavior
   - Verify error propagation
   - Test tool functions

5. **Phase 3** - Add debug logging (Optional but recommended)
   - Better production debugging
   - Helps identify patterns
   - Low implementation cost

---

## Rollout Plan

### Version: v2.0.1 (Patch Release)

**Changes**:
- Fix: Add validation for empty/None responses from Gemini CLI
- Fix: Improve error messages for CLI authentication issues
- Test: Add 6 new error condition tests
- Docs: Update troubleshooting section in README

**Breaking Changes**: None

**Migration**: No changes required for existing users

### Testing Before Release
1. Run full test suite: `pytest`
2. Test with Gemini CLI installed
3. Test with Gemini CLI not installed (expect clear errors)
4. Test with unauthenticated Gemini CLI (expect clear errors)

### Documentation Updates

**README.md** - Add troubleshooting section:
```markdown
## Troubleshooting

### "Gemini CLI returned empty response"

This error occurs when the Gemini CLI fails to generate content. Common causes:

1. **Not authenticated**: Run `gemini` and follow authentication prompts
2. **CLI not installed**: Install with `npm install -g @google/gemini-cli`
3. **Network issues**: Check internet connectivity
4. **Service unavailable**: Try again later

**Verify CLI status**:
```bash
gemini --version
gemini --help
```

If CLI is working but error persists, check logs for more details.
```

---

## Success Criteria

1. ✅ No more NoneType await errors in production
2. ✅ All 57 tests passing (51 existing + 6 new)
3. ✅ Clear error messages for debugging
4. ✅ No breaking changes to existing API
5. ✅ Documentation updated with troubleshooting
6. ✅ Defense in depth validation (multiple layers)

---

## Risk Assessment

### Low Risk
- Changes are additive (validation only)
- No changes to core logic
- Comprehensive test coverage
- Clear rollback path (revert validation)

### Edge Cases Covered
- Empty stdout from CLI
- None response value
- Empty string response
- Whitespace-only response
- JSON decode errors
- CLI not installed
- CLI not authenticated

### Rollback Plan
If issues arise:
1. Revert to v2.0.0
2. Review error logs
3. Add specific test for reported case
4. Re-implement with fix

---

## Post-Implementation

### Monitoring
- Track error rates after deployment
- Monitor for new error patterns
- Collect user feedback

### Future Enhancements
- Add retry logic for transient failures
- Add exponential backoff for CLI calls
- Add response validation schemas
- Add CLI health check tool

---

## References

- Original error report: "'NoneType' object can't be awaited"
- Tool: `create_specification_with_gemini`
- Version: v2.0.0 (current)
- Target version: v2.0.1 (patch)
