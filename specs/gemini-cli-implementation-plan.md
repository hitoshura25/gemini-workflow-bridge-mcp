# Gemini Workflow Bridge MCP - CLI-Only Implementation Plan

**Date:** 2025-01-10
**Objective:** Convert the MCP server to use Gemini CLI exclusively instead of the Gemini API
**Reason:** User has Gemini Code Assist subscription and wants to avoid additional API costs

## Executive Summary

Convert the existing MCP server implementation from `google-genai` API to Gemini CLI (`gemini` command). This is a **minimal change** - we reuse 90% of existing code and only replace the `gemini_client.py` implementation layer.

**Key Decision:** CLI-only (no API fallback) for simplicity and cost avoidance.

## Current State Analysis

### What Works and Should Be Kept

1. **MCP Server Layer** (`server.py` - 44 lines)
   - ✅ FastMCP server initialization
   - ✅ 5 tool definitions with proper schemas
   - ✅ Resource handlers
   - ✅ No changes needed

2. **Tool Implementation** (`generator.py` - 241 lines)
   - ✅ All 5 tool functions (analyze, create_spec, review, document, ask)
   - ✅ Helper functions (_build_codebase_context, _extract_tasks, etc.)
   - ✅ Template system for specifications
   - ✅ Only needs: swap `await client.generate_content()` calls

3. **Supporting Modules**
   - ✅ `codebase_loader.py` - Fully independent, no changes
   - ✅ `resources.py` - Fully independent, no changes

4. **Tests** (`tests/` - 21 tests)
   - ✅ Test structure is solid
   - ✅ Just need to mock subprocess instead of API

### What Needs to Change

1. **`gemini_client.py`** - Complete rewrite (80 lines → 60 lines)
2. **`pyproject.toml`** - Remove `google-genai` dependency
3. **`.env.example`** - Remove API key, document CLI
4. **`README.md`** - Update installation and auth instructions
5. **Tests** - Update mocks to use subprocess

## Implementation Plan

### Phase 1: Create CLI Client Wrapper

**File:** `hitoshura25_gemini_workflow_bridge/gemini_client.py`

**Requirements:**
- Use `asyncio.subprocess` to call `gemini` command
- Support `--output-format json` for structured responses
- Parse JSON output from stdout
- Handle errors from stderr
- Support temperature and model configuration
- Maintain same interface as API client

**Interface (same as before):**
```python
class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash")
    async def generate_content(prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str
    async def analyze_with_context(prompt: str, context: str, temperature: float = 0.7) -> str
    def cache_context(context_id: str, context: Dict[str, Any]) -> None
    def get_cached_context(context_id: str) -> Optional[Dict[str, Any]]
```

**Implementation Strategy:**
```python
async def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    # Build command: gemini --output-format json --model <model> <prompt>
    cmd = ["gemini", "--output-format", "json", "-m", self.model_name, prompt]

    # Execute with asyncio.subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # Parse JSON: {"response": "...", "stats": {...}}
    result = json.loads(stdout.decode())
    return result["response"]
```

**Error Handling:**
- Check if `gemini` command exists (startup validation)
- Parse stderr for CLI errors
- Timeout protection (e.g., 5 minutes per request)
- Handle JSON parsing errors gracefully

**Limitations to Document:**
- Temperature control: CLI may not support (document this)
- max_tokens: CLI may not support (document this)
- If parameters not supported, document as "best effort"

### Phase 2: Update Dependencies

**File:** `pyproject.toml`

**Changes:**
```python
# BEFORE:
dependencies = [
    "mcp>=1.0.0,<2.0.0",
    "google-genai>=1.0.0",  # REMOVE THIS
    "python-dotenv>=1.0.0",
    "gitpython>=3.1.0",
    "pathspec>=0.12.0",
]

# AFTER:
dependencies = [
    "mcp>=1.0.0,<2.0.0",
    # No google-genai needed - uses Gemini CLI
    "python-dotenv>=1.0.0",
    "gitpython>=3.1.0",
    "pathspec>=0.12.0",
]
```

**Benefits:**
- Smaller dependency tree
- Faster installation
- No API client overhead

### Phase 3: Update Configuration

**File:** `.env.example`

**Before:**
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GOOGLE_GENAI_USE_VERTEXAI=false
GOOGLE_CLOUD_PROJECT=your-project-id
...
```

**After:**
```env
# Gemini CLI Configuration
# The Gemini CLI must be installed and authenticated
# Install: npm install -g @google/gemini-cli
# Authenticate: gemini (follow prompts)

GEMINI_MODEL=gemini-2.5-flash-lite
DEFAULT_SPEC_DIR=./specs
DEFAULT_REVIEW_DIR=./reviews
DEFAULT_CONTEXT_DIR=./.workflow-context

# Optional: Override default CLI path
# GEMINI_CLI_PATH=/usr/local/bin/gemini
```

**Key Points:**
- No API key needed
- Document CLI installation steps
- Document authentication process
- Simpler configuration

### Phase 4: Update Documentation

**File:** `README.md`

**New Sections:**

1. **Prerequisites:**
   ```markdown
   ## Prerequisites

   - Python 3.11+
   - Gemini CLI installed and authenticated

   ### Installing Gemini CLI

   ```bash
   npm install -g @google/gemini-cli
   ```

   ### Authenticating Gemini CLI

   Run `gemini` and follow the authentication prompts. Your credentials will be cached.
   ```

2. **Configuration:**
   ```markdown
   ## Configuration

   ### 1. Verify Gemini CLI is Installed

   ```bash
   gemini --version
   # Should show: 0.13.0 or higher
   ```

   ### 2. Authenticate (if not already done)

   ```bash
   gemini
   # Follow the authentication prompts
   ```

   ### 3. Configure the MCP Server

   Create `.env` file (optional):
   ```env
   GEMINI_MODEL=gemini-2.5-flash-lite
   ```

   No API key needed! The MCP server will use your CLI credentials.
   ```

3. **Benefits Section:**
   ```markdown
   ## Why CLI-Based?

   - ✅ **No API costs** - Uses your existing Gemini Code Assist subscription
   - ✅ **Simple auth** - Reuses CLI credentials, no API key management
   - ✅ **Consistent** - Same authentication as your IDE
   - ✅ **Zero config** - Works immediately if CLI is authenticated
   ```

### Phase 5: Update Tests

**File:** `tests/test_generator.py`

**Changes:**

1. **Update mock fixture:**
```python
@pytest.fixture
def mock_gemini_client():
    """Mock Gemini CLI client"""
    with patch('hitoshura25_gemini_workflow_bridge.generator._get_gemini_client') as mock:
        client = Mock()

        # Mock subprocess execution
        async def mock_generate(prompt, temperature=0.7, max_tokens=None):
            return '{"analysis": "test", ...}'

        client.generate_content = AsyncMock(side_effect=mock_generate)
        client.analyze_with_context = AsyncMock(side_effect=mock_generate)
        client.cache_context = Mock()
        client.get_cached_context = Mock(return_value=None)

        mock.return_value = client
        yield client
```

2. **Add CLI-specific tests:**
```python
def test_cli_not_found():
    """Test error when Gemini CLI is not installed"""
    with patch('shutil.which', return_value=None):
        with pytest.raises(RuntimeError, match="Gemini CLI not found"):
            GeminiClient()

def test_cli_json_parsing_error():
    """Test handling of malformed JSON from CLI"""
    # Mock subprocess returning invalid JSON
    ...

def test_cli_timeout():
    """Test timeout handling for long-running CLI calls"""
    # Mock subprocess that takes too long
    ...
```

### Phase 6: Add Startup Validation

**File:** `gemini_client.py`

**Add validation in `__init__`:**
```python
def __init__(self, model: str = "gemini-2.0-flash"):
    # Validate CLI is installed
    cli_path = shutil.which("gemini")
    if not cli_path:
        raise RuntimeError(
            "Gemini CLI not found. Install with: npm install -g @google/gemini-cli"
        )

    # Test CLI is working
    try:
        result = subprocess.run(
            ["gemini", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError("Gemini CLI found but not working")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Gemini CLI not responding")

    self.model_name = model
    self.context_cache: Dict[str, Any] = {}
```

## Implementation Checklist

### Core Changes
- [ ] Rewrite `gemini_client.py` with CLI subprocess wrapper
- [ ] Add startup validation (CLI exists and works)
- [ ] Implement JSON parsing from CLI output
- [ ] Add error handling for CLI failures
- [ ] Add timeout protection

### Configuration
- [ ] Update `pyproject.toml` - remove `google-genai`
- [ ] Update `.env.example` - remove API key, add CLI docs
- [ ] Update `conftest.py` - remove API key mock from tests

### Documentation
- [ ] Update `README.md` - Installation section
- [ ] Update `README.md` - Authentication section
- [ ] Update `README.md` - Add "Why CLI?" section
- [ ] Update `README.md` - Remove API key instructions
- [ ] Update `README.md` - Add troubleshooting for CLI issues

### Testing
- [ ] Update test mocks to use subprocess
- [ ] Add CLI-specific tests (not found, timeout, JSON errors)
- [ ] Update test fixtures in `conftest.py`
- [ ] Run all 21+ tests and verify they pass

### Validation
- [ ] Test with real Gemini CLI
- [ ] Verify all 5 tools work
- [ ] Verify error messages are helpful
- [ ] Test startup validation
- [ ] Test with missing CLI (error message quality)

## Migration Guide for Users

### For New Users
```bash
# 1. Install Gemini CLI
npm install -g @google/gemini-cli

# 2. Authenticate
gemini  # Follow prompts

# 3. Install MCP server
pip install hitoshura25-gemini-workflow-bridge

# 4. Done! No API key needed
```

### For Existing Users (Upgrading from API version)
```bash
# 1. Install Gemini CLI
npm install -g @google/gemini-cli

# 2. Authenticate
gemini

# 3. Update MCP server
pip install -U hitoshura25-gemini-workflow-bridge

# 4. Update .env - REMOVE these lines:
#    GEMINI_API_KEY=...
#    GOOGLE_GENAI_USE_VERTEXAI=...
#    GOOGLE_CLOUD_PROJECT=...

# 5. Restart Claude Code
```

## Technical Considerations

### Performance
- **CLI overhead:** ~50-100ms per call for subprocess spawn
- **Acceptable:** For workflow tasks (analysis, specs), this is negligible
- **Context:** API calls take seconds anyway due to LLM processing

### Limitations
- **Temperature:** May not be fully supported by CLI - document as best-effort
- **max_tokens:** May not be fully supported by CLI - document as best-effort
- **Streaming:** Not supported via CLI (but not used in current implementation)

### Error Handling
- CLI not found → Clear error message with installation instructions
- CLI not authenticated → Clear error message with auth instructions
- CLI errors → Parse stderr and provide helpful context
- Timeouts → 5-minute timeout per request, clear error

### Testing Strategy
- **Unit tests:** Mock subprocess calls
- **Integration tests:** Require real CLI (mark as optional)
- **CI/CD:** Skip integration tests if CLI not available

## Success Criteria

- [ ] All existing functionality works with CLI backend
- [ ] Zero API costs (no API calls made)
- [ ] Simple installation (CLI + pip install)
- [ ] All 21+ tests pass
- [ ] Clear error messages when CLI not available
- [ ] Documentation is clear and complete
- [ ] Migration path documented for existing users

## Rollout Plan

1. **Development:** Implement all changes in feature branch
2. **Testing:** Run full test suite, manual testing with real CLI
3. **Documentation:** Update all docs before release
4. **Release:** New major version (v1.0.0) due to breaking changes
5. **Announcement:** Blog post explaining benefits and migration

## Estimated Effort

- **Core implementation:** 2-3 hours
  - gemini_client.py rewrite: 1 hour
  - Test updates: 1 hour
  - Testing & debugging: 1 hour

- **Documentation:** 1 hour
  - README updates
  - .env.example
  - Migration guide

- **Total:** 3-4 hours

## Risk Mitigation

### Risk: CLI interface changes
- **Mitigation:** Version check on startup, document supported versions
- **Fallback:** Pin to known-working CLI version in docs

### Risk: CLI performance issues
- **Mitigation:** Timeouts, async execution
- **Impact:** Low - workflow tasks are already slow

### Risk: Authentication issues
- **Mitigation:** Clear error messages, troubleshooting guide
- **Testing:** Test error paths thoroughly

## Conclusion

This CLI-only approach is **significantly simpler** than the current API implementation:

- **Fewer dependencies** (no google-genai)
- **Simpler auth** (reuse CLI credentials)
- **Zero API costs** (user's main requirement)
- **90% code reuse** (only ~100 lines change)
- **Fast implementation** (3-4 hours estimated)

The trade-off of minor CLI overhead is acceptable for workflow tasks that take seconds anyway. The user gets what they want: **zero additional API costs** while keeping all MCP server functionality.
