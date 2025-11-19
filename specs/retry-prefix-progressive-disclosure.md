# Implementation Specification: Retry Mechanism, Command Prefixing, and Progressive Tool Disclosure

**Version:** 1.0
**Date:** 2025-11-19
**Status:** Design Document - Not Yet Implemented

## Executive Summary

This document specifies three enhancements to the Gemini Workflow Bridge MCP server:

1. **Retry Mechanism**: Centralized retry logic with exponential backoff for Gemini CLI calls
2. **Command Prefix System**: Namespacing for generated commands/workflows to prevent clashes
3. **Progressive Tool Disclosure**: On-demand tool discovery to reduce initial token overhead

**User Decisions:**
- ✅ 3 retry attempts (default)
- ✅ Prefix enabled by default (`gemini-`)
- ✅ Implement progressive disclosure for all tools
- ✅ Focus: Phase 1 only (retry + prefix + progressive disclosure)

---

## Table of Contents

1. [Feature 1: Centralized Retry Mechanism](#feature-1-centralized-retry-mechanism)
2. [Feature 2: Command Prefix System](#feature-2-command-prefix-system)
3. [Feature 3: Progressive Tool Disclosure](#feature-3-progressive-tool-disclosure)
4. [Integration & Dependencies](#integration--dependencies)
5. [Testing Strategy](#testing-strategy)
6. [Migration & Backward Compatibility](#migration--backward-compatibility)
7. [Configuration Reference](#configuration-reference)
8. [Implementation Checklist](#implementation-checklist)

---

# Feature 1: Centralized Retry Mechanism

## Problem Statement

Currently, Gemini CLI calls have no retry logic. Any transient failure (rate limits, network issues, temporary service unavailability) immediately propagates to the user, creating a poor experience. The original diagnostic report showed "Gemini CLI error: ^" which could have been avoided with proper retry handling.

## Goals

1. **Automatic Recovery**: Retry transient failures without user intervention
2. **Smart Backoff**: Use exponential backoff to avoid overwhelming the service
3. **Centralized Logic**: Single implementation used by all tools (DRY principle)
4. **Configurable**: Allow users to tune retry behavior via environment variables
5. **Transparent**: Tools don't need to know about retry logic
6. **Observable**: Track retry statistics for monitoring

## Non-Goals

- Retry non-retryable errors (authentication, invalid requests)
- Infinite retries (must have max limit)
- Retry logic in individual tools (must be centralized)

## Technical Design

### 1.1 Retry Configuration Class

**Location:** `hitoshura25_gemini_workflow_bridge/utils/retry.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Callable, Any
import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff"""

    # Retry limits
    max_attempts: int = 3  # Total attempts (initial + 2 retries)

    # Backoff configuration
    initial_delay: float = 1.0  # Seconds
    max_delay: float = 60.0     # Seconds
    exponential_base: float = 2.0  # Multiplier for each retry

    # Jitter to prevent thundering herd
    jitter: bool = True
    jitter_range: float = 0.2  # ±20% randomness

    # Error classification
    retryable_error_patterns: List[str] = None
    non_retryable_error_patterns: List[str] = None

    # Feature flag
    enabled: bool = True

    def __post_init__(self):
        """Set default error patterns if not provided"""
        if self.retryable_error_patterns is None:
            self.retryable_error_patterns = [
                "rate limit",
                "quota",
                "timeout",
                "timed out",
                "connection",
                "temporarily unavailable",
                "service unavailable",
                "too many requests",
                "502",  # Bad Gateway
                "503",  # Service Unavailable
                "504",  # Gateway Timeout
            ]

        if self.non_retryable_error_patterns is None:
            self.non_retryable_error_patterns = [
                "authentication",
                "unauthorized",
                "invalid api key",
                "permission denied",
                "invalid request",
                "bad request",
                "not found",
                "400",  # Bad Request
                "401",  # Unauthorized
                "403",  # Forbidden
                "404",  # Not Found
            ]

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)

        Formula: delay = min(initial_delay * (base ^ attempt), max_delay)
        With optional jitter: delay ± (delay * jitter_range)

        Examples with defaults (initial=1.0, base=2.0, max=60.0):
        - Attempt 0: 1.0s
        - Attempt 1: 2.0s
        - Attempt 2: 4.0s
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)  # Never negative

    def is_retryable(self, error_message: str) -> bool:
        """Determine if error should be retried based on message

        Logic:
        1. If matches non_retryable pattern: False (fail fast)
        2. If matches retryable pattern: True (retry)
        3. Otherwise: False (conservative - don't retry unknown errors)
        """
        error_lower = error_message.lower()

        # Check non-retryable first (fail fast)
        for pattern in self.non_retryable_error_patterns:
            if pattern.lower() in error_lower:
                logger.debug(f"Error matched non-retryable pattern: {pattern}")
                return False

        # Check retryable
        for pattern in self.retryable_error_patterns:
            if pattern.lower() in error_lower:
                logger.debug(f"Error matched retryable pattern: {pattern}")
                return True

        # Default: don't retry unknown errors (conservative)
        logger.debug("Error did not match any pattern, not retrying")
        return False


@dataclass
class RetryStatistics:
    """Track retry statistics for monitoring and debugging"""

    total_calls: int = 0
    total_retries: int = 0
    total_successes: int = 0
    total_failures: int = 0
    by_error_type: dict = None  # Dict[str, int] - count by error pattern

    def __post_init__(self):
        if self.by_error_type is None:
            self.by_error_type = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_calls == 0:
            return 0.0
        return self.total_successes / self.total_calls

    @property
    def average_retries(self) -> float:
        """Average retries per call"""
        if self.total_calls == 0:
            return 0.0
        return self.total_retries / self.total_calls

    def record_call(self, success: bool, retries: int, error_type: Optional[str] = None):
        """Record a call result"""
        self.total_calls += 1
        self.total_retries += retries

        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
            if error_type:
                self.by_error_type[error_type] = self.by_error_type.get(error_type, 0) + 1

    def to_dict(self) -> dict:
        """Export statistics as dictionary"""
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.success_rate,
            "average_retries": self.average_retries,
            "errors_by_type": self.by_error_type
        }


class RetryableError(Exception):
    """Exception that can be retried"""
    pass


class NonRetryableError(Exception):
    """Exception that should not be retried"""
    pass


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    operation_name: str = "operation",
    **kwargs
) -> Any:
    """Execute async function with retry logic and exponential backoff

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        operation_name: Name for logging purposes
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        NonRetryableError: If error is not retryable
        RetryableError: If max retries exceeded

    Example:
        result = await retry_async(
            gemini_client._generate_content_impl,
            prompt="What is 2+2?",
            config=RetryConfig(max_attempts=3),
            operation_name="generate_content"
        )
    """
    if config is None:
        config = RetryConfig()

    if not config.enabled:
        # Retry disabled, execute directly
        return await func(*args, **kwargs)

    last_exception = None
    retry_count = 0

    for attempt in range(config.max_attempts):
        try:
            logger.debug(f"{operation_name}: Attempt {attempt + 1}/{config.max_attempts}")
            result = await func(*args, **kwargs)

            if retry_count > 0:
                logger.info(
                    f"{operation_name}: Succeeded after {retry_count} retry(ies)"
                )

            return result

        except Exception as e:
            last_exception = e
            error_message = str(e)

            # Check if we should retry
            is_retryable = config.is_retryable(error_message)
            is_last_attempt = (attempt == config.max_attempts - 1)

            if not is_retryable:
                logger.warning(
                    f"{operation_name}: Non-retryable error on attempt {attempt + 1}: {error_message}"
                )
                raise NonRetryableError(f"Non-retryable error: {error_message}") from e

            if is_last_attempt:
                logger.error(
                    f"{operation_name}: Failed after {config.max_attempts} attempts. "
                    f"Last error: {error_message}"
                )
                raise RetryableError(
                    f"Operation failed after {config.max_attempts} attempts. "
                    f"Last error: {error_message}"
                ) from e

            # Calculate delay and retry
            retry_count += 1
            delay = config.calculate_delay(attempt)

            logger.warning(
                f"{operation_name}: Retryable error on attempt {attempt + 1}: {error_message}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise RetryableError(
        f"Operation failed after {config.max_attempts} attempts"
    ) from last_exception
```

### 1.2 Integration into GeminiClient

**Location:** `hitoshura25_gemini_workflow_bridge/gemini_client.py`

**Changes:**

```python
# Add imports
from .utils.retry import retry_async, RetryConfig, RetryStatistics, RetryableError, NonRetryableError

class GeminiClient:
    """Wrapper for Gemini CLI with caching, context management, and retry logic"""

    def __init__(self, model: str = "auto", retry_config: Optional[RetryConfig] = None):
        # ... existing validation ...

        self.model_name = model
        self.cli_path = cli_path

        # Initialize retry configuration
        if retry_config is None:
            retry_config = self._load_retry_config()
        self.retry_config = retry_config

        # Initialize retry statistics
        self.retry_stats = RetryStatistics()

        # Initialize cache manager (existing)
        ttl_minutes = int(os.getenv("CONTEXT_CACHE_TTL_MINUTES", "30"))
        self.cache_manager = ContextCacheManager(ttl_minutes=ttl_minutes)

    @staticmethod
    def _load_retry_config() -> RetryConfig:
        """Load retry configuration from environment variables"""
        return RetryConfig(
            max_attempts=int(os.getenv("GEMINI_RETRY_MAX_ATTEMPTS", "3")),
            initial_delay=float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "1.0")),
            max_delay=float(os.getenv("GEMINI_RETRY_MAX_DELAY", "60.0")),
            exponential_base=float(os.getenv("GEMINI_RETRY_BASE", "2.0")),
            enabled=os.getenv("GEMINI_RETRY_ENABLED", "true").lower() == "true"
        )

    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate content with Gemini CLI (with retry logic)

        This is the public interface. It wraps _generate_content_impl with retry logic.

        Note: temperature and max_tokens are not currently supported by CLI
        and are included for interface compatibility only.
        """
        retry_count = 0
        success = False
        error_type = None

        try:
            result = await retry_async(
                self._generate_content_impl,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                config=self.retry_config,
                operation_name="gemini_generate_content"
            )
            success = True
            return result

        except NonRetryableError as e:
            # Non-retryable error (auth, invalid request, etc.)
            error_type = "non_retryable"
            logger.error(f"Non-retryable error in Gemini CLI: {e}")
            raise RuntimeError(str(e)) from e

        except RetryableError as e:
            # Max retries exceeded
            retry_count = self.retry_config.max_attempts - 1
            error_type = "max_retries_exceeded"
            logger.error(f"Max retries exceeded for Gemini CLI: {e}")
            raise RuntimeError(str(e)) from e

        except Exception as e:
            # Unexpected error
            error_type = "unexpected"
            logger.error(f"Unexpected error in Gemini CLI: {e}")
            raise

        finally:
            # Record statistics
            self.retry_stats.record_call(
                success=success,
                retries=retry_count,
                error_type=error_type
            )

    async def _generate_content_impl(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Internal implementation of generate_content (without retry logic)

        This is the actual implementation that gets retried. It contains all the
        existing logic from the current generate_content method.
        """
        # Build command (prompt passed via stdin, not as argument)
        cmd = [
            self.cli_path,
            "--output-format", "json",
            "-m", self.model_name
        ]

        try:
            # Execute CLI command asynchronously with stdin support
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout (5 minutes)
            # Pass prompt via stdin to avoid shell escaping issues
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=prompt.encode('utf-8')),
                    timeout=300.0  # 5 minutes
                )
            except asyncio.TimeoutError:
                await process.kill()
                await process.wait()
                # Use specific error message for retry logic
                raise RuntimeError("Gemini CLI request timed out after 5 minutes")

            # ... rest of existing implementation (error handling, JSON parsing, etc.) ...
            # [Keep all existing code from current generate_content]

        except Exception as e:
            # Re-raise with clear error message for retry classification
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error calling Gemini CLI: {e}")

    def get_retry_stats(self) -> dict:
        """Get retry statistics for monitoring

        Returns:
            Dictionary with retry statistics including:
            - total_calls: Total number of generate_content calls
            - total_retries: Total retries across all calls
            - success_rate: Success rate (0.0 to 1.0)
            - average_retries: Average retries per call
            - errors_by_type: Breakdown of errors by type
        """
        return self.retry_stats.to_dict()
```

### 1.3 Configuration

**Environment Variables:**

```bash
# Retry Configuration (all optional, defaults shown)
GEMINI_RETRY_ENABLED=true                # Enable/disable retry mechanism
GEMINI_RETRY_MAX_ATTEMPTS=3              # Total attempts (initial + retries)
GEMINI_RETRY_INITIAL_DELAY=1.0           # Initial delay in seconds
GEMINI_RETRY_MAX_DELAY=60.0              # Maximum delay in seconds
GEMINI_RETRY_BASE=2.0                    # Exponential backoff base
```

**Retry Behavior Examples:**

With defaults (max_attempts=3, initial_delay=1.0, base=2.0):

| Attempt | Delay Before | Notes |
|---------|--------------|-------|
| 1       | 0s           | Initial attempt |
| 2       | ~1s          | First retry (1.0 * 2^0) |
| 3       | ~2s          | Second retry (1.0 * 2^1) |
| Fail    | -            | Max retries exceeded |

**Total time for 3 attempts:** ~3 seconds + execution time

### 1.4 Error Classification

**Retryable Errors** (will retry):
- Rate limit / quota exceeded
- Timeout / connection issues
- Service temporarily unavailable
- HTTP 502, 503, 504

**Non-Retryable Errors** (fail immediately):
- Authentication / authorization failures
- Invalid API key / permissions
- Invalid request / bad parameters
- HTTP 400, 401, 403, 404

### 1.5 Monitoring & Observability

**Get Statistics:**

```python
# In any tool or for debugging
gemini_client = GeminiClient()
stats = gemini_client.get_retry_stats()

# Returns:
{
    "total_calls": 100,
    "total_retries": 15,
    "total_successes": 95,
    "total_failures": 5,
    "success_rate": 0.95,
    "average_retries": 0.15,
    "errors_by_type": {
        "non_retryable": 3,
        "max_retries_exceeded": 2
    }
}
```

**Logging:**

All retry attempts are logged with appropriate levels:
- DEBUG: Each attempt
- WARNING: Retryable errors with retry info
- ERROR: Non-retryable errors or max retries exceeded
- INFO: Success after retries

### 1.6 Testing Requirements

**Unit Tests:**

1. **Retry Logic:**
   - Test successful operation (no retry)
   - Test single retry (fails once, succeeds on second attempt)
   - Test max retries exceeded (fails all attempts)
   - Test non-retryable error (fails immediately)

2. **Backoff Calculation:**
   - Test exponential backoff formula
   - Test jitter application
   - Test max delay cap

3. **Error Classification:**
   - Test retryable error patterns
   - Test non-retryable error patterns
   - Test unknown error handling (should not retry)

4. **Statistics:**
   - Test recording of call results
   - Test success rate calculation
   - Test average retries calculation

**Integration Tests:**

1. Simulate Gemini CLI rate limit (mock subprocess)
2. Simulate network timeout
3. Simulate authentication error (should not retry)
4. Test actual Gemini CLI with retry

**Edge Cases:**

1. Retry disabled (GEMINI_RETRY_ENABLED=false)
2. Zero max attempts (should fail immediately)
3. Very large max delay
4. Negative delays (should clamp to 0)

---

# Feature 2: Command Prefix System

## Problem Statement

Generated commands and workflows use bare names like `spec-only.md`, `feature.md` which can clash with:
- Other MCP servers' commands
- Claude Code's built-in commands
- User's custom commands

There's no way to identify which commands belong to the Gemini MCP server.

## Goals

1. **Namespace Protection**: Prefix all generated commands to prevent clashes
2. **Clear Ownership**: Users can easily identify Gemini commands
3. **Configurable**: Allow users to customize or disable prefix
4. **Backward Compatible**: Existing commands without prefix still work
5. **Consistent**: Apply to both workflows and slash commands
6. **Default Enabled**: Professional namespacing by default

## Non-Goals

- Rename existing commands automatically (manual migration)
- Support for multiple prefixes per server
- Prefix for MCP tool names (only for generated files)

## Technical Design

### 2.1 Prefix Configuration

**Location:** `hitoshura25_gemini_workflow_bridge/resources.py`

```python
class WorkflowResources:
    """Manage workflow resources (specs, reviews, context) and configuration"""

    def __init__(self):
        # Existing directory configuration
        self.specs_dir = Path(os.getenv("DEFAULT_SPEC_DIR", "./specs"))
        self.reviews_dir = Path(os.getenv("DEFAULT_REVIEW_DIR", "./reviews"))
        self.context_dir = Path(os.getenv("DEFAULT_CONTEXT_DIR", "./.workflow-context"))

        # NEW: Prefix configuration
        self.command_prefix = os.getenv("GEMINI_COMMAND_PREFIX", "gemini-")
        self.workflow_prefix = os.getenv("GEMINI_WORKFLOW_PREFIX", "gemini-")

        # Validate prefixes (optional, for safety)
        self._validate_prefix(self.command_prefix, "GEMINI_COMMAND_PREFIX")
        self._validate_prefix(self.workflow_prefix, "GEMINI_WORKFLOW_PREFIX")

        # Ensure directories exist
        self.specs_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)
        self.context_dir.mkdir(exist_ok=True)

    @staticmethod
    def _validate_prefix(prefix: str, env_var: str) -> None:
        """Validate prefix format (optional safety check)

        Rules:
        - Can be empty (disabled)
        - Should be alphanumeric with dashes/underscores
        - Should not contain spaces or special characters
        - Recommend ending with dash for readability
        """
        if prefix == "":
            # Empty prefix is allowed (disables prefixing)
            return

        # Warn about potentially problematic prefixes
        if " " in prefix:
            logger.warning(
                f"{env_var}='{prefix}' contains spaces. "
                f"This may cause issues with command parsing."
            )

        if not prefix.endswith("-") and not prefix.endswith("_"):
            logger.info(
                f"{env_var}='{prefix}' does not end with '-' or '_'. "
                f"Consider adding a separator for better readability "
                f"(e.g., 'gemini-' instead of 'gemini')."
            )


# Create singleton instance for import
workflow_resources = WorkflowResources()
```

**Environment Variables:**

```bash
GEMINI_COMMAND_PREFIX=gemini-        # Default: "gemini-"
GEMINI_WORKFLOW_PREFIX=gemini-       # Default: "gemini-"
```

**Why Separate Prefixes?**
- Allows different prefixes for commands vs workflows (if desired)
- Most users will use the same prefix for both
- Future flexibility

### 2.2 Update setup_workflows.py

**Location:** `hitoshura25_gemini_workflow_bridge/tools/setup_workflows.py`

**Changes:**

```python
from ..resources import workflow_resources

async def setup_workflows(
    workflows: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    include_commands: bool = True,
    command_prefix: Optional[str] = None,  # NEW: Override default prefix
    workflow_prefix: Optional[str] = None   # NEW: Override default prefix
) -> Dict[str, Any]:
    """Set up recommended workflow files and slash commands for the Gemini MCP Server.

    Args:
        workflows: List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']
        output_dir: Base directory for outputs
        overwrite: Whether to overwrite existing files
        include_commands: Whether to also create slash commands
        command_prefix: Command prefix (default from GEMINI_COMMAND_PREFIX env var)
        workflow_prefix: Workflow prefix (default from GEMINI_WORKFLOW_PREFIX env var)

    Returns:
        Dictionary with success status, workflows_created, skipped items, and message
    """
    # Load prefixes (use parameter if provided, otherwise env var)
    cmd_prefix = command_prefix if command_prefix is not None else workflow_resources.command_prefix
    wf_prefix = workflow_prefix if workflow_prefix is not None else workflow_resources.workflow_prefix

    # ... existing setup code ...

    # Process each workflow
    for workflow_name in workflows:
        if workflow_name not in WORKFLOW_TEMPLATES:
            # ... existing validation ...
            continue

        template = WORKFLOW_TEMPLATES[workflow_name]

        # Apply prefixes to filenames
        workflow_filename = f"{wf_prefix}{workflow_name}.md"
        command_filename = f"{cmd_prefix}{workflow_name}.md"

        workflow_path = workflow_dir / workflow_filename
        command_path = command_dir / command_filename if include_commands else None

        workflow_result = {
            "name": workflow_name,
            "prefixed_name": f"{wf_prefix}{workflow_name}",
            "workflow_path": str(workflow_path.relative_to(base_dir)),
            "command_path": str(command_path.relative_to(base_dir)) if command_path else None,
            "status": "created"
        }

        # ... check if exists ...

        # Get template content
        workflow_content = template["workflow_content"]
        command_content = template["command_content"]

        # Replace command name placeholders in templates
        # The command name used in slash commands should include the prefix
        prefixed_command = f"{cmd_prefix}{workflow_name}"
        command_content = command_content.replace("/{COMMAND_NAME}", f"/{prefixed_command}")
        command_content = command_content.replace("{COMMAND_NAME}", prefixed_command)

        # Create workflow file
        try:
            workflow_path.write_text(workflow_content)
        except (PermissionError, OSError) as e:
            # ... existing error handling ...
            continue

        # Create command file if requested
        if include_commands and command_path:
            if command_path.exists() and not overwrite:
                workflow_result["status"] = "workflow created, command skipped (already exists)"
            else:
                try:
                    command_path.write_text(command_content)
                except (PermissionError, OSError) as e:
                    workflow_result["status"] = f"workflow created, command failed: {str(e)}"

        results["workflows_created"].append(workflow_result)

    # ... existing result generation ...

    return results
```

### 2.3 Update generate_command.py

**Location:** `hitoshura25_gemini_workflow_bridge/tools/generate_command.py`

**Changes:**

```python
from ..resources import workflow_resources

async def generate_slash_command(
    command_name: str,
    workflow_type: Literal["feature", "refactor", "debug", "review", "custom"],
    description: str,
    steps: Optional[List[str]] = None,
    save_to: Optional[str] = None,
    prefix: Optional[str] = None  # NEW: Override default prefix
) -> Dict[str, Any]:
    """Auto-generate Claude Code slash commands for common workflows.

    Args:
        command_name: Base name of the command (e.g., "add-feature")
        workflow_type: Type of workflow
        description: Description of what the command does
        steps: Custom steps if workflow_type="custom"
        save_to: Where to save command file
        prefix: Command prefix (default from GEMINI_COMMAND_PREFIX env var)

    Returns:
        Dictionary with command_path, command_content, usage_example, and prefixed_name
    """
    # Load prefix (use parameter if provided, otherwise env var)
    cmd_prefix = prefix if prefix is not None else workflow_resources.command_prefix

    # Apply prefix to command name
    prefixed_command_name = f"{cmd_prefix}{command_name}"

    # Get template
    if workflow_type == "custom" and steps:
        custom_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        template = command_templates["custom"].format(
            command_name=prefixed_command_name,  # Use prefixed name
            description=description,
            custom_steps=custom_steps
        )
    else:
        template = command_templates[workflow_type].format(
            command_name=prefixed_command_name,  # Use prefixed name
            description=description
        )

    # Determine save path (use prefixed filename)
    if not save_to:
        command_dir = Path(os.getenv("DEFAULT_COMMAND_DIR", "./.claude/commands"))
        command_dir.mkdir(parents=True, exist_ok=True)
        save_to = str(command_dir / f"{prefixed_command_name}.md")

    # Save command
    command_file = Path(save_to)
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(template)

    # Generate usage example with prefix
    usage_example = f"/{prefixed_command_name} <description>"

    return {
        "command_path": str(save_to),
        "command_content": template,
        "usage_example": usage_example,
        "prefixed_name": prefixed_command_name,
        "base_name": command_name
    }
```

### 2.4 Update workflow_templates.py

**Location:** `hitoshura25_gemini_workflow_bridge/tools/workflow_templates.py`

**Changes:**

All templates should use `{COMMAND_NAME}` placeholder which gets replaced with the prefixed name:

```python
WORKFLOW_TEMPLATES = {
    "spec-only": {
        "workflow_content": """# Specification-Only Workflow

## Purpose
Create a detailed specification document without implementation.

## Steps

### 1. Gather Facts
Use `query_codebase_tool` to collect factual information:
- Existing patterns and conventions
- Related components and their interfaces
- Dependencies and integration points
- Current architecture decisions

### 2. Create Specification
Write a comprehensive specification including:
- Problem statement and goals
- Technical design and architecture
- API contracts and interfaces
- Data models and schemas
- Integration points
- Security considerations
- Testing strategy

### 3. Validate Specification
Use `validate_against_codebase_tool` with checks:
- missing_files
- undefined_dependencies
- pattern_violations
- completeness

### 4. Refine and Iterate
Address validation issues:
- Add missing details
- Resolve inconsistencies
- Clarify ambiguities
- Update based on codebase facts

### 5. Save Specification
Save to `specs/` directory with descriptive filename.
""",
        "command_content": """# /{COMMAND_NAME} - Create specification document only (no implementation)

## Usage
/{COMMAND_NAME} <feature_description>

## Description
Creates a detailed specification document for a feature without implementing it. Uses the Gemini MCP Server to gather facts about the codebase and validate the specification for completeness.

## Steps
1. Use query_codebase_tool to gather facts about relevant codebase areas
2. Create detailed specification document using the facts
3. Use validate_against_codebase_tool to check completeness
4. Address any validation issues
5. Save specification to specs/ directory

## Example
/{COMMAND_NAME} Add user authentication with OAuth2 support

## Notes
- This command ONLY creates a specification, it does not implement code
- The specification will be saved in the `specs/` directory
- Use this before implementation to ensure clear requirements
""",
        "description": "Spec-only workflow for creating specifications without implementation"
    },

    # Similar updates for "feature", "refactor", "review" templates...
    # All should use {COMMAND_NAME} placeholder
}
```

### 2.5 Configuration Examples

**Example 1: Default (Recommended)**
```bash
# No configuration needed, uses defaults
# Commands created: /gemini-spec-only, /gemini-feature, etc.
# Files: .claude/commands/gemini-spec-only.md
```

**Example 2: Custom Prefix**
```bash
export GEMINI_COMMAND_PREFIX="gw-"
export GEMINI_WORKFLOW_PREFIX="gw-"
# Commands created: /gw-spec-only, /gw-feature, etc.
# Files: .claude/commands/gw-spec-only.md
```

**Example 3: Disable Prefix**
```bash
export GEMINI_COMMAND_PREFIX=""
export GEMINI_WORKFLOW_PREFIX=""
# Commands created: /spec-only, /feature, etc. (legacy behavior)
# Files: .claude/commands/spec-only.md
# WARNING: May clash with other commands!
```

**Example 4: Different Prefixes**
```bash
export GEMINI_COMMAND_PREFIX="gem-"
export GEMINI_WORKFLOW_PREFIX="workflow-"
# Commands: /gem-spec-only
# Workflows: workflow-spec-only.md
```

### 2.6 Migration Guide

**For Existing Installations:**

If you already have commands without prefixes (from before this feature):

**Option 1: Rename Manually (Recommended)**
```bash
cd .claude/commands
mv spec-only.md gemini-spec-only.md
mv feature.md gemini-feature.md
# etc.
```

**Option 2: Disable Prefix (Not Recommended)**
```bash
export GEMINI_COMMAND_PREFIX=""
export GEMINI_WORKFLOW_PREFIX=""
```

**Option 3: Keep Both (Transition Period)**
- Old commands continue to work
- New commands created with prefix
- Gradually migrate to prefixed versions

### 2.7 Testing Requirements

**Unit Tests:**

1. **Prefix Application:**
   - Test with default prefix
   - Test with custom prefix
   - Test with empty prefix (disabled)
   - Test with different command/workflow prefixes

2. **File Creation:**
   - Verify correct filenames with prefix
   - Verify template replacement of {COMMAND_NAME}
   - Verify usage examples include prefix

3. **Edge Cases:**
   - Very long prefix
   - Special characters in prefix (should warn)
   - Prefix with/without trailing dash

**Integration Tests:**

1. Create workflow with setup_workflows_tool
2. Verify files created with correct prefix
3. Verify command content uses prefixed names
4. Test override via parameter vs env var

---

# Feature 3: Progressive Tool Disclosure

## Problem Statement

Currently, all MCP tools are exposed upfront in `list_tools()`. For a server with 11+ tools, this creates:
- High initial token overhead (tool schemas for all tools)
- Cognitive overload for Claude
- Inefficient when only 1-2 tools are needed

Anthropic's article recommends "progressive disclosure" - discover tools on-demand rather than exposing all upfront.

## Goals

1. **Reduce Initial Overhead**: Minimal tools exposed at initialization
2. **On-Demand Discovery**: Allow Claude to search for tools by intent/keyword
3. **Flexible Detail Levels**: Return names only, descriptions, or full schemas
4. **Maintain Compatibility**: Existing tool call patterns still work
5. **Improve Efficiency**: Only load schemas for tools that will be used

## Non-Goals

- Remove direct tool access (tools are still callable directly)
- Force all tool calls through search (search is optional)
- Change existing tool implementations

## Technical Design

### 3.1 Tool Organization & Metadata

**Location:** `hitoshura25_gemini_workflow_bridge/tool_registry.py` (NEW FILE)

```python
"""Central registry for all MCP tools with metadata for progressive disclosure"""

from typing import Dict, List, Optional, TypedDict
from enum import Enum


class ToolCategory(Enum):
    """Categories for organizing tools"""
    FACT_EXTRACTION = "fact_extraction"  # Query, analyze, find
    VALIDATION = "validation"             # Validate, check consistency
    GENERATION = "generation"             # Generate workflows, commands
    WORKFLOW = "workflow"                 # Setup workflows
    GENERAL = "general"                   # Ask Gemini


class ToolMetadata(TypedDict):
    """Metadata for a tool to enable search and progressive disclosure"""
    name: str
    category: ToolCategory
    short_description: str  # One-line summary (for search results)
    keywords: List[str]     # Searchable keywords
    use_cases: List[str]    # Common use cases (for search)
    complexity: str         # "simple", "moderate", "complex"
    requires_codebase: bool # Whether tool needs codebase context


# Tool Registry: Maps tool names to their metadata
TOOL_REGISTRY: Dict[str, ToolMetadata] = {
    "query_codebase_tool": {
        "name": "query_codebase_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Answer multiple questions about codebase with compressed facts",
        "keywords": [
            "query", "questions", "facts", "codebase", "analyze",
            "find", "search", "multiple questions", "compression"
        ],
        "use_cases": [
            "Answer specific questions about codebase structure",
            "Gather facts for specification creation",
            "Understand existing patterns and conventions",
            "Find relevant files and components"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "find_code_by_intent_tool": {
        "name": "find_code_by_intent_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Semantic search for code by natural language intent",
        "keywords": [
            "find", "search", "semantic", "intent", "code",
            "locate", "discover", "natural language"
        ],
        "use_cases": [
            "Find code by describing what it does",
            "Locate APIs or endpoints by intent",
            "Discover relevant files for a feature",
            "Search by functionality rather than exact names"
        ],
        "complexity": "simple",
        "requires_codebase": True
    },

    "trace_feature_tool": {
        "name": "trace_feature_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Trace execution flow of a feature through the codebase",
        "keywords": [
            "trace", "flow", "feature", "execution", "path",
            "follow", "dependencies", "data flow"
        ],
        "use_cases": [
            "Understand how a feature is implemented",
            "Trace data flow through the system",
            "Find dependencies for refactoring",
            "Map execution paths"
        ],
        "complexity": "complex",
        "requires_codebase": True
    },

    "list_error_patterns_tool": {
        "name": "list_error_patterns_tool",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "Extract and categorize patterns (error handling, logging, etc.)",
        "keywords": [
            "patterns", "error handling", "logging", "async",
            "database", "conventions", "consistency"
        ],
        "use_cases": [
            "Find error handling patterns in codebase",
            "Discover logging conventions",
            "Identify async patterns",
            "Analyze database query patterns"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "validate_against_codebase_tool": {
        "name": "validate_against_codebase_tool",
        "category": ToolCategory.VALIDATION,
        "short_description": "Validate specification for completeness and accuracy",
        "keywords": [
            "validate", "spec", "specification", "completeness",
            "accuracy", "check", "verify", "review"
        ],
        "use_cases": [
            "Validate specification against codebase",
            "Check for missing dependencies",
            "Verify spec completeness",
            "Ensure spec aligns with existing patterns"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "check_consistency_tool": {
        "name": "check_consistency_tool",
        "category": ToolCategory.VALIDATION,
        "short_description": "Verify code/spec follows existing codebase patterns",
        "keywords": [
            "consistency", "patterns", "conventions", "naming",
            "style", "standards", "verify", "check"
        ],
        "use_cases": [
            "Check if new code follows conventions",
            "Verify naming consistency",
            "Validate against existing patterns",
            "Ensure architectural alignment"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "generate_feature_workflow_tool": {
        "name": "generate_feature_workflow_tool",
        "category": ToolCategory.GENERATION,
        "short_description": "Generate complete executable workflow for a feature",
        "keywords": [
            "generate", "workflow", "feature", "create",
            "automated", "template", "steps"
        ],
        "use_cases": [
            "Generate feature implementation workflow",
            "Create step-by-step guides",
            "Automate feature development process",
            "Generate refactoring workflows"
        ],
        "complexity": "simple",
        "requires_codebase": True
    },

    "generate_slash_command_tool": {
        "name": "generate_slash_command_tool",
        "category": ToolCategory.GENERATION,
        "short_description": "Auto-generate Claude Code slash commands",
        "keywords": [
            "generate", "slash command", "command", "create",
            "custom", "workflow", "automation"
        ],
        "use_cases": [
            "Create custom slash commands",
            "Automate common workflows",
            "Generate command templates",
            "Create developer shortcuts"
        ],
        "complexity": "simple",
        "requires_codebase": False
    },

    "setup_workflows_tool": {
        "name": "setup_workflows_tool",
        "category": ToolCategory.WORKFLOW,
        "short_description": "Set up recommended workflows and slash commands",
        "keywords": [
            "setup", "install", "initialize", "workflows",
            "commands", "configure", "bootstrap"
        ],
        "use_cases": [
            "Initial setup after MCP installation",
            "Install spec-only workflow",
            "Create recommended workflows",
            "Bootstrap command structure"
        ],
        "complexity": "simple",
        "requires_codebase": False
    },

    "analyze_codebase_with_gemini": {
        "name": "analyze_codebase_with_gemini",
        "category": ToolCategory.FACT_EXTRACTION,
        "short_description": "General-purpose codebase analysis with Gemini",
        "keywords": [
            "analyze", "analysis", "codebase", "architecture",
            "patterns", "general", "overview"
        ],
        "use_cases": [
            "Get overview of codebase architecture",
            "Understand overall patterns",
            "General-purpose analysis",
            "Architecture documentation"
        ],
        "complexity": "moderate",
        "requires_codebase": True
    },

    "ask_gemini": {
        "name": "ask_gemini",
        "category": ToolCategory.GENERAL,
        "short_description": "General-purpose Gemini query (optional codebase context)",
        "keywords": [
            "ask", "query", "question", "gemini", "general",
            "anything", "flexible"
        ],
        "use_cases": [
            "Ask any question to Gemini",
            "Get answers without codebase context",
            "General knowledge queries",
            "Flexible AI assistance"
        ],
        "complexity": "simple",
        "requires_codebase": False
    }
}


def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """Get metadata for a specific tool"""
    return TOOL_REGISTRY.get(tool_name)


def get_all_tool_names() -> List[str]:
    """Get list of all tool names"""
    return list(TOOL_REGISTRY.keys())


def get_tools_by_category(category: ToolCategory) -> List[ToolMetadata]:
    """Get all tools in a specific category"""
    return [
        metadata for metadata in TOOL_REGISTRY.values()
        if metadata["category"] == category
    ]


def search_tools_by_keyword(keyword: str) -> List[ToolMetadata]:
    """Search tools by keyword (case-insensitive, partial match)"""
    keyword_lower = keyword.lower()
    results = []

    for metadata in TOOL_REGISTRY.values():
        # Check keywords
        keyword_match = any(keyword_lower in kw.lower() for kw in metadata["keywords"])

        # Check description
        desc_match = keyword_lower in metadata["short_description"].lower()

        # Check use cases
        use_case_match = any(
            keyword_lower in uc.lower() for uc in metadata["use_cases"]
        )

        if keyword_match or desc_match or use_case_match:
            results.append(metadata)

    return results
```

### 3.2 Tool Discovery Tool

**Location:** `hitoshura25_gemini_workflow_bridge/server.py`

**Add new tool:**

```python
from .tool_registry import (
    TOOL_REGISTRY,
    search_tools_by_keyword,
    get_tools_by_category,
    ToolCategory
)

# ============================================================================
# Tool Discovery (Progressive Disclosure)
# ============================================================================

@mcp.tool()
async def discover_tools(
    query: str = None,
    category: str = None,
    detail_level: str = "with_description"
) -> str:
    """Discover available MCP tools by search query or category

    This tool implements progressive disclosure - instead of exposing all tools
    upfront, Claude can discover relevant tools on-demand by intent or category.

    Args:
        query: Search query (keywords, intent, use case) - searches tool names,
               descriptions, keywords, and use cases
        category: Filter by category - options: "fact_extraction", "validation",
                 "generation", "workflow", "general"
        detail_level: How much detail to return:
                     - "name_only": Just tool names (minimal tokens)
                     - "with_description": Names + short descriptions (recommended)
                     - "with_use_cases": Names + descriptions + use cases
                     - "full": Complete tool information including keywords

    Returns:
        JSON string with discovered tools and their metadata

    Examples:
        # Find tools for codebase analysis
        discover_tools(query="analyze codebase", detail_level="with_description")

        # Find all validation tools
        discover_tools(category="validation", detail_level="with_use_cases")

        # Get all tool names only
        discover_tools(detail_level="name_only")
    """
    results = []

    # Search by query
    if query:
        # Search keywords, descriptions, use cases
        matched_tools = search_tools_by_keyword(query)
        results.extend(matched_tools)

    # Filter by category
    elif category:
        try:
            cat_enum = ToolCategory(category)
            results = get_tools_by_category(cat_enum)
        except ValueError:
            return json.dumps({
                "error": f"Invalid category: {category}",
                "valid_categories": [c.value for c in ToolCategory]
            })

    # No filter - return all tools
    else:
        results = list(TOOL_REGISTRY.values())

    # Format results based on detail level
    formatted_tools = []

    for tool in results:
        if detail_level == "name_only":
            formatted_tools.append(tool["name"])

        elif detail_level == "with_description":
            formatted_tools.append({
                "name": tool["name"],
                "description": tool["short_description"],
                "complexity": tool["complexity"]
            })

        elif detail_level == "with_use_cases":
            formatted_tools.append({
                "name": tool["name"],
                "description": tool["short_description"],
                "complexity": tool["complexity"],
                "use_cases": tool["use_cases"],
                "requires_codebase": tool["requires_codebase"]
            })

        elif detail_level == "full":
            formatted_tools.append(tool)

        else:
            return json.dumps({
                "error": f"Invalid detail_level: {detail_level}",
                "valid_levels": ["name_only", "with_description", "with_use_cases", "full"]
            })

    return json.dumps({
        "query": query,
        "category": category,
        "detail_level": detail_level,
        "tool_count": len(formatted_tools),
        "tools": formatted_tools
    }, indent=2)
```

### 3.3 Modify Tool Registration

**Location:** `hitoshura25_gemini_workflow_bridge/server.py`

**Current approach (expose all tools):**
```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="query_codebase_tool", ...),
        Tool(name="find_code_by_intent_tool", ...),
        # ... all 11 tools ...
    ]
```

**New approach (minimal + discovery):**

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose minimal essential tools + discovery tool

    Progressive disclosure pattern:
    1. Always expose discover_tools for tool discovery
    2. Always expose ask_gemini as general fallback
    3. Always expose setup_workflows_tool for initial setup
    4. Claude can discover other tools on-demand via discover_tools

    This reduces initial token overhead from ~2000 tokens (all tools)
    to ~300 tokens (core tools only).
    """
    return [
        # Tool Discovery (Progressive Disclosure)
        Tool(
            name="discover_tools",
            description="Discover available MCP tools by search query or category. "
                       "Use this to find relevant tools for your task instead of "
                       "guessing tool names. Supports search by keywords, intent, "
                       "use cases, and categories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - keywords, intent, or use case (e.g., 'analyze codebase', 'validate spec', 'generate workflow')"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact_extraction", "validation", "generation", "workflow", "general"],
                        "description": "Filter by tool category"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["name_only", "with_description", "with_use_cases", "full"],
                        "default": "with_description",
                        "description": "How much detail to return about each tool"
                    }
                }
            }
        ),

        # Essential Tools (Always Available)
        Tool(
            name="ask_gemini",
            description="General-purpose Gemini query with optional codebase context. "
                       "Use for any question or analysis task. Can optionally include "
                       "full codebase context for analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Question or task for Gemini"
                    },
                    "include_codebase_context": {
                        "type": "boolean",
                        "description": "Load full codebase context (default: false)",
                        "default": False
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for generation 0.0-1.0 (default: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["prompt"]
            }
        ),

        # Setup Tool (For Initial Configuration)
        Tool(
            name="setup_workflows_tool",
            description="Set up recommended workflow files and slash commands. "
                       "Use this after installing the MCP server to bootstrap "
                       "workflows like spec-only, feature implementation, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflows": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of workflows to set up. Options: ['spec-only', 'feature', 'refactor', 'review', 'all']. Default: ['spec-only']",
                        "default": ["spec-only"]
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Base directory for outputs"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to overwrite existing files"
                    },
                    "include_commands": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to create slash commands"
                    },
                    "command_prefix": {
                        "type": "string",
                        "description": "Command prefix (default from GEMINI_COMMAND_PREFIX env var)"
                    },
                    "workflow_prefix": {
                        "type": "string",
                        "description": "Workflow prefix (default from GEMINI_WORKFLOW_PREFIX env var)"
                    }
                }
            }
        )
    ]
```

### 3.4 Get Tool Schema On-Demand

**Location:** `hitoshura25_gemini_workflow_bridge/server.py`

**Add helper tool:**

```python
@mcp.tool()
async def get_tool_schema(tool_name: str) -> str:
    """Get the full schema for a specific tool

    After discovering a tool via discover_tools, use this to get the complete
    tool schema including all parameters, types, and descriptions.

    Args:
        tool_name: Name of the tool to get schema for

    Returns:
        JSON string with complete tool schema

    Example:
        # Discover tool first
        discover_tools(query="codebase analysis")

        # Get full schema for specific tool
        get_tool_schema("query_codebase_tool")
    """
    # Tool schema definitions (same as current implementation)
    tool_schemas = {
        "query_codebase_tool": {
            "name": "query_codebase_tool",
            "description": "Multi-question factual analysis with massive context compression. "
                          "Analyzes codebase and extracts factual information, compressing large "
                          "codebases (50K+ tokens) into small summaries (300 tokens per answer).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 1-10 specific questions to answer"
                    },
                    "scope": {
                        "type": "string",
                        "description": "Directory to analyze (default: current directory)"
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns to include"
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to exclude"
                    },
                    "max_tokens_per_answer": {
                        "type": "integer",
                        "default": 300,
                        "description": "Target token budget per answer"
                    }
                },
                "required": ["questions"]
            }
        },
        # ... schemas for all other tools ...
    }

    if tool_name not in tool_schemas:
        return json.dumps({
            "error": f"Tool not found: {tool_name}",
            "available_tools": list(tool_schemas.keys())
        })

    return json.dumps(tool_schemas[tool_name], indent=2)
```

### 3.5 Usage Examples

**Example 1: New User Explores Tools**

```
User: "I want to create a specification for a new feature"

Claude: Let me discover relevant tools for specification creation.

[Calls: discover_tools(query="specification", detail_level="with_description")]

Result:
{
  "tools": [
    {
      "name": "query_codebase_tool",
      "description": "Answer multiple questions about codebase with compressed facts",
      "complexity": "moderate"
    },
    {
      "name": "validate_against_codebase_tool",
      "description": "Validate specification for completeness and accuracy",
      "complexity": "moderate"
    }
  ]
}

Claude: I found tools for specification work. Let me get the full schema for query_codebase_tool...

[Calls: get_tool_schema("query_codebase_tool")]

Claude: Now I'll use query_codebase_tool to gather facts...

[Calls: query_codebase_tool(questions=[...])]
```

**Example 2: Find Tools by Category**

```
[Calls: discover_tools(category="validation", detail_level="with_use_cases")]

Result:
{
  "tools": [
    {
      "name": "validate_against_codebase_tool",
      "description": "Validate specification for completeness and accuracy",
      "use_cases": [
        "Validate specification against codebase",
        "Check for missing dependencies",
        "Verify spec completeness"
      ]
    },
    {
      "name": "check_consistency_tool",
      "description": "Verify code/spec follows existing codebase patterns",
      "use_cases": [
        "Check if new code follows conventions",
        "Verify naming consistency"
      ]
    }
  ]
}
```

**Example 3: List All Tools (Name Only)**

```
[Calls: discover_tools(detail_level="name_only")]

Result:
{
  "tools": [
    "query_codebase_tool",
    "find_code_by_intent_tool",
    "trace_feature_tool",
    "list_error_patterns_tool",
    "validate_against_codebase_tool",
    "check_consistency_tool",
    "generate_feature_workflow_tool",
    "generate_slash_command_tool",
    "setup_workflows_tool",
    "analyze_codebase_with_gemini",
    "ask_gemini"
  ]
}
```

### 3.6 Token Savings Analysis

**Before (Current Approach):**
- All 11 tools exposed in `list_tools()`
- Each tool schema: ~150-250 tokens
- Total: ~2000 tokens upfront
- Every conversation starts with 2000 token overhead

**After (Progressive Disclosure):**
- 3 core tools exposed (discover_tools, ask_gemini, setup_workflows_tool)
- Each tool schema: ~150-200 tokens
- Total: ~500 tokens upfront
- **Savings: ~1500 tokens (75% reduction)**

**On-Demand Loading:**
- `discover_tools` with descriptions: ~300 tokens (returns 5-6 relevant tools)
- `get_tool_schema` for 1 tool: ~200 tokens
- Total: ~500 tokens when needed

**Overall Efficiency:**
- No tools needed: 500 tokens (vs 2000 before) - 75% savings
- 1-2 tools needed: ~1000 tokens (vs 2000 before) - 50% savings
- All tools needed: ~2500 tokens (vs 2000 before) - slightly more, but rare

### 3.7 Backward Compatibility

**Option 1: Expose All Tools (Backward Compatible)**
- Add environment variable: `GEMINI_EXPOSE_ALL_TOOLS=false`
- Default: `false` (progressive disclosure)
- If `true`: Expose all tools in `list_tools()` (legacy behavior)

**Option 2: Progressive Disclosure Only (Breaking Change)**
- Force new behavior for all users
- Better efficiency by default
- Requires documentation update

**Recommendation:** Option 1 (backward compatible with opt-in to legacy behavior)

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose tools based on configuration"""
    expose_all = os.getenv("GEMINI_EXPOSE_ALL_TOOLS", "false").lower() == "true"

    if expose_all:
        # Legacy behavior: expose all tools
        return _get_all_tools()
    else:
        # Progressive disclosure: minimal + discovery
        return _get_core_tools()
```

### 3.8 Testing Requirements

**Unit Tests:**

1. **Tool Registry:**
   - Test search by keyword
   - Test filter by category
   - Test get metadata for tool

2. **discover_tools:**
   - Test search query with various keywords
   - Test category filtering
   - Test all detail levels
   - Test invalid inputs

3. **get_tool_schema:**
   - Test valid tool names
   - Test invalid tool names
   - Test schema completeness

**Integration Tests:**

1. Tool discovery workflow
2. Progressive loading of tools
3. Backward compatibility mode

---

# Integration & Dependencies

## File Dependencies

### New Files:
1. `hitoshura25_gemini_workflow_bridge/utils/retry.py`
2. `hitoshura25_gemini_workflow_bridge/tool_registry.py`

### Modified Files:
1. `hitoshura25_gemini_workflow_bridge/gemini_client.py`
2. `hitoshura25_gemini_workflow_bridge/tools/setup_workflows.py`
3. `hitoshura25_gemini_workflow_bridge/tools/generate_command.py`
4. `hitoshura25_gemini_workflow_bridge/tools/workflow_templates.py`
5. `hitoshura25_gemini_workflow_bridge/resources.py`
6. `hitoshura25_gemini_workflow_bridge/server.py`
7. `hitoshura25_gemini_workflow_bridge/utils/__init__.py`

## Feature Interactions

### Retry + Prefix:
- Independent features, no interaction
- Retry works transparently for prefix system

### Retry + Progressive Disclosure:
- Independent features
- Retry applies to all tools regardless of discovery method

### Prefix + Progressive Disclosure:
- `discover_tools` returns tool names (which don't have prefixes)
- `setup_workflows_tool` (exposed by default) includes prefix parameters
- Tool names ≠ command names (tools stay unprefixed, commands get prefixed)

## Configuration Interactions

All features use environment variables:
- `GEMINI_RETRY_*` - Retry configuration
- `GEMINI_COMMAND_PREFIX` - Command prefix
- `GEMINI_WORKFLOW_PREFIX` - Workflow prefix
- `GEMINI_EXPOSE_ALL_TOOLS` - Progressive disclosure toggle

No conflicts between configurations.

---

# Testing Strategy

## Unit Tests

### Retry Mechanism:
- `test_retry_config_defaults()`
- `test_retry_config_from_env()`
- `test_calculate_delay()`
- `test_is_retryable()`
- `test_retry_statistics()`
- `test_retry_async_success()`
- `test_retry_async_transient_failure()`
- `test_retry_async_non_retryable()`
- `test_retry_async_max_attempts()`

### Prefix System:
- `test_prefix_application()`
- `test_empty_prefix()`
- `test_custom_prefix()`
- `test_different_prefixes()`
- `test_template_replacement()`

### Progressive Disclosure:
- `test_tool_registry_search()`
- `test_tool_registry_category()`
- `test_discover_tools_query()`
- `test_discover_tools_category()`
- `test_discover_tools_detail_levels()`
- `test_get_tool_schema()`

## Integration Tests

1. **End-to-End Retry:**
   - Mock Gemini CLI to return rate limit error on first call
   - Verify retry happens
   - Verify success on second call

2. **End-to-End Prefix:**
   - Call `setup_workflows_tool` with default prefix
   - Verify files created with correct names
   - Verify command content uses prefixed names

3. **End-to-End Progressive Disclosure:**
   - Call `discover_tools` to find relevant tools
   - Call `get_tool_schema` to get schema
   - Call discovered tool
   - Verify workflow completes

## Performance Tests

1. **Retry Performance:**
   - Measure overhead of retry logic (should be <1ms when no retry needed)
   - Measure total time for successful retry (should match backoff calculation)

2. **Token Measurement:**
   - Measure token count for initial tools exposure (before vs after)
   - Verify 75% token savings with progressive disclosure

## Regression Tests

1. Verify all existing tools still work
2. Verify all existing workflows still work
3. Verify backward compatibility modes work

---

# Migration & Backward Compatibility

## For Users Upgrading

### Retry Mechanism:
- ✅ Fully backward compatible
- Automatically enabled by default
- Can disable with `GEMINI_RETRY_ENABLED=false`
- No migration needed

### Prefix System:
- ⚠️ **Breaking change** if default prefix enabled
- **Recommended migration:**
  1. Keep `GEMINI_COMMAND_PREFIX=""` temporarily (disable)
  2. Manually rename existing commands when ready
  3. Enable default prefix after migration

- **Alternative:** Document that new commands use prefix, old commands without prefix continue to work

### Progressive Disclosure:
- ✅ Backward compatible with flag
- Default: Progressive disclosure enabled
- Legacy mode: `GEMINI_EXPOSE_ALL_TOOLS=true`
- No migration needed

## Migration Steps

### Step 1: Update Environment Variables (Optional)

```bash
# Retry (enabled by default, no action needed unless customizing)
export GEMINI_RETRY_MAX_ATTEMPTS=3
export GEMINI_RETRY_INITIAL_DELAY=1.0

# Prefix (enabled by default, can customize)
export GEMINI_COMMAND_PREFIX="gemini-"
export GEMINI_WORKFLOW_PREFIX="gemini-"

# Progressive disclosure (enabled by default, no action needed)
export GEMINI_EXPOSE_ALL_TOOLS=false
```

### Step 2: Rename Existing Commands (If Using Prefix)

```bash
cd .claude/commands
for f in *.md; do
    # Skip if already prefixed
    if [[ $f != gemini-* ]]; then
        mv "$f" "gemini-$f"
    fi
done
```

### Step 3: Update Documentation

Update any documentation referencing command names:
- `/spec-only` → `/gemini-spec-only`
- `/feature` → `/gemini-feature`
- etc.

---

# Configuration Reference

## Complete Environment Variables

```bash
# ============================================================================
# Retry Configuration
# ============================================================================
GEMINI_RETRY_ENABLED=true                # Enable/disable retry (default: true)
GEMINI_RETRY_MAX_ATTEMPTS=3              # Total attempts (default: 3)
GEMINI_RETRY_INITIAL_DELAY=1.0           # Initial delay in seconds (default: 1.0)
GEMINI_RETRY_MAX_DELAY=60.0              # Max delay in seconds (default: 60.0)
GEMINI_RETRY_BASE=2.0                    # Exponential base (default: 2.0)

# ============================================================================
# Prefix Configuration
# ============================================================================
GEMINI_COMMAND_PREFIX=gemini-            # Command prefix (default: "gemini-")
GEMINI_WORKFLOW_PREFIX=gemini-           # Workflow prefix (default: "gemini-")

# ============================================================================
# Progressive Disclosure
# ============================================================================
GEMINI_EXPOSE_ALL_TOOLS=false            # Expose all tools upfront (default: false)

# ============================================================================
# Existing Configuration (No Changes)
# ============================================================================
CONTEXT_CACHE_TTL_MINUTES=30
DEFAULT_WORKFLOW_DIR=.claude/workflows
DEFAULT_COMMAND_DIR=.claude/commands
DEFAULT_SPEC_DIR=./specs
DEFAULT_REVIEW_DIR=./reviews
DEFAULT_CONTEXT_DIR=./.workflow-context
```

## Configuration Examples

### Example 1: Recommended Default
```bash
# Use all defaults (nothing to configure)
# - Retry enabled with 3 attempts
# - Prefix enabled with "gemini-"
# - Progressive disclosure enabled
```

### Example 2: Aggressive Retry
```bash
export GEMINI_RETRY_MAX_ATTEMPTS=5
export GEMINI_RETRY_INITIAL_DELAY=0.5
# More retries, faster initial retry
```

### Example 3: Custom Prefix
```bash
export GEMINI_COMMAND_PREFIX="gw-"
export GEMINI_WORKFLOW_PREFIX="gw-"
# Commands: /gw-spec-only, /gw-feature, etc.
```

### Example 4: Disable Prefix (Legacy)
```bash
export GEMINI_COMMAND_PREFIX=""
export GEMINI_WORKFLOW_PREFIX=""
# Commands: /spec-only, /feature, etc. (may clash)
```

### Example 5: Legacy Mode (All Features Disabled)
```bash
export GEMINI_RETRY_ENABLED=false
export GEMINI_COMMAND_PREFIX=""
export GEMINI_WORKFLOW_PREFIX=""
export GEMINI_EXPOSE_ALL_TOOLS=true
# Behavior matches pre-feature codebase
```

---

# Implementation Checklist

## Phase 1: Foundation

### Week 1: Retry Mechanism
- [ ] Create `utils/retry.py` with RetryConfig, RetryStatistics, retry_async
- [ ] Add unit tests for retry logic
- [ ] Integrate into `gemini_client.py`
- [ ] Add environment variable loading
- [ ] Add retry statistics tracking
- [ ] Test with mock Gemini CLI (rate limit simulation)
- [ ] Test with actual Gemini CLI
- [ ] Update documentation

### Week 1: Prefix System
- [ ] Update `resources.py` with prefix configuration
- [ ] Modify `setup_workflows.py` for prefix support
- [ ] Modify `generate_command.py` for prefix support
- [ ] Update `workflow_templates.py` with {COMMAND_NAME} placeholder
- [ ] Add unit tests for prefix logic
- [ ] Test end-to-end workflow generation
- [ ] Create migration guide
- [ ] Update documentation

### Week 2: Progressive Tool Disclosure
- [ ] Create `tool_registry.py` with metadata for all tools
- [ ] Add `discover_tools` tool to `server.py`
- [ ] Add `get_tool_schema` tool to `server.py`
- [ ] Modify `list_tools()` to expose minimal set
- [ ] Add backward compatibility flag (GEMINI_EXPOSE_ALL_TOOLS)
- [ ] Add unit tests for tool discovery
- [ ] Test end-to-end discovery workflow
- [ ] Measure token savings
- [ ] Update documentation

## Phase 2: Integration & Testing

### Week 2-3: Integration Testing
- [ ] Test all three features together
- [ ] Test feature interactions
- [ ] Verify no regressions in existing tools
- [ ] Test backward compatibility modes
- [ ] Performance testing (retry overhead, token savings)

### Week 3: Documentation
- [ ] Update README.md with new features
- [ ] Create configuration guide
- [ ] Create migration guide
- [ ] Add examples for all features
- [ ] Document environment variables
- [ ] Update troubleshooting guide

## Phase 3: Release

### Week 3: Release Preparation
- [ ] Create release notes
- [ ] Version bump (e.g., 1.1.0)
- [ ] Tag release in git
- [ ] Update CHANGELOG.md
- [ ] Announce to users

---

# Success Metrics

## Retry Mechanism
- ✅ 95%+ success rate for retryable errors
- ✅ Average < 0.2 retries per call
- ✅ Non-retryable errors fail immediately (< 1 second)
- ✅ Rate limit errors recover automatically

## Prefix System
- ✅ Zero command clash reports from users
- ✅ 100% of generated commands include prefix
- ✅ Clear identification of Gemini commands
- ✅ Easy opt-out for users who prefer no prefix

## Progressive Tool Disclosure
- ✅ 75% token reduction for initial exposure (2000 → 500 tokens)
- ✅ Claude successfully discovers relevant tools 95%+ of the time
- ✅ Average discovery workflow: < 1000 tokens
- ✅ No user confusion about missing tools

## Overall
- ✅ No regressions in existing functionality
- ✅ Positive user feedback
- ✅ Improved reliability (fewer failures)
- ✅ Improved efficiency (better token usage)

---

# Risk Assessment

## Low Risk
- Retry mechanism (well-established pattern)
- Prefix system (simple string manipulation)
- Progressive disclosure backward compatibility flag

## Medium Risk
- Prefix enabled by default (changes user experience)
  - **Mitigation:** Clear documentation, easy opt-out
  - **Recommendation:** Announce in advance, provide migration guide

- Progressive disclosure by default (changes tool exposure)
  - **Mitigation:** Backward compatibility flag
  - **Recommendation:** Monitor for discovery failures, easy rollback

## High Risk
- None identified

---

# Open Questions

## For User Decision:

1. **Prefix Default Behavior:**
   - ✅ **Decided:** Enabled by default with "gemini-"
   - Should we announce this change in advance to give users time to prepare?

2. **Progressive Disclosure Rollout:**
   - Should we enable progressive disclosure by default immediately?
   - Or start with opt-in (GEMINI_EXPOSE_ALL_TOOLS=false to enable)?
   - **Recommendation:** Enable by default, but monitor closely

3. **Migration Support:**
   - Should we provide a migration script to rename existing commands?
   - Or just document manual migration?
   - **Recommendation:** Document manual migration, optional script later

4. **Retry Telemetry:**
   - Should we log retry statistics to a file for analysis?
   - Or just expose via get_retry_stats()?
   - **Recommendation:** Just API for now, file logging optional later

---

# Appendices

## Appendix A: Token Usage Analysis

**Current State (All Tools Exposed):**
```
list_tools() response: ~2000 tokens
- 11 tools × ~150-200 tokens each
- Every conversation starts with this overhead
```

**Progressive Disclosure:**
```
Initial exposure: ~500 tokens (3 core tools)
discover_tools call: ~300 tokens (returns 5-6 tools)
get_tool_schema call: ~200 tokens (1 tool schema)

Total for typical workflow: ~1000 tokens (50% savings)
```

## Appendix B: Retry Examples

**Scenario 1: Rate Limit**
```
Attempt 1: "rate limit exceeded" → wait 1.0s → retry
Attempt 2: Success
Result: Operation succeeds, user unaware of retry
```

**Scenario 2: Temporary Network Issue**
```
Attempt 1: "connection timeout" → wait 1.0s → retry
Attempt 2: "connection timeout" → wait 2.0s → retry
Attempt 3: Success
Result: Operation succeeds after 2 retries
```

**Scenario 3: Authentication Error**
```
Attempt 1: "invalid api key"
Result: Fail immediately (non-retryable), no retry
```

## Appendix C: Prefix Examples

**Generated Files:**
```
.claude/commands/gemini-spec-only.md
.claude/commands/gemini-feature.md
.claude/commands/gemini-refactor.md
.claude/workflows/gemini-spec-only.md
```

**Usage:**
```
/gemini-spec-only Add user authentication
/gemini-feature Implement rate limiting
/gemini-refactor Extract common utilities
```

## Appendix D: Progressive Disclosure Flow

**User Workflow:**
```
User: "I need to analyze my codebase"

1. Claude calls: discover_tools(query="analyze codebase")
   → Returns: query_codebase_tool, analyze_codebase_with_gemini, find_code_by_intent_tool

2. Claude calls: get_tool_schema("query_codebase_tool")
   → Returns: Full schema with parameters

3. Claude calls: query_codebase_tool(questions=[...])
   → Executes analysis

Result: Only loaded schemas for tools actually used
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-19 | Initial specification | Auto-generated |

---

## Approval

**Pending User Approval:**
- [ ] Retry mechanism design approved
- [ ] Prefix system design approved
- [ ] Progressive disclosure design approved
- [ ] Configuration strategy approved
- [ ] Migration plan approved

**Ready for Implementation:** ⏳ Awaiting approval
