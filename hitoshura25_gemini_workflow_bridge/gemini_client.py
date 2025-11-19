"""Gemini CLI client wrapper using subprocess"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
from typing import Any

from .cache_manager import ContextCacheManager
from .utils.retry import (
    NonRetryableError,
    RetryableError,
    RetryConfig,
    RetryStatistics,
    retry_async,
)

# Set up logger for debugging
logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini CLI with caching, context management, and retry logic

    Uses the `gemini` CLI command instead of API calls.
    Requires Gemini CLI to be installed and authenticated.
    """

    def __init__(self, model: str = "auto", retry_config: RetryConfig | None = None):
        # Validate CLI is installed
        cli_path = shutil.which("gemini")
        if not cli_path:
            raise RuntimeError(
                "Gemini CLI not found. Install with: npm install -g @google/gemini-cli\n"
                "Then authenticate with: gemini"
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
                raise RuntimeError(
                    f"Gemini CLI found but not working. Error: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Gemini CLI not responding (timeout)")
        except Exception as e:
            raise RuntimeError(f"Error testing Gemini CLI: {e}")

        self.model_name = model
        self.cli_path = cli_path

        # Initialize retry configuration
        if retry_config is None:
            retry_config = self._load_retry_config()
        self.retry_config = retry_config

        # Initialize retry statistics
        self.retry_stats = RetryStatistics()

        # Initialize cache manager with configurable TTL
        ttl_minutes = int(os.getenv("CONTEXT_CACHE_TTL_MINUTES", "30"))
        self.cache_manager = ContextCacheManager(ttl_minutes=ttl_minutes)

    @staticmethod
    def _load_retry_config() -> RetryConfig:
        """Load retry configuration from environment variables with error handling"""
        # Default values
        defaults = {
            "max_attempts": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "enabled": True
        }

        # Parse max_attempts with error handling
        try:
            max_attempts = int(os.getenv("GEMINI_RETRY_MAX_ATTEMPTS", str(defaults["max_attempts"])))
        except ValueError:
            logger.warning(
                f"Invalid GEMINI_RETRY_MAX_ATTEMPTS value, using default {defaults['max_attempts']}"
            )
            max_attempts = defaults["max_attempts"]

        # Parse initial_delay with error handling
        try:
            initial_delay = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", str(defaults["initial_delay"])))
        except ValueError:
            logger.warning(
                f"Invalid GEMINI_RETRY_INITIAL_DELAY value, using default {defaults['initial_delay']}"
            )
            initial_delay = defaults["initial_delay"]

        # Parse max_delay with error handling
        try:
            max_delay = float(os.getenv("GEMINI_RETRY_MAX_DELAY", str(defaults["max_delay"])))
        except ValueError:
            logger.warning(
                f"Invalid GEMINI_RETRY_MAX_DELAY value, using default {defaults['max_delay']}"
            )
            max_delay = defaults["max_delay"]

        # Parse exponential_base with error handling
        try:
            exponential_base = float(os.getenv("GEMINI_RETRY_BASE", str(defaults["exponential_base"])))
        except ValueError:
            logger.warning(
                f"Invalid GEMINI_RETRY_BASE value, using default {defaults['exponential_base']}"
            )
            exponential_base = defaults["exponential_base"]

        # Parse enabled (boolean doesn't need try-except)
        enabled = os.getenv("GEMINI_RETRY_ENABLED", "true").lower() == "true"

        return RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            enabled=enabled
        )

    async def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None
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
            result, retry_count = await retry_async(
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
            retry_count = e.retry_count
            error_type = "non_retryable"
            logger.error(f"Non-retryable error in Gemini CLI: {e}")
            raise RuntimeError(str(e)) from e

        except RetryableError as e:
            # Max retries exceeded - all attempts failed
            retry_count = e.retry_count
            error_type = "max_retries_exceeded"
            logger.error(f"Max retries exceeded for Gemini CLI: {e}")
            raise RuntimeError(str(e)) from e

        except Exception as e:
            # Unexpected error (shouldn't happen, but handle gracefully)
            # retry_count remains 0 from initialization since we can't determine actual retries
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
        max_tokens: int | None = None
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
            except TimeoutError:
                await process.kill()
                await process.wait()
                raise RuntimeError("Gemini CLI request timed out after 5 minutes")

            # Decode stderr once for reuse
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

            # Add debug logging
            logger.debug(f"Gemini CLI returncode: {process.returncode}")
            logger.debug(f"Gemini CLI stdout length: {len(stdout)}")
            if stderr_text:
                logger.debug(f"Gemini CLI stderr: {stderr_text}")

            # Log any warnings from stderr (even on success)
            # Common warnings include Node.js top-level await warnings from dependencies
            if stderr_text and process.returncode == 0:
                # Check if stderr contains only warnings
                has_warnings = any(line.strip().startswith('Warning:') for line in stderr_text.split('\n'))
                if has_warnings:
                    logger.warning(f"Gemini CLI produced warnings (but succeeded): {stderr_text}")

            # Check for errors
            if process.returncode != 0:
                # Filter out Node.js warnings (e.g., "Warning: Detected unsettled top-level await")
                # These are common with yoga-layout and other dependencies but don't indicate failure
                error_lines = []

                # Patterns that indicate warning context (not actual errors)
                warning_patterns = (
                    'Warning:',       # Node.js warnings
                    'file://',        # File path references in warnings
                    'const ',         # JavaScript code snippets
                    'let ',
                    'var ',
                    'function ',
                    'import ',
                    'export ',
                    'async ',
                    'await ',
                )

                # Patterns that indicate actual errors
                error_indicators = (
                    'Error:',
                    'TypeError:',
                    'SyntaxError:',
                    'ReferenceError:',
                    'RangeError:',
                    'Failed',
                    'Exception',
                    'FATAL',
                )

                for line in stderr_text.split('\n'):
                    stripped_line = line.strip()  # Cache stripped value

                    if not stripped_line:  # Skip empty lines
                        continue

                    # Check if this is clearly an error (whitelist approach)
                    is_error = any(stripped_line.startswith(pattern) for pattern in error_indicators)
                    if is_error:
                        error_lines.append(line)
                        continue

                    # Check if this is warning context (blacklist approach)
                    is_warning_context = any(stripped_line.startswith(pattern) for pattern in warning_patterns)
                    if not is_warning_context:
                        # Not clearly a warning pattern, but also not clearly an error
                        # Include it as a potential error (conservative approach)
                        error_lines.append(line)

                # If we have actual error content after filtering warnings, raise it
                if error_lines:
                    error_msg = '\n'.join(error_lines).strip()
                    raise RuntimeError(f"Gemini CLI error: {error_msg}")
                else:
                    # Only warnings were present, log but don't fail
                    logger.warning(f"Gemini CLI returned non-zero exit code but only warnings were present: {stderr_text}")

            # Parse JSON response
            try:
                output = stdout.decode('utf-8', errors='replace')

                # Log raw output for debugging (truncated)
                logger.debug(f"Gemini CLI raw output (first 500 chars): {output[:500]}")

                # Validate output is not empty
                if not output or not output.strip():
                    raise RuntimeError(
                        "Gemini CLI returned empty response. "
                        "Check authentication with: gemini --version"
                    )

                result = json.loads(output)

                # Extract response text from CLI JSON format
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
                    # Fallback: return raw output if format unexpected
                    return output

            except json.JSONDecodeError as e:
                # If JSON parsing fails, return raw output
                output = stdout.decode('utf-8', errors='replace')

                # Validate decoded output
                if not output or not output.strip():
                    raise RuntimeError(
                        f"Gemini CLI returned invalid/empty output. "
                        f"JSON decode error: {str(e)}"
                    )

                return output

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error calling Gemini CLI: {e}")

    async def analyze_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """Generate content with provided context

        Combines context and prompt into a single prompt for the CLI.
        """
        full_prompt = f"""Context:
{context}

Task:
{prompt}

Please provide a detailed, structured response."""

        return await self.generate_content(full_prompt, temperature)

    def cache_context(self, context_id: str, context: dict[str, Any]) -> None:
        """Cache context for reuse (automatically sets as current context).

        Args:
            context_id: Unique identifier for context
            context: Context data to cache
        """
        self.cache_manager.cache_context(
            context_id,
            context,
            set_as_current=True  # Always set as current context
        )

    def get_cached_context(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve cached context (checks TTL expiration).

        Args:
            context_id: Context ID to retrieve

        Returns:
            Context data or None if not found/expired
        """
        return self.cache_manager.get_cached_context(context_id)

    def get_current_context(self) -> tuple[dict[str, Any], str] | None:
        """Get the current active context.

        Returns:
            Tuple of (context_data, context_id) or None if no current context/expired
        """
        return self.cache_manager.get_current_context()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hit rate
        """
        return self.cache_manager.get_stats()

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
