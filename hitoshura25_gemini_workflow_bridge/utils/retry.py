"""Retry mechanism with exponential backoff for Gemini CLI calls"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
import asyncio
import logging
import random

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
    retryable_error_patterns: List[str] = field(default_factory=list)
    non_retryable_error_patterns: List[str] = field(default_factory=list)

    # Feature flag
    enabled: bool = True

    def __post_init__(self):
        """Set default error patterns if not provided"""
        if not self.retryable_error_patterns:
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

        if not self.non_retryable_error_patterns:
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
    by_error_type: dict = field(default_factory=dict)  # Dict[str, int] - count by error pattern

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
