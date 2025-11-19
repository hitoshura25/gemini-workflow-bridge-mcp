"""
Tests for retry mechanism with exponential backoff.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from hitoshura25_gemini_workflow_bridge.utils.retry import (
    NonRetryableError,
    RetryableError,
    RetryConfig,
    RetryStatistics,
    retry_async,
)


class TestRetryConfig:
    """Tests for RetryConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            enabled=False
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.enabled is False

    def test_calculate_delay_no_jitter(self):
        """Test delay calculation without jitter"""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0   # 1.0 * 2^0 = 1.0
        assert config.calculate_delay(1) == 2.0   # 1.0 * 2^1 = 2.0
        assert config.calculate_delay(2) == 4.0   # 1.0 * 2^2 = 4.0
        assert config.calculate_delay(3) == 8.0   # 1.0 * 2^3 = 8.0

    def test_calculate_delay_with_max(self):
        """Test delay calculation respects max_delay"""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 5.0  # Capped at max_delay
        assert config.calculate_delay(10) == 5.0  # Still capped

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation includes jitter"""
        config = RetryConfig(initial_delay=10.0, jitter=True, jitter_range=0.2)

        # With jitter, delay should be within ±20% of base value
        delay = config.calculate_delay(0)
        assert 8.0 <= delay <= 12.0  # 10.0 ± 20%

    def test_is_retryable_patterns(self):
        """Test error classification based on patterns"""
        config = RetryConfig()

        # Retryable errors
        assert config.is_retryable("rate limit exceeded") is True
        assert config.is_retryable("quota exceeded") is True
        assert config.is_retryable("connection timeout") is True
        assert config.is_retryable("service temporarily unavailable") is True
        assert config.is_retryable("502 Bad Gateway") is True
        assert config.is_retryable("503 Service Unavailable") is True

        # Non-retryable errors (fail fast)
        assert config.is_retryable("authentication failed") is False
        assert config.is_retryable("unauthorized access") is False
        assert config.is_retryable("invalid api key") is False
        assert config.is_retryable("400 bad request") is False
        assert config.is_retryable("404 not found") is False

        # Unknown errors (conservative - don't retry)
        assert config.is_retryable("some random error") is False

    def test_is_retryable_case_insensitive(self):
        """Test error classification is case-insensitive"""
        config = RetryConfig()

        assert config.is_retryable("RATE LIMIT EXCEEDED") is True
        assert config.is_retryable("Rate Limit Exceeded") is True
        assert config.is_retryable("AUTHENTICATION FAILED") is False
        assert config.is_retryable("Authentication Failed") is False


class TestRetryStatistics:
    """Tests for RetryStatistics class"""

    def test_initial_stats(self):
        """Test initial statistics are zero"""
        stats = RetryStatistics()
        assert stats.total_calls == 0
        assert stats.total_retries == 0
        assert stats.total_successes == 0
        assert stats.total_failures == 0
        assert stats.success_rate == 0.0
        assert stats.average_retries == 0.0

    def test_record_successful_call(self):
        """Test recording successful call"""
        stats = RetryStatistics()
        stats.record_call(success=True, retries=0)

        assert stats.total_calls == 1
        assert stats.total_successes == 1
        assert stats.total_failures == 0
        assert stats.total_retries == 0
        assert stats.success_rate == 1.0
        assert stats.average_retries == 0.0

    def test_record_failed_call(self):
        """Test recording failed call"""
        stats = RetryStatistics()
        stats.record_call(success=False, retries=2, error_type="max_retries")

        assert stats.total_calls == 1
        assert stats.total_successes == 0
        assert stats.total_failures == 1
        assert stats.total_retries == 2
        assert stats.success_rate == 0.0
        assert stats.average_retries == 2.0
        assert stats.by_error_type["max_retries"] == 1

    def test_multiple_calls(self):
        """Test statistics across multiple calls"""
        stats = RetryStatistics()

        # 3 successful calls (1 with retry)
        stats.record_call(success=True, retries=0)
        stats.record_call(success=True, retries=1)
        stats.record_call(success=True, retries=0)

        # 1 failed call
        stats.record_call(success=False, retries=2, error_type="max_retries")

        assert stats.total_calls == 4
        assert stats.total_successes == 3
        assert stats.total_failures == 1
        assert stats.total_retries == 3
        assert stats.success_rate == 0.75
        assert stats.average_retries == 0.75

    def test_to_dict(self):
        """Test exporting statistics to dictionary"""
        stats = RetryStatistics()
        stats.record_call(success=True, retries=1)
        stats.record_call(success=False, retries=2, error_type="timeout")

        result = stats.to_dict()
        assert result["total_calls"] == 2
        assert result["total_successes"] == 1
        assert result["total_failures"] == 1
        assert result["total_retries"] == 3
        assert result["success_rate"] == 0.5
        assert result["average_retries"] == 1.5
        assert result["errors_by_type"]["timeout"] == 1


class TestRetryAsync:
    """Tests for retry_async function"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful operation without retry"""
        mock_func = AsyncMock(return_value="success")
        config = RetryConfig(max_attempts=3)

        result, retry_count = await retry_async(mock_func, config=config, operation_name="test")

        assert result == "success"
        assert retry_count == 0
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_disabled(self):
        """Test retry is disabled when config.enabled = False"""
        mock_func = AsyncMock(side_effect=RuntimeError("temporary error"))
        config = RetryConfig(enabled=False)

        with pytest.raises(RuntimeError, match="temporary error"):
            await retry_async(mock_func, config=config, operation_name="test")

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_single_retry_then_success(self):
        """Test operation succeeds after single retry"""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("rate limit exceeded"),
            "success"
        ])
        config = RetryConfig(max_attempts=3, initial_delay=0.01)

        result, retry_count = await retry_async(mock_func, config=config, operation_name="test")

        assert result == "success"
        assert retry_count == 1
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_retries_then_success(self):
        """Test operation succeeds after multiple retries"""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            "success"
        ])
        config = RetryConfig(max_attempts=3, initial_delay=0.01)

        result, retry_count = await retry_async(mock_func, config=config, operation_name="test")

        assert result == "success"
        assert retry_count == 2
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test max retries exceeded raises RetryableError"""
        mock_func = AsyncMock(side_effect=RuntimeError("timeout"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01)

        with pytest.raises(RetryableError, match="failed after 3 attempts"):
            await retry_async(mock_func, config=config, operation_name="test")

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test non-retryable error fails immediately"""
        mock_func = AsyncMock(side_effect=RuntimeError("authentication failed"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01)

        with pytest.raises(NonRetryableError, match="Non-retryable error"):
            await retry_async(mock_func, config=config, operation_name="test")

        # Should only be called once (no retries)
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_delay(self):
        """Test retry delays are applied"""
        mock_func = AsyncMock(side_effect=[
            RuntimeError("timeout"),
            "success"
        ])
        config = RetryConfig(max_attempts=3, initial_delay=0.1, jitter=False)

        start_time = asyncio.get_event_loop().time()
        result, retry_count = await retry_async(mock_func, config=config, operation_name="test")
        end_time = asyncio.get_event_loop().time()

        assert result == "success"
        assert retry_count == 1
        # Should have waited at least initial_delay
        assert (end_time - start_time) >= 0.1

    @pytest.mark.asyncio
    async def test_retry_with_kwargs(self):
        """Test retry_async passes kwargs correctly"""
        mock_func = AsyncMock(return_value="success")
        config = RetryConfig(max_attempts=3)

        result, retry_count = await retry_async(
            mock_func,
            config=config,
            operation_name="test",
            prompt="test prompt",
            temperature=0.7
        )

        assert result == "success"
        assert retry_count == 0
        mock_func.assert_called_once_with(prompt="test prompt", temperature=0.7)
