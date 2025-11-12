"""
Tests for ContextCacheManager.
"""

import time
from datetime import timedelta
from hitoshura25_gemini_workflow_bridge.cache_manager import ContextCacheManager


def test_cache_manager_initialization():
    """Test cache manager initializes with correct defaults."""
    cache = ContextCacheManager()
    assert cache.ttl_minutes == 30
    assert cache.current_context_id is None
    assert len(cache.cache) == 0


def test_cache_context():
    """Test caching context with automatic current context setting."""
    cache = ContextCacheManager()

    test_context = {
        "files_content": {"test.py": "content"},
        "project_structure": "test/",
        "analysis": {"summary": "test"}
    }

    cache.cache_context("ctx_test123", test_context)

    # Should be cached
    assert "ctx_test123" in cache.cache
    # Should be set as current
    assert cache.current_context_id == "ctx_test123"
    # Should have metadata
    entry = cache.cache["ctx_test123"]
    assert "cached_at" in entry
    assert "expires_at" in entry
    assert "access_count" in entry
    assert entry["data"] == test_context


def test_cache_context_without_setting_current():
    """Test caching without setting as current context."""
    cache = ContextCacheManager()

    test_context = {"test": "data"}
    cache.cache_context("ctx_test", test_context, set_as_current=False)

    assert "ctx_test" in cache.cache
    assert cache.current_context_id is None


def test_get_cached_context():
    """Test retrieving cached context."""
    cache = ContextCacheManager()

    test_context = {"test": "data"}
    cache.cache_context("ctx_test", test_context)

    # Retrieve context
    retrieved = cache.get_cached_context("ctx_test")
    assert retrieved == test_context

    # Stats should reflect hit
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_get_cached_context_miss():
    """Test cache miss for non-existent context."""
    cache = ContextCacheManager()

    result = cache.get_cached_context("ctx_nonexistent")
    assert result is None

    # Stats should reflect miss
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_get_cached_context_expired():
    """Test that expired contexts are automatically removed."""
    cache = ContextCacheManager(ttl_minutes=0.01)  # 0.6 seconds

    test_context = {"test": "data"}
    cache.cache_context("ctx_test", test_context)

    # Wait for expiration
    time.sleep(1)

    # Should return None and remove from cache
    result = cache.get_cached_context("ctx_test")
    assert result is None
    assert "ctx_test" not in cache.cache

    # Stats should reflect expiration
    stats = cache.get_stats()
    assert stats["expirations"] == 1


def test_get_current_context():
    """Test retrieving current active context."""
    cache = ContextCacheManager()

    # No current context initially
    assert cache.get_current_context() is None

    # Cache context (automatically sets as current)
    test_context = {"test": "data"}
    cache.cache_context("ctx_test", test_context)

    # Should retrieve current context
    current = cache.get_current_context()
    assert current is not None
    context_data, context_id = current
    assert context_data == test_context
    assert context_id == "ctx_test"


def test_get_current_context_expired():
    """Test that expired current context returns None."""
    cache = ContextCacheManager(ttl_minutes=0.01)

    test_context = {"test": "data"}
    cache.cache_context("ctx_test", test_context)

    # Wait for expiration
    time.sleep(1)

    # Should return None
    assert cache.get_current_context() is None
    # current_context_id should be cleared
    assert cache.current_context_id is None


def test_set_current_context():
    """Test manually setting current context."""
    cache = ContextCacheManager()

    # Cache two contexts
    cache.cache_context("ctx_1", {"data": "1"}, set_as_current=False)
    cache.cache_context("ctx_2", {"data": "2"}, set_as_current=False)

    # Manually set ctx_1 as current
    success = cache.set_current_context("ctx_1")
    assert success is True
    assert cache.current_context_id == "ctx_1"

    # Try to set non-existent context
    success = cache.set_current_context("ctx_nonexistent")
    assert success is False
    assert cache.current_context_id == "ctx_1"  # Unchanged


def test_is_expired():
    """Test checking if context is expired."""
    cache = ContextCacheManager(ttl_minutes=0.01)

    cache.cache_context("ctx_test", {"test": "data"})

    # Should not be expired immediately
    assert cache.is_expired("ctx_test") is False

    # Wait for expiration
    time.sleep(1)

    # Should be expired
    assert cache.is_expired("ctx_test") is True

    # Non-existent context should be considered expired
    assert cache.is_expired("ctx_nonexistent") is True


def test_cleanup_expired():
    """Test manual cleanup of expired contexts."""
    cache = ContextCacheManager(ttl_minutes=0.01)

    # Cache multiple contexts
    cache.cache_context("ctx_1", {"data": "1"}, set_as_current=False)
    cache.cache_context("ctx_2", {"data": "2"}, set_as_current=False)
    cache.cache_context("ctx_3", {"data": "3"})

    assert len(cache.cache) == 3

    # Wait for expiration
    time.sleep(1)

    # Cleanup
    removed_count = cache.cleanup_expired()
    assert removed_count == 3
    assert len(cache.cache) == 0
    assert cache.current_context_id is None


def test_access_count_tracking():
    """Test that access count is tracked correctly."""
    cache = ContextCacheManager()

    cache.cache_context("ctx_test", {"test": "data"})

    # Access multiple times
    for _ in range(5):
        cache.get_cached_context("ctx_test")

    # Check access count
    entry = cache.cache["ctx_test"]
    assert entry["access_count"] == 5


def test_cache_stats():
    """Test cache statistics calculation."""
    cache = ContextCacheManager()

    # Cache context
    cache.cache_context("ctx_1", {"data": "1"})

    # Perform hits and misses
    cache.get_cached_context("ctx_1")  # Hit
    cache.get_cached_context("ctx_1")  # Hit
    cache.get_cached_context("ctx_2")  # Miss

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["cached_contexts"] == 1
    assert stats["current_context_id"] == "ctx_1"
    assert stats["hit_rate"] == 2.0 / 3.0  # 2 hits out of 3 total


def test_cache_stats_with_expiration():
    """Test that expirations are tracked in stats."""
    cache = ContextCacheManager(ttl_minutes=0.01)

    cache.cache_context("ctx_1", {"data": "1"})

    # Wait for expiration
    time.sleep(1)

    # Try to get expired context
    cache.get_cached_context("ctx_1")

    stats = cache.get_stats()
    assert stats["expirations"] == 1


def test_clear_cache():
    """Test clearing cached contexts preserves statistics."""
    cache = ContextCacheManager()

    # Cache multiple contexts
    cache.cache_context("ctx_1", {"data": "1"})
    cache.cache_context("ctx_2", {"data": "2"})
    cache.cache_context("ctx_3", {"data": "3"})

    # Access some contexts to generate stats
    cache.get_cached_context("ctx_1")
    cache.get_cached_context("ctx_nonexistent")

    assert len(cache.cache) == 3
    assert cache.current_context_id is not None

    # Clear cache
    cache.clear()

    # Cache cleared but stats preserved
    assert len(cache.cache) == 0
    assert cache.current_context_id is None
    assert cache.stats["hits"] == 1      # Stats preserved!
    assert cache.stats["misses"] == 1    # Stats preserved!


def test_reset_cache():
    """Test resetting cache and all statistics."""
    cache = ContextCacheManager()

    # Cache multiple contexts
    cache.cache_context("ctx_1", {"data": "1"})
    cache.cache_context("ctx_2", {"data": "2"})

    # Access some contexts to generate stats
    cache.get_cached_context("ctx_1")
    cache.get_cached_context("ctx_nonexistent")

    assert len(cache.cache) == 2
    assert cache.current_context_id is not None
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1

    # Reset everything
    cache.reset()

    # Everything reset
    assert len(cache.cache) == 0
    assert cache.current_context_id is None
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0
    assert cache.stats["expirations"] == 0


def test_multiple_context_switching():
    """Test switching between multiple contexts."""
    cache = ContextCacheManager()

    # Cache multiple contexts
    cache.cache_context("ctx_1", {"version": "1"}, set_as_current=False)
    cache.cache_context("ctx_2", {"version": "2"}, set_as_current=False)
    cache.cache_context("ctx_3", {"version": "3"}, set_as_current=False)

    # Switch to ctx_2
    cache.set_current_context("ctx_2")
    current = cache.get_current_context()
    assert current[1] == "ctx_2"
    assert current[0]["version"] == "2"

    # Switch to ctx_1
    cache.set_current_context("ctx_1")
    current = cache.get_current_context()
    assert current[1] == "ctx_1"
    assert current[0]["version"] == "1"


def test_ttl_configuration():
    """Test that TTL can be configured."""
    cache_short = ContextCacheManager(ttl_minutes=1)
    cache_long = ContextCacheManager(ttl_minutes=60)

    assert cache_short.ttl_minutes == 1
    assert cache_long.ttl_minutes == 60

    # Verify expires_at calculation
    cache_short.cache_context("ctx_test", {"test": "data"})
    entry = cache_short.cache["ctx_test"]

    expected_expiry = entry["cached_at"] + timedelta(minutes=1)
    assert abs((entry["expires_at"] - expected_expiry).total_seconds()) < 1


def test_get_current_context_no_stat_inflation():
    """Test that get_current_context() doesn't inflate cache statistics."""
    cache = ContextCacheManager()

    # Cache a context
    cache.cache_context("ctx_test", {"test": "data"})

    # Reset stats to zero
    cache.stats["hits"] = 0
    cache.stats["misses"] = 0

    # Call get_current_context multiple times
    for _ in range(5):
        result = cache.get_current_context()
        assert result is not None
        assert result[1] == "ctx_test"

    # Verify stats were NOT inflated
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0

    # Verify access_count was NOT incremented
    entry = cache.cache["ctx_test"]
    assert entry["access_count"] == 0


def test_set_current_context_no_stat_inflation():
    """Test that set_current_context() doesn't inflate cache statistics."""
    cache = ContextCacheManager()

    # Cache contexts
    cache.cache_context("ctx_1", {"data": "1"}, set_as_current=False)
    cache.cache_context("ctx_2", {"data": "2"}, set_as_current=False)

    # Reset stats to zero
    cache.stats["hits"] = 0
    cache.stats["misses"] = 0

    # Set current context multiple times
    for _ in range(5):
        success = cache.set_current_context("ctx_1")
        assert success is True

    # Verify stats were NOT inflated
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0

    # Verify access_count was NOT incremented
    entry = cache.cache["ctx_1"]
    assert entry["access_count"] == 0


def test_only_get_cached_context_inflates_stats():
    """Test that only get_cached_context() increments statistics."""
    cache = ContextCacheManager()

    # Cache a context
    cache.cache_context("ctx_test", {"test": "data"})

    # Reset stats
    cache.stats["hits"] = 0
    cache.stats["misses"] = 0

    # Call get_current_context and set_current_context (should not inflate)
    cache.get_current_context()
    cache.set_current_context("ctx_test")

    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0

    # Now call get_cached_context (should inflate)
    cache.get_cached_context("ctx_test")

    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 0

    # Check access count
    entry = cache.cache["ctx_test"]
    assert entry["access_count"] == 1
