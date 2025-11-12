"""
Context cache manager with TTL support for automatic context reuse.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class ContextCacheManager:
    """Manages context caching with TTL and automatic reuse.

    This cache manager provides:
    - Time-based expiration (TTL)
    - Automatic "current context" tracking
    - Cache statistics (hits/misses/expirations)
    - Automatic cleanup of expired entries
    """

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

    def _is_valid_context(self, context_id: str) -> bool:
        """Check if context exists and is not expired without incrementing stats.

        Internal method for validation that doesn't have side effects.

        Args:
            context_id: Context ID to validate

        Returns:
            True if context exists and is not expired, False otherwise
        """
        if context_id not in self.cache:
            return False

        # Check expiration without side effects
        return datetime.now() <= self.cache[context_id]["expires_at"]

    def get_current_context(self) -> Optional[tuple[Dict[str, Any], str]]:
        """Get the current active context.

        Returns:
            Tuple of (context_data, context_id) or None if no current context/expired
        """
        if not self.current_context_id:
            return None

        # Check validity without inflating stats
        if not self._is_valid_context(self.current_context_id):
            # Current context expired, clear it
            self.current_context_id = None
            return None

        # Return data directly without going through get_cached_context
        return self.cache[self.current_context_id]["data"], self.current_context_id

    def set_current_context(self, context_id: str) -> bool:
        """Set a context as the current context.

        Args:
            context_id: Context ID to set as current

        Returns:
            True if successful, False if context not found/expired
        """
        if self._is_valid_context(context_id):
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
            Dictionary with cache statistics including hit rate
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            **self.stats,
            "cached_contexts": len(self.cache),
            "current_context_id": self.current_context_id,
            "hit_rate": hit_rate
        }

    def clear(self) -> None:
        """Clear all cached contexts, preserving statistics.

        Use this for cache maintenance while keeping session statistics
        for monitoring and debugging purposes.
        """
        self.cache.clear()
        self.current_context_id = None
        # Stats intentionally NOT reset - use reset() to clear stats

    def reset(self) -> None:
        """Reset cache and all statistics.

        Use this for complete cleanup including session statistics.
        """
        self.clear()
        self.stats = {"hits": 0, "misses": 0, "expirations": 0}
