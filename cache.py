"""
Response caching module for the GLM Token Saver API.

This module provides caching functionality for model responses to reduce
API calls and improve response times for repeated queries.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cached response entry."""

    def __init__(
        self,
        response: Dict[str, Any],
        prompt_hash: str,
        ttl: int,
        model_used: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.response = response
        self.prompt_hash = prompt_hash
        self.created_at = datetime.now()
        self.ttl = ttl  # Time to live in seconds
        self.model_used = model_used
        self.metadata = metadata or {}
        self.hit_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    def access(self) -> None:
        """Record an access to this cache entry."""
        self.hit_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return {
            "prompt_hash": self.prompt_hash,
            "created_at": self.created_at.isoformat(),
            "ttl": self.ttl,
            "model_used": self.model_used,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat(),
            "is_expired": self.is_expired(),
            "age_seconds": (datetime.now() - self.created_at).total_seconds(),
            "metadata": self.metadata,
        }


class ResponseCache:
    """
    In-memory response cache for model responses.

    Features:
    - Configurable TTL (time-to-live) for cache entries
    - Size limits with LRU eviction
    - Cache statistics tracking
    - Support for different cache keys (by model, temperature, etc.)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 1000,
        default_ttl: int = 3600,
        cache_by_temperature: bool = True,
        cache_by_max_tokens: bool = False,
    ):
        """
        Initialize the response cache.

        Args:
            enabled: Whether caching is enabled
            max_size: Maximum number of entries in the cache
            default_ttl: Default time-to-live in seconds (1 hour)
            cache_by_temperature: Whether to consider temperature in cache key
            cache_by_max_tokens: Whether to consider max_tokens in cache key
        """
        self.enabled = enabled
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_by_temperature = cache_by_temperature
        self.cache_by_max_tokens = cache_by_max_tokens

        self._cache: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        logger.info(
            f"Response cache initialized: enabled={enabled}, max_size={max_size}, "
            f"default_ttl={default_ttl}s"
        )

    def generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """
        Generate a cache key for a request.

        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Additional parameters

        Returns:
            A hash string to use as cache key
        """
        # Create a dictionary with the parameters that matter for caching
        key_data = {
            "messages": messages,
        }

        # Include temperature if configured
        if self.cache_by_temperature:
            key_data["temperature"] = round(temperature, 2)

        # Include max_tokens if configured
        if self.cache_by_max_tokens and max_tokens:
            key_data["max_tokens"] = max_tokens

        # Don't include other kwargs in cache key for simplicity
        # They're typically less relevant for cache hits

        # Generate hash from the key data
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if available.

        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Additional parameters

        Returns:
            Cached response if found and not expired, None otherwise
        """
        if not self.enabled:
            self._miss_count += 1
            return None

        cache_key = self.generate_cache_key(messages, temperature, max_tokens, **kwargs)

        entry = self._cache.get(cache_key)

        if entry is None:
            self._miss_count += 1
            logger.debug(f"Cache miss for key: {cache_key[:16]}...")
            return None

        if entry.is_expired():
            # Remove expired entry
            del self._cache[cache_key]
            logger.debug(f"Cache entry expired and removed: {cache_key[:16]}...")
            self._miss_count += 1
            return None

        # Cache hit!
        entry.access()
        self._hit_count += 1
        logger.info(
            f"Cache hit for key: {cache_key[:16]}... "
            f"(model: {entry.model_used}, hits: {entry.hit_count})"
        )

        # Return a copy to prevent modification of cached data
        response_copy = json.loads(json.dumps(entry.response))

        # Add cache metadata
        response_copy["_cached"] = True
        response_copy["_cache_metadata"] = {
            "cached_at": entry.created_at.isoformat(),
            "cache_hit_count": entry.hit_count,
            "cache_age_seconds": (datetime.now() - entry.created_at).total_seconds(),
        }

        return response_copy

    def set(
        self,
        messages: List[Dict[str, str]],
        response: Dict[str, Any],
        temperature: float,
        max_tokens: Optional[int],
        model_used: str,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Cache a response.

        Args:
            messages: List of message dictionaries
            response: Response to cache
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            model_used: Model that generated the response
            ttl: Custom TTL in seconds (uses default if None)
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        cache_key = self.generate_cache_key(messages, temperature, max_tokens, **kwargs)

        # Check if we need to evict entries
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Create cache entry
        entry = CacheEntry(
            response=response,
            prompt_hash=cache_key,
            ttl=ttl or self.default_ttl,
            model_used=model_used,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages),
            },
        )

        self._cache[cache_key] = entry
        logger.info(
            f"Cached response for key: {cache_key[:16]}... "
            f"(model: {model_used}, ttl: {entry.ttl}s, size: {len(self._cache)})"
        )

    def _evict_oldest(self) -> None:
        """Evict the oldest accessed entry from the cache."""
        if not self._cache:
            return

        # Find entry with oldest last_accessed time
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )

        del self._cache[oldest_key]
        self._eviction_count += 1
        logger.info(f"Evicted oldest cache entry: {oldest_key[:16]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {size} cache entries")

    def invalidate_model(self, model: str) -> None:
        """
        Invalidate all cache entries for a specific model.

        Args:
            model: Model name to invalidate (e.g., "local", "cerebras")
        """
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if entry.model_used == model
        ]
        for key in keys_to_remove:
            del self._cache[key]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for model: {model}")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hit_count + self._miss_count
        hit_rate = (
            self._hit_count / total_requests * 100
            if total_requests > 0
            else 0.0
        )

        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "eviction_count": self._eviction_count,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }

    def get_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of cache entries (for admin/debugging).

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of cache entry dictionaries
        """
        entries = [
            entry.to_dict()
            for entry in sorted(
                self._cache.values(),
                key=lambda e: e.last_accessed,
                reverse=True,
            )
        ]
        return entries[:limit]

    def reset_statistics(self) -> None:
        """Reset cache statistics."""
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        logger.info("Cache statistics reset")