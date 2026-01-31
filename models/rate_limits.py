"""Rate limit information from API headers."""

from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class CerebrasRateLimits:
    """Store and manage rate limit information from Cerebras API headers."""
    
    def __init__(self):
        # Limits from API headers
        self.limit_requests_day: Optional[int] = None
        self.limit_tokens_minute: Optional[int] = None
        
        # Remaining from API headers
        self.remaining_requests_day: Optional[int] = None
        self.remaining_tokens_minute: Optional[int] = None
        
        # Reset times (seconds until reset)
        self.reset_requests_day: Optional[int] = None
        self.reset_tokens_minute: Optional[int] = None
        
        # Timestamp of last update
        self.last_updated: Optional[float] = None
    
    def update_from_headers(self, headers: dict):
        """Update rate limit info from response headers."""
        updated = False
        
        # Parse header values
        if 'x-ratelimit-limit-requests-day' in headers:
            self.limit_requests_day = int(headers['x-ratelimit-limit-requests-day'])
            updated = True
        
        if 'x-ratelimit-limit-tokens-minute' in headers:
            self.limit_tokens_minute = int(headers['x-ratelimit-limit-tokens-minute'])
            updated = True
        
        if 'x-ratelimit-remaining-requests-day' in headers:
            self.remaining_requests_day = int(headers['x-ratelimit-remaining-requests-day'])
            updated = True
        
        if 'x-ratelimit-remaining-tokens-minute' in headers:
            self.remaining_tokens_minute = int(headers['x-ratelimit-remaining-tokens-minute'])
            updated = True
        
        if 'x-ratelimit-reset-requests-day' in headers:
            self.reset_requests_day = int(headers['x-ratelimit-reset-requests-day'])
            updated = True
        
        if 'x-ratelimit-reset-tokens-minute' in headers:
            self.reset_tokens_minute = int(headers['x-ratelimit-reset-tokens-minute'])
            updated = True
        
        if updated:
            self.last_updated = time.time()
            logger.debug(f"Updated rate limits from headers: {self.get_summary()}")
    
    def is_near_limit(self, threshold_percent: float = 0.2) -> tuple[bool, str]:
        """
        Check if we're near the rate limit.
        
        Args:
            threshold_percent: Percentage threshold (e.g., 0.2 for 20%)
        
        Returns:
            Tuple of (is_near_limit, reason)
        """
        # Check tokens per minute (most critical for streaming)
        if (self.remaining_tokens_minute is not None and 
            self.limit_tokens_minute is not None):
            remaining_percent = self.remaining_tokens_minute / self.limit_tokens_minute
            if remaining_percent <= threshold_percent:
                reason = (
                    f"Token limit approaching: "
                    f"{self.remaining_tokens_minute}/{self.limit_tokens_minute} tokens/minute "
                    f"({remaining_percent*100:.1f}% remaining)"
                )
                return True, reason
        
        # Check requests per day
        if (self.remaining_requests_day is not None and 
            self.limit_requests_day is not None):
            remaining_percent = self.remaining_requests_day / self.limit_requests_day
            if remaining_percent <= threshold_percent:
                reason = (
                    f"Daily request limit approaching: "
                    f"{self.remaining_requests_day}/{self.limit_requests_day} requests "
                    f"({remaining_percent*100:.1f}% remaining)"
                )
                return True, reason
        
        return False, "Rate limits are safe"
    
    def get_time_until_reset_minutes(self) -> Optional[int]:
        """Get estimated seconds until token limit resets."""
        if self.reset_tokens_minute is not None:
            return max(0, self.reset_tokens_minute)
        return None
    
    def get_time_until_reset_day(self) -> Optional[int]:
        """Get estimated seconds until daily limit resets."""
        if self.reset_requests_day is not None:
            return max(0, self.reset_requests_day)
        return None
    
    def is_stale(self, max_age_seconds: float = 120) -> bool:
        """Check if rate limit info is stale (old data)."""
        if self.last_updated is None:
            return True
        return (time.time() - self.last_updated) > max_age_seconds
    
    def get_summary(self) -> dict:
        """Get summary of current rate limit status."""
        return {
            "limits": {
                "requests_day": self.limit_requests_day,
                "tokens_minute": self.limit_tokens_minute,
            },
            "remaining": {
                "requests_day": self.remaining_requests_day,
                "tokens_minute": self.remaining_tokens_minute,
            },
            "reset_seconds": {
                "requests_day": self.reset_requests_day,
                "tokens_minute": self.reset_tokens_minute,
            },
            "last_updated": self.last_updated,
        }
    
    def __str__(self) -> str:
        """String representation for logging."""
        parts = []
        if self.remaining_tokens_minute is not None and self.limit_tokens_minute is not None:
            pct = (self.remaining_tokens_minute / self.limit_tokens_minute * 100)
            parts.append(f"tokens: {self.remaining_tokens_minute}/{self.limit_tokens_minute} ({pct:.1f}%)")
        if self.remaining_requests_day is not None and self.limit_requests_day is not None:
            pct = (self.remaining_requests_day / self.limit_requests_day * 100)
            parts.append(f"requests: {self.remaining_requests_day}/{self.limit_requests_day} ({pct:.1f}%)")
        return ", ".join(parts) if parts else "No data"