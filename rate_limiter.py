"""Rate limit tracking and management for Cerebras API."""

import time
from datetime import datetime, timedelta
from typing import Optional
from collections import deque
import logging
from config import get_config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Track and manage Cerebras API rate limits."""
    
    def __init__(self):
        self.config = get_config().rate_limit
        
        # Request tracking (timestamps)
        self._minute_requests = deque()
        self._hour_requests = deque()
        self._day_requests = deque()
        
        # Token tracking (timestamp, count)
        self._minute_tokens = deque()  # [(timestamp, count), ...]
        self._hour_tokens = deque()
        self._day_tokens = deque()
        
        # Statistics
        self._fallback_count = 0
        self._rate_limit_hits = 0
        self._total_requests = 0
        self._total_tokens = 0
        
        logger.info("RateLimiter initialized")
    
    def _cleanup_old_entries(self, window_seconds: int, entries: deque) -> deque:
        """Remove entries older than the time window."""
        cutoff = time.time() - window_seconds
        while entries and (isinstance(entries[0], float) and entries[0] < cutoff):
            entries.popleft()
        while entries and (isinstance(entries[0], tuple) and entries[0][0] < cutoff):
            entries.popleft()
        return entries
    
    def record_request(self, tokens_used: int):
        """Record a Cerebras API request with tokens used."""
        if not self.config.enabled:
            return
        
        now = time.time()
        self._total_requests += 1
        self._total_tokens += tokens_used
        
        # Record request timestamps
        self._minute_requests.append(now)
        self._hour_requests.append(now)
        self._day_requests.append(now)
        
        # Record token usage with timestamps
        self._minute_tokens.append((now, tokens_used))
        self._hour_tokens.append((now, tokens_used))
        self._day_tokens.append((now, tokens_used))
        
        # Cleanup old entries
        self._minute_requests = self._cleanup_old_entries(60, self._minute_requests)
        self._hour_requests = self._cleanup_old_entries(3600, self._hour_requests)
        self._day_requests = self._cleanup_old_entries(86400, self._day_requests)
        
        self._minute_tokens = self._cleanup_old_entries(60, self._minute_tokens)
        self._hour_tokens = self._cleanup_old_entries(3600, self._hour_tokens)
        self._day_tokens = self._cleanup_old_entries(86400, self._day_tokens)
        
        logger.debug(
            f"Recorded request: {tokens_used} tokens. "
            f"Minute: {self.get_requests_in_minute()} req, {self.get_tokens_in_minute()} tokens"
        )
    
    def get_requests_in_minute(self) -> int:
        """Get request count in the last minute."""
        return len(self._minute_requests)
    
    def get_requests_in_hour(self) -> int:
        """Get request count in the last hour."""
        return len(self._hour_requests)
    
    def get_requests_in_day(self) -> int:
        """Get request count in the last day."""
        return len(self._day_requests)
    
    def get_tokens_in_minute(self) -> int:
        """Get token count in the last minute."""
        return sum(count for _, count in self._minute_tokens)
    
    def get_tokens_in_hour(self) -> int:
        """Get token count in the last hour."""
        return sum(count for _, count in self._hour_tokens)
    
    def get_tokens_in_day(self) -> int:
        """Get token count in the last day."""
        return sum(count for _, count in self._day_tokens)
    
    def should_use_cerebras(self) -> tuple[bool, str]:
        """
        Check if Cerebras should be used based on rate limits.
        
        Returns:
            Tuple of (should_use, reason)
        """
        if not self.config.enabled:
            return True, "Rate limiting disabled"
        
        # Check request limits
        req_min = self.get_requests_in_minute()
        req_hour = self.get_requests_in_hour()
        req_day = self.get_requests_in_day()
        
        req_min_limit = self.config.requests_per_minute
        req_hour_limit = self.config.requests_per_hour
        req_day_limit = self.config.requests_per_day
        
        # Calculate thresholds
        req_min_threshold = int(req_min_limit * self.config.request_fallback_threshold)
        req_hour_threshold = int(req_hour_limit * self.config.request_fallback_threshold)
        req_day_threshold = int(req_day_limit * self.config.request_fallback_threshold)
        
        # Check request thresholds
        if req_min >= req_min_threshold:
            reason = f"Request limit approaching: {req_min}/{req_min_limit}/min (threshold: {req_min_threshold})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        if req_hour >= req_hour_threshold:
            reason = f"Request limit approaching: {req_hour}/{req_hour_limit}/hour (threshold: {req_hour_threshold})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        if req_day >= req_day_threshold:
            reason = f"Request limit approaching: {req_day}/{req_day_limit}/day (threshold: {req_day_threshold})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        # Check token limits
        token_min = self.get_tokens_in_minute()
        token_hour = self.get_tokens_in_hour()
        token_day = self.get_tokens_in_day()
        
        token_min_limit = self.config.tokens_per_minute
        token_hour_limit = self.config.tokens_per_hour
        token_day_limit = self.config.tokens_per_day
        
        # Calculate thresholds
        token_min_threshold = int(token_min_limit * self.config.token_fallback_threshold)
        token_hour_threshold = int(token_hour_limit * self.config.token_fallback_threshold)
        token_day_threshold = int(token_day_limit * self.config.token_fallback_threshold)
        
        # Check token thresholds
        if token_min >= token_min_threshold:
            reason = f"Token limit approaching: {token_min:,}/{token_min_limit:,}/min (threshold: {token_min_threshold:,})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        if token_hour >= token_hour_threshold:
            reason = f"Token limit approaching: {token_hour:,}/{token_hour_limit:,}/hour (threshold: {token_hour_threshold:,})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        if token_day >= token_day_threshold:
            reason = f"Token limit approaching: {token_day:,}/{token_day_limit:,}/day (threshold: {token_day_threshold:,})"
            logger.warning(reason)
            self._fallback_count += 1
            return False, reason
        
        return True, "Rate limits are safe"
    
    def record_rate_limit_hit(self):
        """Record when a 429 rate limit error is hit."""
        self._rate_limit_hits += 1
        logger.warning(f"Rate limit hit! Total hits: {self._rate_limit_hits}")
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        return {
            "current": {
                "requests": {
                    "minute": self.get_requests_in_minute(),
                    "hour": self.get_requests_in_hour(),
                    "day": self.get_requests_in_day(),
                },
                "tokens": {
                    "minute": self.get_tokens_in_minute(),
                    "hour": self.get_tokens_in_hour(),
                    "day": self.get_tokens_in_day(),
                }
            },
            "limits": {
                "requests": {
                    "minute": self.config.requests_per_minute,
                    "hour": self.config.requests_per_hour,
                    "day": self.config.requests_per_day,
                },
                "tokens": {
                    "minute": self.config.tokens_per_minute,
                    "hour": self.config.tokens_per_hour,
                    "day": self.config.tokens_per_day,
                }
            },
            "thresholds": {
                "requests": {
                    "minute": int(self.config.requests_per_minute * self.config.request_fallback_threshold),
                    "hour": int(self.config.requests_per_hour * self.config.request_fallback_threshold),
                    "day": int(self.config.requests_per_day * self.config.request_fallback_threshold),
                },
                "tokens": {
                    "minute": int(self.config.tokens_per_minute * self.config.token_fallback_threshold),
                    "hour": int(self.config.tokens_per_hour * self.config.token_fallback_threshold),
                    "day": int(self.config.tokens_per_day * self.config.token_fallback_threshold),
                }
            },
            "statistics": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "fallback_count": self._fallback_count,
                "rate_limit_hits": self._rate_limit_hits,
            }
        }
    
    def log_status(self):
        """Log current rate limit status."""
        status = self.get_status()
        logger.info("=== RATE LIMIT STATUS ===")
        logger.info(f"Requests (current/limit):")
        logger.info(f"  Minute: {status['current']['requests']['minute']}/{status['limits']['requests']['minute']} (threshold: {status['thresholds']['requests']['minute']})")
        logger.info(f"  Hour:   {status['current']['requests']['hour']}/{status['limits']['requests']['hour']} (threshold: {status['thresholds']['requests']['hour']})")
        logger.info(f"  Day:    {status['current']['requests']['day']}/{status['limits']['requests']['day']} (threshold: {status['thresholds']['requests']['day']})")
        logger.info(f"Tokens (current/limit):")
        logger.info(f"  Minute: {status['current']['tokens']['minute']:,}/{status['limits']['tokens']['minute']:,} (threshold: {status['thresholds']['tokens']['minute']:,})")
        logger.info(f"  Hour:   {status['current']['tokens']['hour']:,}/{status['limits']['tokens']['hour']:,} (threshold: {status['thresholds']['tokens']['hour']:,})")
        logger.info(f"  Day:    {status['current']['tokens']['day']:,}/{status['limits']['tokens']['day']:,} (threshold: {status['thresholds']['tokens']['day']:,})")
        logger.info(f"Statistics: total_requests={status['statistics']['total_requests']}, fallback_count={status['statistics']['fallback_count']}, rate_limit_hits={status['statistics']['rate_limit_hits']}")