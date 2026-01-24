"""Statistics tracking for the API."""

from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from threading import Lock


class StatisticsTracker:
    """Track API usage statistics."""

    def __init__(self):
        self._lock = Lock()

        # Request counts
        self.total_requests = 0
        self.requests_by_model = {
            "local": 0,
            "cerebras": 0,
        }

        # Token usage
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokens_by_model = {
            "local": {"prompt": 0, "completion": 0, "total": 0},
            "cerebras": {"prompt": 0, "completion": 0, "total": 0},
        }

        # Speculative decoding stats
        self.speculative_decoding_stats = {
            "total_drafts": 0,
            "total_verifications": 0,
            "drafts_accepted": 0,
            "drafts_rejected": 0,
            "total_draft_tokens": 0,
            "total_verification_tokens": 0,
            "tokens_saved": 0,
        }

        # Rate limits
        self.rate_limit_hits = 0

        # Errors
        self.errors = {
            "local": 0,
            "cerebras": 0,
            "total": 0,
        }

        # Start time
        self.start_time = datetime.now().isoformat()

    def record_request(self, model_used: str, response: Dict[str, Any]):
        """Record a request and its token usage."""
        with self._lock:
            self.total_requests += 1
            self.requests_by_model[model_used] += 1

            # Extract token usage
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Update totals
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            # Update per-model stats
            self.tokens_by_model[model_used]["prompt"] += prompt_tokens
            self.tokens_by_model[model_used]["completion"] += completion_tokens
            self.tokens_by_model[model_used]["total"] += total_tokens

    def record_speculative_decoding(
        self, draft_tokens: int, verified_tokens: int, accepted: bool
    ):
        """Record speculative decoding stats."""
        with self._lock:
            self.speculative_decoding_stats["total_drafts"] += 1
            self.speculative_decoding_stats["total_draft_tokens"] += draft_tokens

            if verified_tokens > 0:
                self.speculative_decoding_stats["total_verifications"] += 1
                self.speculative_decoding_stats["total_verification_tokens"] += (
                    verified_tokens
                )

            if accepted:
                self.speculative_decoding_stats["drafts_accepted"] += 1
                # Tokens saved = verification tokens that didn't need to be used
                self.speculative_decoding_stats["tokens_saved"] += verified_tokens
            else:
                self.speculative_decoding_stats["drafts_rejected"] += 1

    def record_error(self, model: str):
        """Record an error."""
        with self._lock:
            self.errors[model] += 1
            self.errors["total"] += 1

    def record_rate_limit(self):
        """Record a rate limit hit."""
        with self._lock:
            self.rate_limit_hits += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            # Calculate costs
            cerebras_cost_per_1k = 0.002  # From config
            cerebras_tokens = self.tokens_by_model["cerebras"]["total"]
            total_cost = (cerebras_tokens / 1000) * cerebras_cost_per_1k

            # Calculate savings from speculative decoding
            tokens_saved = self.speculative_decoding_stats["tokens_saved"]
            cost_saved = (tokens_saved / 1000) * cerebras_cost_per_1k

            # Calculate acceptance rate
            total_verifications = self.speculative_decoding_stats["total_verifications"]
            acceptance_rate = (
                (
                    self.speculative_decoding_stats["drafts_accepted"]
                    / total_verifications
                    * 100
                )
                if total_verifications > 0
                else 0
            )

            return {
                "uptime": self._calculate_uptime(),
                "requests": {
                    "total": self.total_requests,
                    "by_model": self.requests_by_model,
                },
                "tokens": {
                    "total": self.total_tokens,
                    "prompt": self.prompt_tokens,
                    "completion": self.completion_tokens,
                    "by_model": self.tokens_by_model,
                },
                "costs": {
                    "total_cost_usd": round(total_cost, 4),
                    "cost_saved_usd": round(cost_saved, 4),
                    "cerebras_cost_per_1k_tokens": cerebras_cost_per_1k,
                },
                "speculative_decoding": {
                    "total_drafts": self.speculative_decoding_stats["total_drafts"],
                    "total_verifications": self.speculative_decoding_stats[
                        "total_verifications"
                    ],
                    "drafts_accepted": self.speculative_decoding_stats[
                        "drafts_accepted"
                    ],
                    "drafts_rejected": self.speculative_decoding_stats[
                        "drafts_rejected"
                    ],
                    "acceptance_rate_percent": round(acceptance_rate, 2),
                    "total_draft_tokens": self.speculative_decoding_stats[
                        "total_draft_tokens"
                    ],
                    "total_verification_tokens": self.speculative_decoding_stats[
                        "total_verification_tokens"
                    ],
                    "tokens_saved": tokens_saved,
                    "tokens_saved_percent": round(
                        (tokens_saved / self.total_tokens * 100)
                        if self.total_tokens > 0
                        else 0,
                        2,
                    ),
                },
                "errors": {
                    "total": self.errors["total"],
                    "by_model": {
                        "local": self.errors["local"],
                        "cerebras": self.errors["cerebras"],
                    },
                },
                "rate_limits": {
                    "total_hits": self.rate_limit_hits,
                },
                "performance": {
                    "avg_tokens_per_request": round(
                        self.total_tokens / self.total_requests, 2
                    )
                    if self.total_requests > 0
                    else 0,
                },
            }

    def _calculate_uptime(self) -> str:
        """Calculate uptime since start."""
        start = datetime.fromisoformat(self.start_time)
        now = datetime.now()
        uptime = now - start

        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self.__init__()

    def save_to_file(self, filepath: str):
        """Save statistics to a JSON file."""
        with self._lock:
            stats = self.get_statistics()
            stats["saved_at"] = datetime.now().isoformat()
            stats["start_time"] = self.start_time

            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load statistics from a JSON file."""
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        with self._lock:
            self.total_requests = data.get("requests", {}).get("total", 0)
            self.requests_by_model = data.get("requests", {}).get(
                "by_model", self.requests_by_model
            )

            tokens = data.get("tokens", {})
            self.total_tokens = tokens.get("total", 0)
            self.prompt_tokens = tokens.get("prompt", 0)
            self.completion_tokens = tokens.get("completion", 0)
            self.tokens_by_model = tokens.get("by_model", self.tokens_by_model)

            self.speculative_decoding_stats = data.get(
                "speculative_decoding", self.speculative_decoding_stats
            )
            self.errors = data.get("errors", self.errors)
            self.rate_limit_hits = data.get("rate_limits", {}).get("total_hits", 0)
            self.start_time = data.get("start_time", datetime.now().isoformat())


# Global statistics instance
_stats: Optional[StatisticsTracker] = None


def get_stats() -> StatisticsTracker:
    """Get the global statistics instance."""
    global _stats
    if _stats is None:
        _stats = StatisticsTracker()
    return _stats
