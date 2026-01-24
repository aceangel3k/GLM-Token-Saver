from typing import Dict, Any, Optional, Tuple
import logging
from config import get_config
from models import LocalModelClient, CerebrasModelClient

logger = logging.getLogger(__name__)


class TaskComplexityClassifier:
    """Classify task complexity based on prompt analysis."""

    def __init__(self):
        config = get_config()
        self.threshold = config.routing.simple_task_threshold
        self.complexity_keywords = config.routing.complexity_keywords

    def classify(self, messages: list[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Classify if task is complex.

        Returns:
            Tuple of (is_complex, reason)
        """
        # Combine all messages into a single text
        full_text = " ".join([msg.get("content", "") for msg in messages])

        # Check for complexity keywords
        text_lower = full_text.lower()
        found_keywords = [kw for kw in self.complexity_keywords if kw in text_lower]

        if found_keywords:
            reason = f"Found complexity keywords: {', '.join(found_keywords)}"
            logger.debug(f"Task classified as complex: {reason}")
            return True, reason

        # Check token count (simple heuristic)
        estimated_tokens = len(full_text.split()) * 1.3  # Rough estimate
        if estimated_tokens > self.threshold:
            reason = f"Estimated tokens ({estimated_tokens:.0f}) exceed threshold ({self.threshold})"
            logger.debug(f"Task classified as complex: {reason}")
            return True, reason

        logger.debug("Task classified as simple")
        return False, "Simple task based on analysis"


class SmartRouter:
    """Route requests to appropriate model based on strategy."""

    def __init__(self):
        self.config = get_config()
        self.classifier = TaskComplexityClassifier()

        # Initialize model clients
        self.local_client = LocalModelClient(self.config.models["local"].model_dump())
        self.cerebras_client = CerebrasModelClient(
            self.config.models["cerebras"].model_dump()
        )

        # Track rate limits
        self.rate_limit_hits = 0
        self.last_rate_limit_time = None

    async def route(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route request to appropriate model based on strategy.

        Returns:
            Response from the selected model
        """
        strategy = self.config.routing.strategy

        if strategy == "always_local":
            return await self._use_local(messages, temperature, max_tokens, **kwargs)

        elif strategy == "always_cerebras":
            return await self._use_cerebras(messages, temperature, max_tokens, **kwargs)

        elif strategy == "smart_routing":
            return await self._smart_routing(
                messages, temperature, max_tokens, **kwargs
            )

        elif strategy == "speculative_decoding":
            return await self._speculative_decoding(
                messages, temperature, max_tokens, **kwargs
            )

        else:
            logger.warning(
                f"Unknown strategy: {strategy}, falling back to smart_routing"
            )
            return await self._smart_routing(
                messages, temperature, max_tokens, **kwargs
            )

    async def _smart_routing(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Smart routing based on task complexity."""
        is_complex, reason = self.classifier.classify(messages)

        if is_complex:
            logger.info(f"=== ROUTING DECISION ===")
            logger.info(f"Routing to Cerebras: {reason}")
            try:
                return await self._use_cerebras(
                    messages, temperature, max_tokens, **kwargs
                )
            except Exception as e:
                logger.warning(f"Cerebras failed, falling back to local: {e}")
                return await self._use_local(
                    messages, temperature, max_tokens, **kwargs
                )
        else:
            logger.info(f"=== ROUTING DECISION ===")
            logger.info(f"Routing to local: {reason}")
            return await self._use_local(messages, temperature, max_tokens, **kwargs)

    async def _speculative_decoding(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Use speculative decoding: draft from local, verify with Cerebras."""
        if not self.config.speculative_decoding.enabled:
            logger.debug("Speculative decoding disabled, using smart routing")
            return await self._smart_routing(
                messages, temperature, max_tokens, **kwargs
            )

        logger.debug("Using speculative decoding")

        # Generate draft from local model
        draft_response = await self._use_local(
            messages, temperature, max_tokens, **kwargs
        )

        # For now, return the draft response
        # In a full implementation, we would verify with Cerebras
        # and only send corrections when needed
        logger.debug("Speculative decoding: returning draft from local model")
        return draft_response

    async def _use_local(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Use local model."""
        try:
            response = await self.local_client.chat_completion(
                messages, temperature, max_tokens, **kwargs
            )
            response["model_used"] = "local"
            return response
        except Exception as e:
            logger.error(f"Local model failed: {e}")
            raise

    async def _use_cerebras(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Use Cerebras model."""
        try:
            response = await self.cerebras_client.chat_completion(
                messages, temperature, max_tokens, **kwargs
            )
            response["model_used"] = "cerebras"
            return response
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error (429)
            if "429" in error_str or "Too Many Requests" in error_str:
                self.rate_limit_hits += 1
                self.last_rate_limit_time = None
                logger.warning(
                    f"Cerebras rate limit hit (count: {self.rate_limit_hits}): {e}"
                )
                # Re-raise to allow fallback to local model
                raise
            else:
                logger.error(f"Cerebras model failed: {e}")
                raise
