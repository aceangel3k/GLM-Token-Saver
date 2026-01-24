from typing import Dict, Any, Optional
from .base import BaseModelClient
import logging

logger = logging.getLogger(__name__)


class CerebrasModelClient(BaseModelClient):
    """Client for Cerebras GLM 4.7 model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.debug(f"Initialized Cerebras model client: {self.model_name}")

    async def chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send chat completion request to Cerebras API."""
        payload = self._build_payload(messages, temperature, max_tokens, **kwargs)

        logger.debug(f"Sending request to Cerebras model: {self.model_name}")
        response = await self._make_request(payload)

        # Extract token usage
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.debug(
            f"Cerebras model response: {total_tokens} tokens "
            f"(prompt: {prompt_tokens}, completion: {completion_tokens})"
        )

        return response

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with Cerebras-specific headers."""
        headers = super()._get_headers()
        # Add any Cerebras-specific headers if needed
        return headers
