from typing import Dict, Any, Optional, AsyncGenerator
from .base import BaseModelClient
import logging

logger = logging.getLogger(__name__)


class LocalModelClient(BaseModelClient):
    """Client for local llama.cpp model."""
    
    # Parameters that llama.cpp server doesn't support - filter these out
    UNSUPPORTED_PARAMS = {
        'stream_options', 'tools', 'tool_choice', 'response_format',
        'parallel_tool_calls', 'service_tier', 'logprobs', 'top_logprobs',
        'n', 'user', 'metadata', 'store', 'reasoning_effort'
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.debug(f"Initialized local model client: {self.model_name}")
    
    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out unsupported parameters for llama.cpp."""
        filtered = {k: v for k, v in kwargs.items() if k not in self.UNSUPPORTED_PARAMS}
        removed = set(kwargs.keys()) - set(filtered.keys())
        if removed:
            logger.debug(f"Filtered unsupported params for local model: {removed}")
        return filtered

    async def chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send chat completion request to local llama.cpp model."""
        kwargs = self._filter_kwargs(kwargs)
        payload = self._build_payload(messages, temperature, max_tokens, **kwargs)

        logger.info(f"=== LOCAL MODEL REQUEST ===")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Messages: {len(messages)}")
        logger.info(
            f"Temperature: {temperature}, Max tokens: {max_tokens or self.max_tokens}"
        )
        logger.debug(f"Full payload: {payload}")

        response = await self._make_request(payload)

        # Extract token usage if available
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info(f"=== LOCAL MODEL RESPONSE ===")
        logger.info(f"Status: Success")
        logger.info(
            f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})"
        )
        logger.debug(f"Full response: {response}")

        return response

    async def chat_completion_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send streaming chat completion request to local llama.cpp model."""
        kwargs = self._filter_kwargs(kwargs)
        payload = self._build_payload(messages, temperature, max_tokens, **kwargs)

        logger.info(f"=== LOCAL MODEL STREAMING REQUEST ===")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Messages: {len(messages)}")
        logger.info(
            f"Temperature: {temperature}, Max tokens: {max_tokens or self.max_tokens}"
        )

        async for chunk in self._make_streaming_request(payload):
            yield chunk
