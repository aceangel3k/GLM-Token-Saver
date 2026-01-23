from typing import Dict, Any, Optional
from .base import BaseModelClient
import logging

logger = logging.getLogger(__name__)


class LocalModelClient(BaseModelClient):
    """Client for local llama.cpp model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"Initialized local model client: {self.model_name}")
    
    async def chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request to local llama.cpp model."""
        payload = self._build_payload(messages, temperature, max_tokens, **kwargs)
        
        logger.info(f"Sending request to local model: {self.model_name}")
        response = await self._make_request(payload)
        
        # Extract token usage if available
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        logger.info(
            f"Local model response: {total_tokens} tokens "
            f"(prompt: {prompt_tokens}, completion: {completion_tokens})"
        )
        
        return response
