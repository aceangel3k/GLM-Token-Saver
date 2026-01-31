from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
import httpx
import logging
import json

logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """Base class for model clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config["endpoint"]
        self.api_key = config.get("api_key", "")
        self.model_name = config["model"]
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self._last_response_headers: Dict[str, str] = {}

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send chat completion request to the model."""
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send streaming chat completion request to the model."""
        pass

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Build request payload."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        payload.update(kwargs)
        return payload

    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to the model endpoint."""
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(f"Making HTTP request to: {self.endpoint}")
                response = await client.post(
                    self.endpoint, headers=headers, json=payload
                )
                response.raise_for_status()
                logger.info(f"HTTP request successful: {response.status_code}")
                # Store response headers for rate limit tracking
                self._last_response_headers = dict(response.headers)
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from {self.model_name}: {e}")
                logger.error(
                    f"Response body: {e.response.text if e.response else 'No response body'}"
                )
                raise
            except httpx.RequestError as e:
                logger.error(f"Request error to {self.model_name}: {e}")
                raise

    async def _make_streaming_request(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make streaming HTTP request to the model endpoint."""
        headers = self._get_headers()
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(f"Making streaming HTTP request to: {self.endpoint}")
                async with client.stream(
                    "POST", self.endpoint, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    logger.info(f"Streaming HTTP request successful: {response.status_code}")
                    # Store response headers for rate limit tracking
                    self._last_response_headers = dict(response.headers)
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        
                        # SSE format: "data: {...}"
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            # Check for [DONE] marker
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                yield data
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE data: {e}, data: {data_str}")
                                continue
                                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from {self.model_name} during streaming: {e}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Request error to {self.model_name} during streaming: {e}")
                raise
    
    def get_last_response_headers(self) -> Dict[str, str]:
        """Get headers from the last API response."""
        return self._last_response_headers.copy()
