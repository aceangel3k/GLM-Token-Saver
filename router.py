from typing import Dict, Any, Optional, Tuple, Callable
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
        # Extract user messages only (ignore system messages for complexity detection)
        user_messages = [msg for msg in messages if msg.get("role") == "user"]

        # Combine user messages into a single text
        user_text = " ".join([msg.get("content", "") for msg in user_messages])

        # Combine all messages for token count estimation
        full_text = " ".join([msg.get("content", "") for msg in messages])

        # Check for complexity keywords in user messages only
        text_lower = user_text.lower()
        found_keywords = [kw for kw in self.complexity_keywords if kw in text_lower]

        if found_keywords:
            reason = f"Found complexity keywords in user messages: {', '.join(found_keywords)}"
            logger.debug(f"Task classified as complex: {reason}")
            return True, reason

        # Check token count (simple heuristic) - use all messages
        estimated_tokens = len(full_text.split()) * 1.3  # Rough estimate
        if estimated_tokens > self.threshold:
            reason = f"Estimated tokens ({estimated_tokens:.0f}) exceed threshold ({self.threshold})"
            logger.debug(f"Task classified as complex: {reason}")
            return True, reason

        logger.debug("Task classified as simple")
        return False, "Simple task based on analysis"


class SmartRouter:
    """Route requests to appropriate model based on strategy."""

    def __init__(self, get_stats: Callable):
        self.get_stats = get_stats
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

        elif strategy == "smart_speculative":
            return await self._smart_speculative(
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

    async def _smart_speculative(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Smart routing with speculative decoding for complex tasks."""
        is_complex, reason = self.classifier.classify(messages)

        if is_complex:
            logger.info(f"=== ROUTING DECISION ===")
            logger.info(f"Complex task, using speculative decoding: {reason}")
            try:
                return await self._speculative_decoding(
                    messages, temperature, max_tokens, **kwargs
                )
            except Exception as e:
                logger.warning(
                    f"Speculative decoding failed, falling back to local: {e}"
                )
                return await self._use_local(
                    messages, temperature, max_tokens, **kwargs
                )
        else:
            logger.info(f"=== ROUTING DECISION ===")
            logger.info(f"Simple task, using local directly: {reason}")
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

        logger.info("=== SPECULATIVE DECODING START ===")

        spec_config = self.config.speculative_decoding
        max_draft_tokens = spec_config.max_draft_tokens
        min_confidence = spec_config.min_confidence

        # Step 1: Generate draft from local model with limited tokens
        logger.info(
            f"Step 1: Generating draft from local model (max {max_draft_tokens} tokens)"
        )
        draft_response = await self._use_local(
            messages,
            temperature,
            min(max_draft_tokens, max_tokens) if max_tokens else max_draft_tokens,
            **kwargs,
        )

        # Extract draft content
        draft_content = self._extract_content(draft_response)
        draft_tokens = draft_response.get("usage", {}).get("completion_tokens", 0)

        logger.info(
            f"Draft generated: {draft_tokens} tokens, {len(draft_content)} chars"
        )

        # Step 2: Check if we need verification
        should_verify = self._should_verify_draft(
            draft_content, draft_tokens, max_tokens
        )

        if not should_verify:
            logger.info("Draft accepted without verification (short/simple response)")
            draft_response["model_used"] = "local"
            draft_response["speculative_decoding"] = {
                "draft_tokens": draft_tokens,
                "verified_tokens": 0,
                "accepted": True,
                "spilled_over": False,
            }
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, 0, True)
            return draft_response

        # Step 3: Verify with Cerebras
        logger.info("Step 2: Verifying draft with Cerebras model")
        try:
            verification_response = await self._use_cerebras(
                messages, temperature, max_tokens, **kwargs
            )

            verify_content = self._extract_content(verification_response)
            verify_tokens = verification_response.get("usage", {}).get(
                "completion_tokens", 0
            )

            logger.info(
                f"Verification response: {verify_tokens} tokens, {len(verify_content)} chars"
            )

            # Step 4: Compare and merge responses
            similarity = self._calculate_similarity(draft_content, verify_content)
            logger.info(f"Similarity between draft and verification: {similarity:.2f}")

            # Determine which response to use
            if similarity >= min_confidence:
                # Responses are similar enough, use draft (saves tokens)
                logger.info(f"Responses similar ({similarity:.2f}), using draft")
                merged_response = draft_response
                merged_response["model_used"] = "local"
                spilled_over = False
                accepted = True
            else:
                # Responses differ significantly, use verification
                logger.info(
                    f"Responses differ ({similarity:.2f}), using verified response"
                )
                merged_response = verification_response
                merged_response["model_used"] = "cerebras"
                spilled_over = True
                accepted = False

            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, verify_tokens, accepted)

            merged_response["speculative_decoding"] = {
                "draft_tokens": draft_tokens,
                "verified_tokens": verify_tokens,
                "accepted": not spilled_over,
                "spilled_over": spilled_over,
                "similarity": similarity,
            }

            logger.info(f"=== SPECULATIVE DECODING COMPLETE ===")
            logger.info(
                f"Draft accepted: {merged_response['speculative_decoding']['accepted']}"
            )
            logger.info(
                f"Spilled over: {merged_response['speculative_decoding']['spilled_over']}"
            )

            return merged_response

        except Exception as e:
            logger.warning(f"Cerebras verification failed: {e}, using draft")
            draft_response["model_used"] = "local"
            draft_response["speculative_decoding"] = {
                "draft_tokens": draft_tokens,
                "verified_tokens": 0,
                "accepted": True,
                "spilled_over": False,
                "error": str(e),
            }
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
            stats = self.get_stats()
            stats.record_request("local", response)
            logger.info(
                f"Recorded local request to statistics. Total requests: {stats.total_requests}"
            )
            return response
        except Exception as e:
            logger.error(f"Local model failed: {e}")
            stats = self.get_stats()
            stats.record_error("local")
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
            stats = self.get_stats()
            stats.record_request("cerebras", response)
            return response
        except Exception as e:
            error_str = str(e)
            stats = self.get_stats()
            # Check if it's a rate limit error (429)
            if "429" in error_str or "Too Many Requests" in error_str:
                self.rate_limit_hits += 1
                self.last_rate_limit_time = None
                stats.record_rate_limit()
                stats.record_error("cerebras")
                logger.warning(
                    f"Cerebras rate limit hit (count: {self.rate_limit_hits}): {e}"
                )
                # Re-raise to allow fallback to local model
                raise
            else:
                stats.record_error("cerebras")
                logger.error(f"Cerebras model failed: {e}")
                raise

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from response."""
        try:
            choices = response.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})

                # Try content field first
                content = message.get("content", "")

                # If content is empty, try reasoning_content (some models use this)
                if not content:
                    content = message.get("reasoning_content", "")

                # If still empty, try reasoning field (Cerebras uses this)
                if not content:
                    content = message.get("reasoning", "")

                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    return "".join(str(item) for item in content)
                else:
                    return str(content)
            return ""
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return ""

    def _should_verify_draft(
        self,
        draft_content: str,
        draft_tokens: int,
        max_tokens: Optional[int],
    ) -> bool:
        """Determine if draft should be verified with Cerebras."""
        spec_config = self.config.speculative_decoding

        # Always verify if draft hit the max draft token limit (incomplete)
        if draft_tokens >= spec_config.max_draft_tokens:
            logger.info(
                f"Draft hit limit ({draft_tokens} >= {spec_config.max_draft_tokens}), will verify"
            )
            return True

        # Verify if draft is very short (might be incomplete)
        if len(draft_content) < 50:
            logger.info("Draft very short, will verify")
            return True

        # Verify if draft contains uncertain markers
        uncertain_markers = ["...", "I think", "maybe", "possibly", "unclear"]
        if any(marker in draft_content.lower() for marker in uncertain_markers):
            logger.info("Draft contains uncertain markers, will verify")
            return True

        # Don't verify if draft is complete and within token limit
        logger.info(
            f"Draft complete ({draft_tokens} < {spec_config.max_draft_tokens}), skipping verification"
        )
        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple Jaccard-like)."""
        if not text1 or not text2:
            logger.debug(
                f"Similarity calculation: empty text (text1={len(text1)}, text2={len(text2)})"
            )
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            logger.debug(
                f"Similarity calculation: no words (words1={len(words1)}, words2={len(words2)})"
            )
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0.0
        logger.debug(
            f"Similarity: {similarity:.2f} (intersection={len(intersection)}, union={len(union)})"
        )
        return similarity
