from typing import Dict, Any, Optional, Tuple, Callable, AsyncGenerator
import logging
import asyncio
import re
import time
from config import get_config
from models import LocalModelClient, CerebrasModelClient
from rate_limiter import RateLimiter
from cache import ResponseCache

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when Cerebras rate limits are exceeded."""
    pass


class TaskComplexityClassifier:
    """
    Classify task complexity for coding assistant routing.
    
    Philosophy:
    - Local model: FREE but slow (62 tk/s) - use for simple/quick tasks
    - Cerebras: FAST (1000+ tk/s) but costs money - use for complex tasks
    
    Simple tasks (→ local):
    - Greetings, acknowledgments, short questions
    - Quick fixes, typos, small edits
    - Simple questions about existing code
    - Short clarifications, confirmations
    
    Complex tasks (→ Cerebras):
    - Multi-file refactoring or changes
    - New feature implementation
    - Architecture design discussions
    - Debugging complex issues
    - Large code generation (new files, classes)
    """

    def __init__(self):
        config = get_config()
        self.threshold = config.routing.simple_task_threshold
        
        # Patterns that indicate SIMPLE tasks (prefer local)
        self.simple_patterns = [
            "hi", "hello", "hey", "thanks", "thank you", "yes", "no", "ok", "okay",
            "got it", "sounds good", "perfect", "great", "nice", "cool", "sure",
            "please", "can you", "could you", "what is", "what's", "where is",
        ]
        
        # Quick task indicators (prefer local)
        self.quick_task_patterns = [
            "fix this", "fix the", "add a comment", "rename", "typo",
            "change this", "update this", "small change", "minor",
            "quick", "simple", "just", "only",
        ]
        
        # Complex task indicators (prefer Cerebras for speed)
        self.complex_patterns = [
            "refactor", "restructure", "redesign", "architect", "implement",
            "create a new", "build a", "develop", "design", "integrate",
            "migrate", "convert", "rewrite", "overhaul",
            "multiple files", "across files", "all files", "entire",
            "debug this", "fix the bug", "troubleshoot",
            "analyze", "review", "audit",
        ]
        
        # Scale indicators (bigger = Cerebras)
        self.scale_patterns = [
            "all", "every", "entire", "whole", "complete", "full",
            "multiple", "several", "many", "across",
        ]

    def classify(self, messages: list[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Classify if task is complex. Biased toward LOCAL for cost savings.
        Only routes to Cerebras when task genuinely benefits from speed.
        
        IMPORTANT: Only analyzes the LAST USER MESSAGE, not system prompts.
        """
        # Filter to user messages only (ignore system prompts which contain keywords like "architect")
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        if not user_messages:
            return False, "No user messages"
        
        # ONLY look at the last user message - this is what the user actually typed
        last_user_msg = user_messages[-1].get("content", "") if user_messages else ""
        
        # Handle case where content might be a list (multimodal)
        if isinstance(last_user_msg, list):
            last_user_msg = " ".join([
                item.get("text", "") for item in last_user_msg 
                if isinstance(item, dict) and item.get("type") == "text"
            ])
        
        # Extract actual task from Cline's XML tags if present
        # Cline wraps user input in <task>...</task> or <answer>...</answer> with extra instructions outside
        import re
        
        # Try <task> tags first (initial requests)
        task_match = re.search(r'<task>\s*(.*?)\s*</task>', last_user_msg, re.DOTALL | re.IGNORECASE)
        if task_match:
            last_user_msg = task_match.group(1).strip()
            logger.info(f"Extracted from <task> tags: {last_user_msg[:100]}")
        else:
            # Try <answer> tags (follow-up question responses)
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', last_user_msg, re.DOTALL | re.IGNORECASE)
            if answer_match:
                last_user_msg = answer_match.group(1).strip()
                logger.info(f"Extracted from <answer> tags: {last_user_msg[:100]}")
        
        last_msg_lower = last_user_msg.lower().strip()
        last_msg_words = len(last_msg_lower.split())
        
        # DEBUG: Log what we're actually analyzing
        logger.info(f"=== CLASSIFIER DEBUG ===")
        logger.info(f"Analyzing ({last_msg_words} words): {last_msg_lower[:200]}")
        
        # === SIMPLE TASK CHECKS (prefer local) ===
        
        # Very short messages almost always simple
        if last_msg_words <= 5:
            if any(p in last_msg_lower for p in self.simple_patterns):
                return False, "Short simple message"
            # Even without patterns, very short = simple
            return False, "Very short message"
        
        # Quick task keywords = simple
        if any(p in last_msg_lower for p in self.quick_task_patterns):
            # Unless also has scale indicators
            if not any(s in last_msg_lower for s in self.scale_patterns):
                return False, "Quick task request"
        
        # Short to medium messages without complexity indicators = simple
        if last_msg_words <= 30:
            has_complex = any(p in last_msg_lower for p in self.complex_patterns)
            if not has_complex:
                return False, "Short request without complexity indicators"
        
        # === COMPLEX TASK CHECKS (use Cerebras for speed) ===
        
        # Explicit complex task patterns
        for pattern in self.complex_patterns:
            if pattern in last_msg_lower:
                return True, f"Complex task: '{pattern}'"
        
        # Scale + action = complex
        has_scale = any(s in last_msg_lower for s in self.scale_patterns)
        action_words = ["change", "update", "fix", "modify", "edit", "add", "remove", "delete"]
        has_action = any(a in last_msg_lower for a in action_words)
        if has_scale and has_action:
            return True, "Scaled action request"
        
        # Very long messages suggest complex task
        if last_msg_words > 100:
            return True, f"Long detailed request ({last_msg_words} words)"
        
        # Multiple code blocks in the request = providing context for complex task
        code_block_count = last_user_msg.count("```")
        if code_block_count >= 4:  # 2+ complete code blocks
            return True, "Multiple code blocks provided"
        
        # Default: prefer local (free)
        return False, "Default to local (cost savings)"


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

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

        # Initialize cache
        cache_config = self.config.cache
        self.cache = ResponseCache(
            enabled=cache_config.enabled,
            max_size=cache_config.max_size,
            default_ttl=cache_config.default_ttl,
            cache_by_temperature=cache_config.cache_by_temperature,
            cache_by_max_tokens=cache_config.cache_by_max_tokens,
        )

        # Track rate limits (legacy, kept for compatibility)
        self.rate_limit_hits = 0
        self.last_rate_limit_time = None

        # Command patterns for model override
        self.local_patterns = [r'/\s*local\s*$', r'/\s*force-local\s*$']
        self.remote_patterns = [r'/\s*remote\s*$', r'/\s*force-remote\s*$']

    def reload_config(self) -> None:
        """
        Reload configuration and reinitialize components that depend on it.
        
        This method should be called when config.yaml is modified and the
        /admin/config/reload endpoint is triggered.
        """
        logger.info("=== RELOADING ROUTER CONFIGURATION ===")
        
        # Reload the global config
        from config import reload_config
        reload_config()
        
        # Update local config reference
        self.config = get_config()
        logger.info(f"Config reloaded: strategy={self.config.routing.strategy}")
        
        # Reinitialize the classifier with new settings
        self.classifier = TaskComplexityClassifier()
        logger.info(f"Classifier reinitialized: threshold={self.classifier.threshold}")
        
        # Reinitialize model clients with new configuration
        self.local_client = LocalModelClient(self.config.models["local"].model_dump())
        self.cerebras_client = CerebrasModelClient(
            self.config.models["cerebras"].model_dump()
        )
        logger.info(f"Model clients reinitialized")
        logger.info(f"  Local endpoint: {self.config.models['local'].endpoint}")
        logger.info(f"  Cerebras model: {self.config.models['cerebras'].model}")
        
        # Reinitialize rate limiter with new thresholds
        self.rate_limiter = RateLimiter()
        logger.info(f"Rate limiter reinitialized")
        
        # Reinitialize cache with new settings
        cache_config = self.config.cache
        self.cache = ResponseCache(
            enabled=cache_config.enabled,
            max_size=cache_config.max_size,
            default_ttl=cache_config.default_ttl,
            cache_by_temperature=cache_config.cache_by_temperature,
            cache_by_max_tokens=cache_config.cache_by_max_tokens,
        )
        logger.info(f"Cache reinitialized: enabled={cache_config.enabled}, max_size={cache_config.max_size}")
        
        logger.info("=== ROUTER CONFIGURATION RELOAD COMPLETE ===")

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
        # Check cache first
        cached_response = self.cache.get(messages, temperature, max_tokens, **kwargs)
        if cached_response is not None:
            logger.info("=== CACHE HIT ===")
            return cached_response

        # Cache miss - route to model
        strategy = self.config.routing.strategy

        if strategy == "always_local":
            response = await self._use_local(messages, temperature, max_tokens, **kwargs)

        elif strategy == "always_cerebras":
            response = await self._use_cerebras(messages, temperature, max_tokens, **kwargs)

        elif strategy == "smart_routing":
            response = await self._smart_routing(
                messages, temperature, max_tokens, **kwargs
            )

        elif strategy == "smart_speculative":
            response = await self._smart_speculative(
                messages, temperature, max_tokens, **kwargs
            )

        elif strategy == "speculative_decoding":
            response = await self._speculative_decoding(
                messages, temperature, max_tokens, **kwargs
            )

        elif strategy == "adaptive_cerebras":
            response = await self._adaptive_cerebras(
                messages, temperature, max_tokens, **kwargs
            )

        else:
            logger.warning(
                f"Unknown strategy: {strategy}, falling back to smart_routing"
            )
            response = await self._smart_routing(
                messages, temperature, max_tokens, **kwargs
            )

        # Cache the response
        model_used = response.get("model_used", "unknown")
        self.cache.set(
            messages, response, temperature, max_tokens, model_used, **kwargs
        )

        return response

    async def route_with_override(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route request to appropriate model, checking for override commands in the prompt.

        Looks for commands like /local, /force-local, /remote, /force-remote
        in the last user message and overrides the routing accordingly.

        Returns:
            Response from the selected model
        """
        # Check for override commands in the last user message
        override_result = self._detect_override(messages)

        if override_result["override"]:
            # An override command was found
            messages_cleaned = override_result["messages"]
            model_type = override_result["model"]
            command = override_result["command"]

            logger.info(f"=== OVERRIDE DETECTED ===")
            logger.info(f"Command: {command}")
            logger.info(f"Forcing model: {model_type}")

            if model_type == "local":
                return await self._use_local(
                    messages_cleaned, temperature, max_tokens, **kwargs
                )
            elif model_type == "remote":
                return await self._use_cerebras(
                    messages_cleaned, temperature, max_tokens, bypass_rate_limit=True, **kwargs
                )
        else:
            # No override, use normal routing
            return await self.route(
                messages, temperature, max_tokens, **kwargs
            )

    async def route_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Route request with streaming response.

        For speculative decoding strategies, uses Silent Mode:
        - Streams the draft response (OpenAI compatible)
        - Runs verification in background
        - Logs metadata on server only (not sent to client)
        - Maintains full OpenAI API compatibility

        Yields:
            Streaming chunks from the selected model
        """
        strategy = self.config.routing.strategy

        # Handle speculative decoding with Silent Mode streaming
        if strategy == "smart_speculative":
            async for chunk in self._smart_speculative_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            return
        
        elif strategy == "speculative_decoding":
            # For speculative_decoding, fall back to local streaming with background verification
            async for chunk in self._speculative_decoding_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            return

        elif strategy == "adaptive_cerebras":
            async for chunk in self._adaptive_cerebras_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            return

        # Handle other strategies with streaming (with fallback support)
        if strategy == "always_local":
            try:
                async for chunk in self._use_local_stream(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk
            except Exception as e:
                logger.warning(f"Local model failed, falling back to Cerebras: {e}")
                async for chunk in self._use_cerebras_stream(
                    messages, temperature, max_tokens, bypass_rate_limit=True, **kwargs
                ):
                    yield chunk

        elif strategy == "always_cerebras":
            try:
                async for chunk in self._use_cerebras_stream(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk
            except Exception as e:
                logger.warning(f"Cerebras failed, falling back to local: {e}")
                async for chunk in self._use_local_stream(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk

        elif strategy == "smart_routing":
            is_complex, reason = self.classifier.classify(messages)
            if is_complex:
                logger.info(f"=== ROUTING DECISION (STREAM) ===")
                logger.info(f"Routing to Cerebras: {reason}")
                try:
                    async for chunk in self._use_cerebras_stream(
                        messages, temperature, max_tokens, **kwargs
                    ):
                        yield chunk
                except Exception as e:
                    logger.warning(f"Cerebras failed, falling back to local: {e}")
                    async for chunk in self._use_local_stream(
                        messages, temperature, max_tokens, **kwargs
                    ):
                        yield chunk
            else:
                logger.info(f"=== ROUTING DECISION (STREAM) ===")
                logger.info(f"Routing to local: {reason}")
                try:
                    async for chunk in self._use_local_stream(
                        messages, temperature, max_tokens, **kwargs
                    ):
                        yield chunk
                except Exception as e:
                    logger.warning(f"Local failed, falling back to Cerebras: {e}")
                    async for chunk in self._use_cerebras_stream(
                        messages, temperature, max_tokens, bypass_rate_limit=True, **kwargs
                    ):
                        yield chunk

        else:
            logger.warning(
                f"Unknown strategy: {strategy}, falling back to smart_routing (stream)"
            )
            is_complex, reason = self.classifier.classify(messages)
            if is_complex:
                async for chunk in self._use_cerebras_stream(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk
            else:
                async for chunk in self._use_local_stream(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk

    def _detect_override(
        self,
        messages: list[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Detect model override commands in the messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Dict with keys:
                - override: bool, whether an override was detected
                - messages: list of messages with commands stripped
                - model: str, "local" or "remote" (if override detected)
                - command: str, the command that was detected (if override detected)
        """
        # Find the last user message
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx == -1:
            # No user message found, no override
            return {
                "override": False,
                "messages": messages,
                "model": None,
                "command": None
            }

        # Check for local override patterns
        for pattern in self.local_patterns:
            match = re.search(pattern, messages[last_user_idx].get("content", ""), re.IGNORECASE)
            if match:
                # Strip the command from the message
                cleaned_content = re.sub(pattern, "", messages[last_user_idx].get("content", ""), flags=re.IGNORECASE).strip()
                cleaned_messages = messages.copy()
                cleaned_messages[last_user_idx] = cleaned_messages[last_user_idx].copy()
                cleaned_messages[last_user_idx]["content"] = cleaned_content

                return {
                    "override": True,
                    "messages": cleaned_messages,
                    "model": "local",
                    "command": match.group(0).strip()
                }

        # Check for remote override patterns
        for pattern in self.remote_patterns:
            match = re.search(pattern, messages[last_user_idx].get("content", ""), re.IGNORECASE)
            if match:
                # Strip the command from the message
                cleaned_content = re.sub(pattern, "", messages[last_user_idx].get("content", ""), flags=re.IGNORECASE).strip()
                cleaned_messages = messages.copy()
                cleaned_messages[last_user_idx] = cleaned_messages[last_user_idx].copy()
                cleaned_messages[last_user_idx]["content"] = cleaned_content

                return {
                    "override": True,
                    "messages": cleaned_messages,
                    "model": "remote",
                    "command": match.group(0).strip()
                }

        # No override found
        return {
            "override": False,
            "messages": messages,
            "model": None,
            "command": None
        }

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

        # Check if parallel speculative decoding is enabled
        if self.config.speculative_decoding.parallel_enabled:
            return await self._parallel_speculative_decoding(
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

    async def _parallel_speculative_decoding(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Use parallel speculative decoding: run draft and verification concurrently."""
        logger.info("=== PARALLEL SPECULATIVE DECODING START ===")

        spec_config = self.config.speculative_decoding
        max_draft_tokens = spec_config.max_draft_tokens
        min_confidence = spec_config.min_confidence
        draft_timeout = spec_config.draft_timeout

        # Step 1: Launch draft and verification requests concurrently
        logger.info(
            f"Step 1: Launching concurrent draft (max {max_draft_tokens} tokens) and verification requests "
            f"(timeout {draft_timeout}s)"
        )

        # Create tasks for draft and verification
        draft_task = asyncio.create_task(
            self._use_local(
                messages,
                temperature,
                min(max_draft_tokens, max_tokens) if max_tokens else max_draft_tokens,
                **kwargs,
            )
        )
        
        verify_task = asyncio.create_task(
            self._use_cerebras(
                messages,
                temperature,
                max_tokens,
                **kwargs,
            )
        )

        # Wait for draft to complete with timeout
        draft_response = None
        verification_response = None
        draft_completed = False
        
        try:
            draft_response = await asyncio.wait_for(draft_task, timeout=draft_timeout)
            draft_completed = True
            logger.info(f"Draft completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Draft generation timed out after {draft_timeout}s")
            # Cancel draft task if it's still running
            if not draft_task.done():
                draft_task.cancel()
        
        # Extract draft content if available
        draft_content = ""
        draft_tokens = 0
        
        if draft_response:
            draft_content = self._extract_content(draft_response)
            draft_tokens = draft_response.get("usage", {}).get("completion_tokens", 0)
            logger.info(f"Draft: {draft_tokens} tokens, {len(draft_content)} chars")

        # Step 2: Wait for verification to complete (it's already running)
        try:
            verification_response = await verify_task
            logger.info(f"Verification completed successfully")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            # If verification fails but we have a draft, use the draft
            if draft_response:
                logger.info("Using draft due to verification failure")
                draft_response["model_used"] = "local"
                draft_response["speculative_decoding"] = {
                    "parallel": True,
                    "draft_tokens": draft_tokens,
                    "verified_tokens": 0,
                    "accepted": True,
                    "spilled_over": False,
                    "verification_failed": True,
                }
                stats = self.get_stats()
                stats.record_speculative_decoding(draft_tokens, 0, True)
                return draft_response
            else:
                # Both failed, fall back to sequential approach
                return await self._speculative_decoding(
                    messages, temperature, max_tokens, **kwargs
                )

        # Step 3: If draft failed, use verification
        if not draft_completed or not draft_response:
            logger.info("Draft failed, using verification response")
            verify_content = self._extract_content(verification_response)
            verify_tokens = verification_response.get("usage", {}).get("completion_tokens", 0)
            
            verification_response["model_used"] = "cerebras"
            verification_response["speculative_decoding"] = {
                "parallel": True,
                "draft_tokens": draft_tokens,
                "verified_tokens": verify_tokens,
                "accepted": False,
                "spilled_over": True,
                "draft_failed": True,
            }
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, verify_tokens, False)
            return verification_response

        # Step 4: Both completed - compare and select best response
        verify_content = self._extract_content(verification_response)
        verify_tokens = verification_response.get("usage", {}).get("completion_tokens", 0)

        logger.info(
            f"Verification: {verify_tokens} tokens, {len(verify_content)} chars"
        )

        # Check if we should skip verification (draft looks complete)
        should_verify = self._should_verify_draft(
            draft_content, draft_tokens, max_tokens
        )

        if not should_verify:
            logger.info("Draft looks complete, accepting without comparison")
            draft_response["model_used"] = "local"
            draft_response["speculative_decoding"] = {
                "parallel": True,
                "draft_tokens": draft_tokens,
                "verified_tokens": verify_tokens,
                "accepted": True,
                "spilled_over": False,
                "skipped_verification": True,
            }
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, verify_tokens, True)
            return draft_response

        # Calculate similarity
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
            "parallel": True,
            "draft_tokens": draft_tokens,
            "verified_tokens": verify_tokens,
            "accepted": not spilled_over,
            "spilled_over": spilled_over,
            "similarity": similarity,
        }

        logger.info(f"=== PARALLEL SPECULATIVE DECODING COMPLETE ===")
        logger.info(
            f"Draft accepted: {merged_response['speculative_decoding']['accepted']}"
        )
        logger.info(
            f"Spilled over: {merged_response['speculative_decoding']['spilled_over']}"
        )

        return merged_response

    def _select_best_draft(
        self,
        drafts: list[Tuple[int, Dict[str, Any]]]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select the best draft from multiple candidates.

        Selection criteria:
        1. Prefer longer, more complete responses
        2. Prefer responses that look complete (end with proper punctuation)
        3. Prefer responses with good structure (multiple sentences/lists)
        """
        if not drafts:
            raise ValueError("No drafts to select from")

        # Score each draft
        scored_drafts = []
        for idx, draft in drafts:
            content = self._extract_content(draft)
            tokens = draft.get("usage", {}).get("completion_tokens", 0)
            
            # Calculate score
            score = 0
            
            # 1. Token count (more is better, up to a point)
            score += min(tokens, 100) / 100 * 30  # Max 30 points
            
            # 2. Content length (more is better, up to a point)
            score += min(len(content), 1000) / 1000 * 20  # Max 20 points
            
            # 3. Completeness check
            if self._draft_looks_complete(content):
                score += 30  # Max 30 points
            
            # 4. Structure bonus
            if '\n' in content:
                score += 10  # Has structure
            if content.count('. ') >= 2:
                score += 10  # Multiple sentences
            
            scored_drafts.append((score, idx, draft))
            logger.debug(f"Draft {idx}: score={score:.1f}, tokens={tokens}, chars={len(content)}")

        # Sort by score descending
        scored_drafts.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_idx, best_draft = scored_drafts[0]
        logger.info(f"Best draft selected: index={best_idx}, score={best_score:.1f}")
        
        return best_idx, best_draft

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
            response["_from_cache"] = False
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

    async def _use_local_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Use local model with streaming. Buffers from content closing tag to insert notification."""
        try:
            import time
            import re
            
            # Track content and detect when to start buffering
            full_content = ""
            buffer = ""
            buffering = False  # Start buffering when we see content closing tags
            completion_tokens = 0
            response_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            model_name = self.config.models["local"].model
            
            # Tags that indicate main content is ending (notification goes before these)
            content_end_patterns = ['</response>', '</question>', '</result>']
            
            async for chunk in self.local_client.chat_completion_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content") or ""
                    full_content += content
                    completion_tokens += len(content) // 4 + 1
                    
                    if "id" in chunk:
                        response_id = chunk["id"]
                    if "created" in chunk:
                        created = chunk["created"]
                    
                    if buffering:
                        # Already buffering, accumulate
                        buffer += content
                    else:
                        # Check if this chunk contains a content closing tag
                        combined = buffer + content
                        tag_found = None
                        tag_pos = -1
                        for tag in content_end_patterns:
                            pos = combined.lower().find(tag.lower())
                            if pos != -1 and (tag_pos == -1 or pos < tag_pos):
                                tag_pos = pos
                                tag_found = tag
                        
                        if tag_found:
                            # Found closing tag - send everything before it, buffer the rest
                            to_send = combined[:tag_pos]
                            buffer = combined[tag_pos:]
                            buffering = True
                            if to_send:
                                yield {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "model_used": "local",
                                    "choices": [{"index": 0, "delta": {"content": to_send}}],
                                }
                        else:
                            # No closing tag yet, stream normally but keep small lookahead
                            buffer += content
                            if len(buffer) > 50:
                                to_send = buffer[:-50]
                                buffer = buffer[-50:]
                                if to_send:
                                    yield {
                                        "id": response_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "model_used": "local",
                                        "choices": [{"index": 0, "delta": {"content": to_send}}],
                                    }
            
            # Stream complete - insert notification only for user-facing responses
            logger.info(f"Local stream complete. Buffering={buffering}, Buffer length: {len(buffer)}")
            config = self.config
            
            # Only add notification to user-facing responses, NOT tool calls
            # User-facing: plan_mode_respond, ask_followup_question, attempt_completion
            # Tool calls: write_to_file, replace_in_file, read_file, etc (skip notification)
            full_lower = full_content.lower()
            user_facing_tags = ['<plan_mode_respond>', '<ask_followup_question>', '<attempt_completion>']
            tool_use_tags = ['<write_to_file>', '<replace_in_file>', '<read_file>', '<execute_command>', 
                            '<search_files>', '<list_files>', '<list_code_definition_names>',
                            '<browser_action>', '<use_mcp_tool>', '<access_mcp_resource>']
            
            is_user_facing = any(tag in full_lower for tag in user_facing_tags)
            is_tool_use = any(tag in full_lower for tag in tool_use_tags)
            should_add_notification = is_user_facing and not is_tool_use
            
            logger.info(f"Notification check: user_facing={is_user_facing}, tool_use={is_tool_use}, add={should_add_notification}")
            
            if config.response_notifications.enabled and should_add_notification:
                from main import format_response_notification
                
                notification = format_response_notification(
                    model_used="local",
                    total_tokens=completion_tokens,
                    cost=0.0,
                    config_notifications=config.response_notifications,
                )
                logger.info(f"Notification generated: {notification[:50] if notification else 'None'}")
                if notification and buffering:
                    # Only prepend if we found content closing tags (user-facing response)
                    buffer = "\n\n" + notification + "\n" + buffer
                    logger.info(f"After insertion: {buffer[:100] if buffer else 'empty'}")
            
            # Send the buffered content with notification inserted
            if buffer:
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "model_used": "local",
                    "_notification_sent": True,
                    "choices": [{"index": 0, "delta": {"content": buffer}}],
                }
            
            # Send finish chunk
            yield {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "model_used": "local",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            
            # Record statistics
            stats = self.get_stats()
            stats.record_request("local", {
                "choices": [{"message": {"content": full_content}}],
                "usage": {"completion_tokens": completion_tokens, "total_tokens": completion_tokens},
                "model_used": "local",
            })
            logger.info(f"Local streaming complete: {completion_tokens} tokens")
            
        except Exception as e:
            logger.error(f"Local model streaming failed: {e}")
            stats = self.get_stats()
            stats.record_error("local")
            raise

    async def _adaptive_cerebras(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Adaptive Cerebras strategy:
        - Always use Cerebras by default
        - Check API rate limit headers
        - Switch to local when approaching limit (e.g., <20% tokens remaining)
        - Use local for cooldown period (default: 60 seconds)
        - Return to Cerebras after cooldown
        
        Respects headers:
        - x-ratelimit-limit-tokens-minute
        - x-ratelimit-remaining-tokens-minute
        - x-ratelimit-reset-tokens-minute
        """
        config = self.config.adaptive_cerebras
        cooldown_seconds = config.cooldown_seconds
        
        # Get current rate limits from Cerebras client
        rate_limits = self.cerebras_client.get_rate_limits()
        
        # Check if we should use local (cooldown period)
        now = time.time()
        last_switch_time = getattr(self, '_adaptive_last_switch_time', 0)
        
        if now - last_switch_time < cooldown_seconds:
            logger.info(f"=== ADAPTIVE CEREBRAS (COOLDOWN) ===")
            logger.info(f"Using local for cooldown ({cooldown_seconds}s remaining)")
            logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
            
            # Use local for cooldown
            response = await self._use_local(
                messages, temperature, max_tokens, **kwargs
            )
            
            # Update switch time
            self._adaptive_last_switch_time = now
            
            return response
        
        # Check if we should switch to local due to rate limits
        should_use_local, reason = rate_limits.is_near_limit(threshold_percent=config.threshold_percent)
        
        if should_use_local:
            logger.info(f"=== ADAPTIVE CEREBRAS (SWITCH TO LOCAL) ===")
            logger.info(f"Reason: {reason}")
            logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
            
            # Use local for cooldown period
            response = await self._use_local(
                messages, temperature, max_tokens, **kwargs
            )
            
            # Update switch time
            self._adaptive_last_switch_time = now
            
            return response
        
        # Rate limits are safe, use Cerebras
        logger.info(f"=== ADAPTIVE CEREBRAS (USE CEREBRAS) ===")
        logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
        
        try:
            response = await self.cerebras_client.chat_completion(
                messages, temperature, max_tokens, **kwargs
            )
            response["model_used"] = "cerebras"
            response["_from_cache"] = False
            stats = self.get_stats()
            stats.record_request("cerebras", response)
            
            # Record the request in rate limiter
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            self.rate_limiter.record_request(tokens_used)
            
            return response
        except Exception as e:
            logger.error(f"Cerebras model failed: {e}")
            stats = self.get_stats()
            stats.record_error("cerebras")
            raise

    async def _use_cerebras_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        bypass_rate_limit: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Use Cerebras model with streaming. Buffers from content closing tag to insert notification."""
        # Check rate limits unless bypassing
        if not bypass_rate_limit:
            should_use, reason = self.rate_limiter.should_use_cerebras()
            if not should_use:
                logger.info(f"Rate limit check: {reason} - skipping Cerebras")
                raise RateLimitError(reason)
        
        try:
            import time
            
            # Track content and detect when to start buffering
            full_content = ""
            buffer = ""
            buffering = False  # Start buffering when we see content closing tags
            completion_tokens = 0
            response_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            model_name = self.config.models["cerebras"].model
            
            # Tags that indicate main content is ending (notification goes before these)
            content_end_patterns = ['</response>', '</question>', '</result>']
            
            async for chunk in self.cerebras_client.chat_completion_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content") or ""
                    full_content += content
                    
                    if "usage" in chunk:
                        completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                    else:
                        completion_tokens += len(content) // 4 + 1
                    
                    if "id" in chunk:
                        response_id = chunk["id"]
                    if "created" in chunk:
                        created = chunk["created"]
                    
                    if buffering:
                        # Already buffering, accumulate
                        buffer += content
                    else:
                        # Check if this chunk contains a content closing tag
                        combined = buffer + content
                        tag_found = None
                        tag_pos = -1
                        for tag in content_end_patterns:
                            pos = combined.lower().find(tag.lower())
                            if pos != -1 and (tag_pos == -1 or pos < tag_pos):
                                tag_pos = pos
                                tag_found = tag
                        
                        if tag_found:
                            # Found closing tag - send everything before it, buffer the rest
                            to_send = combined[:tag_pos]
                            buffer = combined[tag_pos:]
                            buffering = True
                            if to_send:
                                yield {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "model_used": "cerebras",
                                    "choices": [{"index": 0, "delta": {"content": to_send}}],
                                }
                        else:
                            # No closing tag yet, stream normally but keep small lookahead
                            buffer += content
                            if len(buffer) > 50:
                                to_send = buffer[:-50]
                                buffer = buffer[-50:]
                                if to_send:
                                    yield {
                                        "id": response_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "model_used": "cerebras",
                                        "choices": [{"index": 0, "delta": {"content": to_send}}],
                                    }
            
            # Stream complete - insert notification only for user-facing responses
            logger.info(f"Cerebras stream complete. Buffering={buffering}, Buffer length: {len(buffer)}")
            config = self.config
            cost = (completion_tokens / 1000) * config.cost_tracking.cerebras_cost_per_1k_tokens
            
            # Only add notification to user-facing responses, NOT tool calls
            full_lower = full_content.lower()
            user_facing_tags = ['<plan_mode_respond>', '<ask_followup_question>', '<attempt_completion>']
            tool_use_tags = ['<write_to_file>', '<replace_in_file>', '<read_file>', '<execute_command>', 
                            '<search_files>', '<list_files>', '<list_code_definition_names>',
                            '<browser_action>', '<use_mcp_tool>', '<access_mcp_resource>']
            
            is_user_facing = any(tag in full_lower for tag in user_facing_tags)
            is_tool_use = any(tag in full_lower for tag in tool_use_tags)
            should_add_notification = is_user_facing and not is_tool_use
            
            logger.info(f"Notification check: user_facing={is_user_facing}, tool_use={is_tool_use}, add={should_add_notification}")
            
            if config.response_notifications.enabled and should_add_notification:
                from main import format_response_notification
                
                notification = format_response_notification(
                    model_used="cerebras",
                    total_tokens=completion_tokens,
                    cost=cost,
                    config_notifications=config.response_notifications,
                )
                logger.info(f"Notification generated: {notification[:50] if notification else 'None'}")
                if notification and buffering:
                    # Only prepend if we found content closing tags (user-facing response)
                    buffer = "\n\n" + notification + "\n" + buffer
                    logger.info(f"After insertion: {buffer[:100] if buffer else 'empty'}")
            
            # Send the buffered content with notification inserted
            if buffer:
                yield {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "model_used": "cerebras",
                    "_notification_sent": True,
                    "choices": [{"index": 0, "delta": {"content": buffer}}],
                }
            
            # Send finish chunk
            yield {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "model_used": "cerebras",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            
            # Record statistics
            stats = self.get_stats()
            stats.record_request("cerebras", {
                "choices": [{"message": {"content": full_content}}],
                "usage": {"completion_tokens": completion_tokens, "total_tokens": completion_tokens},
                "model_used": "cerebras",
            })
            self.rate_limiter.record_request(completion_tokens)
            logger.info(f"Cerebras streaming complete: {completion_tokens} tokens, cost: ${cost:.4f}")
            
        except RateLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            stats = self.get_stats()
            # Check if it's a rate limit error (429)
            if "429" in error_str or "Too Many Requests" in error_str:
                self.rate_limit_hits += 1
                self.last_rate_limit_time = None
                self.rate_limiter.record_rate_limit_hit()
                stats.record_rate_limit()
                stats.record_error("cerebras")
                logger.warning(f"Cerebras rate limit hit during streaming: {e}")
                raise
            else:
                stats.record_error("cerebras")
                logger.error(f"Cerebras model streaming failed: {e}")
                raise
    
    async def _adaptive_cerebras_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Adaptive Cerebras streaming:
        - Always stream from Cerebras by default
        - Monitor rate limit headers
        - Switch to local when approaching limit
        - Use local for cooldown period
        - Return to Cerebras after cooldown
        
        Yields:
            Streaming chunks from selected model
        """
        config = self.config.adaptive_cerebras
        cooldown_seconds = config.cooldown_seconds
        
        # Get current rate limits from Cerebras client
        rate_limits = self.cerebras_client.get_rate_limits()
        
        # Check if we should use local (cooldown period)
        now = time.time()
        last_switch_time = getattr(self, '_adaptive_last_switch_time', 0)
        
        if now - last_switch_time < cooldown_seconds:
            logger.info(f"=== ADAPTIVE CEREBRAS (COOLDOWN) ===")
            logger.info(f"Using local for cooldown ({cooldown_seconds}s remaining)")
            logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
            
            # Use local for cooldown
            async for chunk in self._use_local_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            
            # Update switch time
            self._adaptive_last_switch_time = now
            return
        
        # Check if we should switch to local due to rate limits
        should_use_local, reason = rate_limits.is_near_limit(threshold_percent=config.threshold_percent)
        
        if should_use_local:
            logger.info(f"=== ADAPTIVE CEREBRAS (SWITCH TO LOCAL) ===")
            logger.info(f"Reason: {reason}")
            logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
            
            # Use local for cooldown period
            async for chunk in self._use_local_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            
            # Update switch time
            self._adaptive_last_switch_time = now
            return
        
        # Rate limits are safe, use Cerebras
        logger.info(f"=== ADAPTIVE CEREBRAS (USE CEREBRAS) ===")
        logger.info(f"Cerebras tokens remaining: {rate_limits.remaining_tokens_minute}")
        
        try:
            async for chunk in self._use_cerebras_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Cerebras model streaming failed: {e}")
            # Fall back to local
            async for chunk in self._use_local_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk

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

        # Quick accept: if draft looks complete, skip verification
        if self._draft_looks_complete(draft_content):
            logger.info("Draft looks complete, accepting without verification")
            return False

        # Always verify if draft hit the max draft token limit (incomplete)
        if draft_tokens >= spec_config.max_draft_tokens:
            logger.info(
                f"Draft hit limit ({draft_tokens} >= {spec_config.max_draft_tokens}), will verify"
            )
            return True

        # Verify if draft is very short (increased threshold)
        if len(draft_content) < 100:
            logger.info("Draft very short, will verify")
            return True

        # Verify if draft contains clear incomplete markers
        incomplete_markers = ["...", "(to be continued)", "[more to come]", "(continued)"]
        if any(marker in draft_content.lower() for marker in incomplete_markers):
            logger.info("Draft contains incomplete markers, will verify")
            return True

        # Don't verify if draft is complete and within token limit
        logger.info(
            f"Draft complete ({draft_tokens} < {spec_config.max_draft_tokens}), skipping verification"
        )
        return False

    def _draft_looks_complete(self, draft_content: str) -> bool:
        """Check if draft appears to be a complete response."""
        if not draft_content:
            return False

        # Remove trailing whitespace
        content = draft_content.strip()

        # Check if ends with proper punctuation
        if content[-1] in ['.', '!', '?', '」', '】']:
            # Additional checks for completeness
            # Check if has structure (multiple sentences or bullet points)
            has_structure = (
                '\n' in content or  # Has newlines
                content.count('. ') >= 2 or  # Multiple sentences
                any(marker in content for marker in ['1.', '2.', '-', '•', '*'])  # Has list items
            )
            return has_structure

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

    def _insert_notification(self, content: str, notification: str) -> str:
        """
        Insert notification into content, handling XML structures used by clients like Cline.
        
        If content ends with XML closing tags, insert notification before them.
        Otherwise, append notification at the end.
        """
        import re
        
        # Common XML closing tag patterns used by Cline and similar clients
        # Order matters: more specific patterns first
        xml_patterns = [
            r'(</task_progress>\s*</plan_mode_respond>\s*)$',
            r'(</task_progress>\s*</ask_followup_question>\s*)$',
            r'(</response>\s*</plan_mode_respond>\s*)$',
            r'(</response>\s*</ask_followup_question>\s*)$',
            r'(</response>\s*</attempt_completion>\s*)$',
            r'(</result>\s*</attempt_completion>\s*)$',
            r'(</question>\s*</ask_followup_question>\s*)$',
            r'(</task_progress>\s*)$',
            r'(</plan_mode_respond>\s*)$',
            r'(</ask_followup_question>\s*)$',
            r'(</attempt_completion>\s*)$',
            r'(</response>\s*)$',
            r'(</question>\s*)$',
            r'(</result>\s*)$',
        ]
        
        for pattern in xml_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Insert notification before the XML closing tags
                insert_pos = match.start()
                return content[:insert_pos] + "\n\n" + notification + "\n" + content[insert_pos:]
        
        # No XML structure found, append at end
        return content + "\n\n" + notification

    async def _smart_speculative_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Smart routing with speculative decoding for streaming (Silent Mode).
        
        For complex tasks:
        - Stream draft from local model (OpenAI compatible)
        - Run Cerebras verification in background
        - Log metadata on server only (not sent to client)
        
        For simple tasks:
        - Stream directly from local model
        
        Yields:
            Streaming chunks (OpenAI compatible format)
        """
        is_complex, reason = self.classifier.classify(messages)

        if is_complex:
            logger.info(f"=== ROUTING DECISION (STREAM - SMART SPECULATIVE) ===")
            logger.info(f"Complex task, using speculative decoding: {reason}")
            async for chunk in self._speculative_decoding_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
        else:
            logger.info(f"=== ROUTING DECISION (STREAM - SMART SPECULATIVE) ===")
            logger.info(f"Simple task, using local directly: {reason}")
            async for chunk in self._use_local_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk

    async def _speculative_decoding_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Speculative decoding with streaming (Silent Mode).
        
        Strategy:
        1. Generate draft from local model internally (no streaming)
        2. Run Cerebras verification on the draft
        3. Stream only the final result:
           - If draft accepted: stream the draft content
           - If draft rejected: stream the Cerebras response
        4. Maintain full OpenAI API compatibility
        
        Yields:
            Streaming chunks from the final result (OpenAI compatible format)
        """
        if not self.config.speculative_decoding.enabled:
            logger.debug("Speculative decoding disabled, using local stream")
            async for chunk in self._use_local_stream(
                messages, temperature, max_tokens, **kwargs
            ):
                yield chunk
            return

        logger.info("=== SPECULATIVE DECODING STREAM START ===")

        spec_config = self.config.speculative_decoding
        max_draft_tokens = spec_config.max_draft_tokens
        min_confidence = spec_config.min_confidence
        import time
        response_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # Step 1: Generate draft internally (don't stream to client)
        logger.info("Generating draft internally (no streaming)...")
        draft_response = await self._use_local(
            messages,
            temperature,
            min(max_draft_tokens, max_tokens) if max_tokens else max_draft_tokens,
            **kwargs,
        )

        draft_content = self._extract_content(draft_response)
        draft_tokens = draft_response.get("usage", {}).get("completion_tokens", 0)

        logger.info(f"Draft generated internally: {draft_tokens} tokens, {len(draft_content)} chars")

        # Step 2: Check if verification is needed
        should_verify = self._should_verify_draft(
            draft_content, draft_tokens, max_tokens
        )

        # Step 3: If no verification needed, stream the draft
        if not should_verify:
            logger.info("Draft accepted without verification (short/simple response)")
            logger.info("=== SPECULATIVE DECODING STREAM COMPLETE ===")
            logger.info(f"Draft accepted: True, Spilled over: False, Tokens saved: {draft_tokens}")
            
            # Insert notification into content if enabled
            config = self.config
            content_to_stream = draft_content
            if config.response_notifications.enabled:
                from main import format_response_notification
                
                notification = format_response_notification(
                    model_used="local",
                    total_tokens=draft_tokens,
                    cost=0.0,  # Local model is free
                    config_notifications=config.response_notifications,
                )
                if notification:
                    content_to_stream = self._insert_notification(draft_content, notification)
                    logger.info("Inserted local notification into draft (no verification)")
            
            # Stream the draft content as if it's a fresh response
            chunk_size = 50  # Small chunks for smooth streaming
            for i in range(0, len(content_to_stream), chunk_size):
                content_chunk = content_to_stream[i:i + chunk_size]
                draft_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.models["local"].model,
                    "_notification_sent": True,  # Flag for main.py to prevent duplicate
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content_chunk,
                            },
                        }
                    ],
                }
                yield draft_chunk
                # Small delay to simulate natural streaming
                import asyncio
                await asyncio.sleep(0.01)
            
            # Send finish_reason chunk
            finish_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.models["local"].model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield finish_chunk
            
            # Record statistics
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, 0, True)
            return

        # Step 4: Run verification and check for spill over
        logger.info("Running verification to check for spill over...")
        spilled_over = False
        verify_tokens = 0
        verify_content = ""
        
        try:
            verify_response = await self._use_cerebras(
                messages, temperature, max_tokens, **kwargs
            )

            verify_content = self._extract_content(verify_response)
            verify_tokens = verify_response.get("usage", {}).get(
                "completion_tokens", 0
            )

            similarity = self._calculate_similarity(draft_content, verify_content)
            
            # Determine outcome
            accepted = similarity >= min_confidence
            spilled_over = not accepted

            # Log results
            logger.info(f"=== SPECULATIVE DECODING VERIFICATION RESULTS ===")
            logger.info(f"Draft tokens: {draft_tokens}")
            logger.info(f"Verified tokens: {verify_tokens}")
            logger.info(f"Similarity: {similarity:.2f}")
            logger.info(f"Draft accepted: {accepted}")
            logger.info(f"Spilled over: {spilled_over}")
            logger.info(f"Tokens saved: {draft_tokens if accepted else 0}")
            
            # Record statistics
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, verify_tokens, accepted)
            
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            logger.info(f"Draft acceptance assumed (verification failed)")
            stats = self.get_stats()
            stats.record_speculative_decoding(draft_tokens, 0, True)
            accepted = True
            spilled_over = False

        # Step 5: Stream the final result (either draft or verification)
        if spilled_over:
            # Stream the Cerebras response
            logger.info(f"Draft rejected, streaming Cerebras response to client")
            
            # Append notification directly to content if enabled (so it's part of the response)
            config = self.config
            content_to_stream = verify_content
            if config.speculative_decoding.notification_on_spillover and config.response_notifications.enabled:
                from main import format_response_notification
                
                cost = (verify_tokens / 1000) * config.cost_tracking.cerebras_cost_per_1k_tokens
                notification = format_response_notification(
                    model_used="cerebras",
                    total_tokens=verify_tokens,
                    cost=cost,
                    config_notifications=config.response_notifications,
                )
                if notification:
                    # Insert notification before XML closing tags if present (for Cline compatibility)
                    content_to_stream = self._insert_notification(verify_content, notification)
                    logger.info("Inserted Cerebras notification into response content")
            
            logger.info(f"Streaming verified response: {len(content_to_stream)} chars")
            
            chunk_size = 50  # Small chunks for smooth streaming
            for i in range(0, len(content_to_stream), chunk_size):
                content_chunk = content_to_stream[i:i + chunk_size]
                verified_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.models["cerebras"].model,
                    "_notification_sent": True,  # Flag for main.py to prevent duplicate
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content_chunk,
                            },
                        }
                    ],
                }
                yield verified_chunk
                # Small delay to simulate natural streaming
                import asyncio
                await asyncio.sleep(0.01)
            
            logger.info(f"Verified response streaming complete")
            
            # Send finish_reason chunk for spillover path
            finish_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.models["cerebras"].model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield finish_chunk
        else:
            # Draft was accepted, stream the draft content
            logger.info(f"Draft accepted, streaming draft content to client")
            
            # Insert notification into content if enabled
            config = self.config
            content_to_stream = draft_content
            if config.response_notifications.enabled:
                from main import format_response_notification
                
                notification = format_response_notification(
                    model_used="local",
                    total_tokens=draft_tokens,
                    cost=0.0,  # Local model is free
                    config_notifications=config.response_notifications,
                )
                if notification:
                    content_to_stream = self._insert_notification(draft_content, notification)
                    logger.info("Inserted local notification into draft content")
            
            chunk_size = 50  # Small chunks for smooth streaming
            for i in range(0, len(content_to_stream), chunk_size):
                content_chunk = content_to_stream[i:i + chunk_size]
                draft_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.models["local"].model,
                    "_notification_sent": True,  # Flag for main.py to prevent duplicate
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": content_chunk,
                            },
                        }
                    ],
                }
                yield draft_chunk
                # Small delay to simulate natural streaming
                import asyncio
                await asyncio.sleep(0.01)
            
            logger.info(f"Draft content streaming complete. Tokens saved: {draft_tokens}")
            
            # Send finish_reason chunk for accepted draft path
            finish_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.models["local"].model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield finish_chunk
        
        logger.info("=== SPECULATIVE DECODING STREAM COMPLETE ===")
