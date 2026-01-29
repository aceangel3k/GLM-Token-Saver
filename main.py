from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging
from logging.handlers import RotatingFileHandler
import os
import json
import sys
import importlib.util
from router import SmartRouter
from config import get_config


def format_response_notification(
    model_used: str,
    total_tokens: int,
    cost: float,
    config_notifications,
) -> str:
    """
    Format a response notification based on configuration.

    Args:
        model_used: The model that generated the response (local/cerebras)
        total_tokens: Total tokens used in the response
        cost: Cost of the response in USD
        config_notifications: ResponseNotificationsConfig object

    Returns:
        Formatted notification string
    """
    if not config_notifications.enabled:
        return ""

    # Determine model display name and emoji
    model_display = model_used.capitalize()
    
    # Map models to emojis
    model_emojis = {
        "local": "🏠",
        "cerebras": "🌎",
    }
    emoji = model_emojis.get(model_used.lower(), "")

    # Format cost with appropriate precision
    if cost == 0:
        cost_str = "0.00"
    else:
        cost_str = f"{cost:.4f}" if cost < 0.01 else f"{cost:.2f}"

    # Format the notification using the template
    notification = config_notifications.template.format(
        emoji=emoji,
        model=model_display,
        tokens=total_tokens,
        cost=cost_str,
    )

    return notification


# Import custom statistics module (avoid name conflict with Python's built-in statistics)
spec = importlib.util.spec_from_file_location(
    "api_statistics",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics.py"),
)
if spec is not None and spec.loader is not None:
    api_statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_statistics)
    get_stats = api_statistics.get_stats
    StatisticsTracker = api_statistics.StatisticsTracker
else:
    raise ImportError("Failed to load statistics module")


# Setup logging
def setup_logging():
    """Setup logging configuration."""
    config = get_config()

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.logging.file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get root logger
    root_logger = logging.getLogger()

    # Remove all existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Set root logger level
    root_logger.setLevel(logging.DEBUG)  # Always capture all levels

    # Add rotating file handler (captures all levels)
    file_handler = RotatingFileHandler(
        config.logging.file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Add console handler (only warnings and errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.server.log_level.upper()))
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="GLM Token Saver API",
    description="Speculative decoding API for GLM models with smart routing",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize router
router = SmartRouter(get_stats)


# Pydantic models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: Any  # Accept any type: string, list, dict, etc.

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None, description="Model name (optional, will be auto-routed)"
    )
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = Field(default=False)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

    model_config = ConfigDict(extra="allow")


class TokenDetails(BaseModel):
    """Token details with optional breakdown."""

    cached_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    """Usage statistics for the request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Union[TokenDetails, Dict[str, Any]]] = None
    completion_tokens_details: Optional[Union[TokenDetails, Dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Union[Usage, Dict[str, Any]]
    routing_model: Optional[str] = None


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GLM Token Saver API",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "models": "/v1/models",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "local": get_config().models["local"].enabled,
            "cerebras": get_config().models["cerebras"].enabled,
        },
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List available models."""
    config = get_config()
    return {
        "object": "list",
        "data": [
            {
                "id": config.models["local"].model,
                "object": "model",
                "owned_by": "local",
                "permission": [],
                "root": config.models["local"].model,
                "parent": None,
            },
            {
                "id": config.models["cerebras"].model,
                "object": "model",
                "owned_by": "cerebras",
                "permission": [],
                "root": config.models["cerebras"].model,
                "parent": None,
            },
        ],
    }


async def stream_response(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
) -> AsyncGenerator[str, None]:
    """
    Generate streaming response in OpenAI-compatible SSE format using real streaming.
    """
    import time
    
    try:
        config = get_config()
        response_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        
        # Determine routing strategy and model BEFORE streaming starts
        strategy = router.config.routing.strategy
        # Simple classification to determine model (same logic as router.route_stream)
        is_complex, _ = router.classifier.classify(messages)
        
        if strategy == "always_local":
            model_used = "local"
        elif strategy == "always_cerebras":
            model_used = "cerebras"
        elif strategy == "smart_routing":
            model_used = "cerebras" if is_complex else "local"
        elif strategy in ["smart_speculative", "speculative_decoding"]:
            model_used = "local"  # Draft comes from local
        else:
            model_used = "local"
        
        # Track streaming state
        total_tokens = 0
        chunk_count = 0
        
        # Track if router sent a notification (initialized once, never overwritten)
        notification_already_sent = False
        
        # Route streaming request
        logger.info(f"=== STREAMING STARTED ===")
        logger.info(f"Model determined: {model_used}")
        
        async for chunk in router.route_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ):
            # Check if router already sent a notification (set to True if any chunk had the flag)
            if chunk.pop("_notification_sent", False):
                notification_already_sent = True
                logger.info(f"=== NOTIFICATION CHUNK DETECTED ===")
                logger.info(f"Notification content: {chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')[:100]}")
            
            # Track token usage
            if "usage" in chunk:
                usage = chunk["usage"]
                total_tokens = usage.get("total_tokens", total_tokens)
            
            # Add routing model info if present
            if "model_used" in chunk:
                chunk["routing_model"] = chunk["model_used"]
                del chunk["model_used"]
            
            # Yield the actual chunk
            chunk["id"] = response_id
            chunk["created"] = created
            yield f"data: {json.dumps(chunk)}\n\n"
            chunk_count += 1
        
        # Prepare and send notification if enabled (only after streaming completes)
        # Skip if router already sent a notification
        if config.response_notifications.enabled and not notification_already_sent:
            # Calculate cost based on model used
            cost = 0.0
            if model_used == "cerebras":
                cost = (total_tokens / 1000) * config.cost_tracking.cerebras_cost_per_1k_tokens
            
            # Format notification with correct token count
            notification = format_response_notification(
                model_used=model_used,
                total_tokens=total_tokens,
                cost=cost,
                config_notifications=config.response_notifications,
            )
            
            if notification:
                position = config.response_notifications.position
                
                # For streaming, only support append position since we can't prepend with unknown token count
                if position == "append":
                    # Send the notification as a delta chunk at the end
                    final_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "glm-4.7",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": notification,
                                },
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    logger.info(f"Notification sent as final chunk (append): {total_tokens} tokens")
                elif position == "prepend":
                    logger.info(f"Notification position 'prepend' not supported for streaming (unknown token count). Use 'append' instead.")
                elif position == "both":
                    # Send notification at end (append position for streaming)
                    final_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "glm-4.7",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": notification,
                                },
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    logger.info(f"Notification sent as final chunk (both): {total_tokens} tokens")
        
        # Send final usage chunk if we have token count
        if total_tokens > 0:
            usage_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "glm-4.7",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,  # Not tracked during streaming
                    "completion_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"
        
        logger.info(f"=== STREAMING COMPLETE ===")
        logger.info(f"Total tokens: {total_tokens}")

    except Exception as e:
        logger.error(f"Error in stream_response: {e}", exc_info=True)
        # Send error chunk
        error_chunk = {
            "id": response_id if 'response_id' in locals() else "chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": created if 'created' in locals() else int(time.time()),
            "model": "glm-4.7",
            "choices": [],
            "error": {"message": str(e), "type": "stream_error", "code": "stream_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    Chat completions endpoint (OpenAI-compatible).

    This endpoint automatically routes requests to the appropriate model
    based on task complexity and configuration.
    """
    try:
        # Parse request
        try:
            request_data = await request.json()
            logger.info(f"=== INCOMING REQUEST ===")
            logger.info(f"Model: {request_data.get('model', 'not specified')}")
            logger.info(f"Message count: {len(request_data.get('messages', []))}")
            logger.info(f"Temperature: {request_data.get('temperature', 'default')}")
            logger.info(f"Max tokens: {request_data.get('max_tokens', 'default')}")
            logger.info(f"Stream: {request_data.get('stream', 'default')}")
            logger.info(f"Full request: {json.dumps(request_data, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Validate and create ChatCompletionRequest
        try:
            chat_request = ChatCompletionRequest(**request_data)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "message": f"Validation error: {str(e)}",
                        "type": "validation_error",
                        "code": "validation_error",
                    }
                },
            )

        # Convert messages to dict format and handle multimodal content
        messages = []
        for msg in chat_request.messages:
            msg_dict = msg.model_dump()

            # Normalize content to string
            content = msg_dict.get("content", "")

            # Handle content as list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    # Skip None/undefined blocks
                    if block is None:
                        continue

                    # Handle dict blocks
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "image_url":
                            text_parts.append("[image]")
                        elif block_type is None:
                            # No type field, try to extract content directly
                            if "text" in block:
                                text_parts.append(block["text"])
                            elif "content" in block:
                                text_parts.append(str(block["content"]))
                            else:
                                text_parts.append(str(block))
                    # Handle string blocks
                    elif isinstance(block, str):
                        text_parts.append(block)
                    # Handle objects with type attribute
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(getattr(block, "text", ""))
                        elif block.type == "image_url":
                            text_parts.append("[image]")
                    # Handle any other type
                    else:
                        text_parts.append(str(block))

                msg_dict["content"] = "".join(text_parts) if text_parts else ""
            # Handle content as other types
            elif not isinstance(content, str):
                msg_dict["content"] = str(content)

            messages.append(msg_dict)

        logger.info(f"=== PROCESSED MESSAGES ===")
        logger.info(f"Processed {len(messages)} messages")
        for i, msg in enumerate(messages):
            logger.debug(
                f"Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}"
            )

        # Check if streaming is requested
        stream = chat_request.stream

        # If streaming is requested, return streaming response
        if stream:
            logger.info("=== STREAMING REQUESTED ===")
            return StreamingResponse(
                stream_response(
                    messages=messages,
                    temperature=chat_request.temperature or 0.9,
                    max_tokens=chat_request.max_tokens,
                    top_p=chat_request.top_p,
                    frequency_penalty=chat_request.frequency_penalty,
                    presence_penalty=chat_request.presence_penalty,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Route request through smart router with override detection (non-streaming)
        response = await router.route_with_override(
            messages=messages,
            temperature=chat_request.temperature or 0.9,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p,
            frequency_penalty=chat_request.frequency_penalty,
            presence_penalty=chat_request.presence_penalty,
        )

        # Extract response data
        choices = response.get("choices", [])
        usage = response.get("usage", {})
        model_used = response.get("model_used", "unknown")
        speculative_decoding = response.get("speculative_decoding", None)

        # Validate response structure
        if not choices:
            raise ValueError("No choices in response from model")

        if not isinstance(choices, list):
            raise ValueError(f"Choices is not a list: {type(choices)}")

        # Clean up usage to match OpenAI format
        cleaned_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        # Only include tokens_details if they exist and are not None/empty
        pt_details = usage.get("prompt_tokens_details")
        if pt_details and pt_details is not None and pt_details != {}:
            cleaned_usage["prompt_tokens_details"] = pt_details
        ct_details = usage.get("completion_tokens_details")
        if ct_details and ct_details is not None and ct_details != {}:
            cleaned_usage["completion_tokens_details"] = ct_details

        # Clean up choices to ensure OpenAI compatibility
        cleaned_choices = []
        for choice in choices:
            if not isinstance(choice, dict):
                logger.warning(f"Skipping invalid choice (not a dict): {choice}")
                continue

            cleaned_choice = choice.copy()
            message = cleaned_choice.get("message", {})

            if not isinstance(message, dict):
                logger.warning(f"Skipping invalid message (not a dict): {message}")
                message = {}
                cleaned_choice["message"] = message

            # Remove non-OpenAI fields like 'reasoning_content' or 'reasoning'
            for reasoning_field in ["reasoning_content", "reasoning"]:
                if reasoning_field in message:
                    # If content is empty but reasoning field exists, use reasoning as content
                    content = message.get("content", "")
                    if not content or content == "":
                        message["content"] = message[reasoning_field]
                    # Remove the reasoning field
                    del message[reasoning_field]

            # Ensure content is a string (not a list or other type)
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if block is None:
                        continue

                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "image_url":
                            text_parts.append("[image]")
                        elif block_type is None:
                            if "text" in block:
                                text_parts.append(block["text"])
                            elif "content" in block:
                                text_parts.append(str(block["content"]))
                            else:
                                text_parts.append(str(block))
                    elif isinstance(block, str):
                        text_parts.append(block)
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(getattr(block, "text", ""))
                        elif block.type == "image_url":
                            text_parts.append("[image]")
                    else:
                        text_parts.append(str(block))

                message["content"] = "".join(text_parts) if text_parts else ""
            elif not isinstance(content, str):
                message["content"] = str(content)

            # Ensure content is not None or undefined
            if message.get("content") is None:
                message["content"] = ""

            # Ensure role is present
            if not message.get("role"):
                message["role"] = "assistant"

            cleaned_choices.append(cleaned_choice)

        # Final validation: ensure all required fields are valid
        validated_choices = []
        for choice in cleaned_choices:
            validated_choice = choice.copy()
            message = validated_choice.get("message", {})

            # Ensure content is a valid string
            if not isinstance(message.get("content"), str):
                message["content"] = ""

            # Ensure role is a valid string
            if not isinstance(message.get("role"), str):
                message["role"] = "assistant"

            validated_choice["message"] = message
            validated_choices.append(validated_choice)

        logger.info(f"=== RESPONSE PROCESSING ===")
        logger.info(f"Model used: {model_used}")
        logger.info(f"Choices before cleaning: {len(choices)}")
        logger.info(f"Choices after validation: {len(validated_choices)}")
        for i, choice in enumerate(validated_choices):
            content = choice.get("message", {}).get("content", "")
            logger.info(
                f"Choice {i}: content_length={len(content)}, finish_reason={choice.get('finish_reason', 'unknown')}"
            )
        logger.info(f"Usage: {usage.get('total_tokens', 0)} total tokens")
        logger.debug(f"Original choices: {choices}")
        logger.debug(f"Validated choices: {validated_choices}")

        # Apply response notifications if enabled
        config = get_config()
        if config.response_notifications.enabled:
            total_tokens = cleaned_usage.get("total_tokens", 0)
            
            # Check if this was a spillover from speculative decoding
            # If so, we want to show the cerebras model info with the verified tokens
            display_model = model_used
            display_tokens = total_tokens
            display_cost = 0.0
            
            if speculative_decoding and speculative_decoding.get("spilled_over", False):
                # Use cerebras model info for spillover notifications
                display_model = "cerebras"
                display_tokens = speculative_decoding.get("verified_tokens", total_tokens)
                display_cost = (display_tokens / 1000) * config.cost_tracking.cerebras_cost_per_1k_tokens
                logger.info("Using cerebras model info for spillover notification")
            elif model_used == "cerebras":
                display_cost = (total_tokens / 1000) * config.cost_tracking.cerebras_cost_per_1k_tokens
            elif model_used == "local":
                display_cost = 0.0
            
            # Format the notification
            notification = format_response_notification(
                model_used=display_model,
                total_tokens=display_tokens,
                cost=display_cost,
                config_notifications=config.response_notifications,
            )
            
            # Apply notification based on position setting
            if notification:
                position = config.response_notifications.position
                for choice in validated_choices:
                    content = choice.get("message", {}).get("content", "")
                    
                    if position == "prepend":
                        choice["message"]["content"] = notification + "\n" + content
                    elif position == "append":
                        choice["message"]["content"] = content + notification
                    elif position == "both":
                        choice["message"]["content"] = notification + "\n" + content + "\n" + notification
                
                logger.info(f"Applied notification to response (position: {position}, model: {display_model}, tokens: {display_tokens})")

        # Return OpenAI-compatible response
        response_data = {
            "id": response.get("id", "chatcmpl-unknown"),
            "object": "chat.completion",
            "created": response.get("created", 0),
            "model": response.get("model", "glm-4.7"),
            "choices": validated_choices,
            "usage": cleaned_usage,
            "model_used": model_used,
        }

        # Add speculative_decoding stats if available
        if speculative_decoding:
            response_data["speculative_decoding"] = speculative_decoding

        # Remove non-OpenAI fields, but keep model_used and speculative_decoding for debugging
        for field in ["routing_model", "reasoning_content"]:
            if field in response_data:
                del response_data[field]

        logger.info(f"=== FINAL RESPONSE ===")
        logger.info(f"Response: {json.dumps(response_data, indent=2)}")
        logger.info(f"Response keys: {list(response_data.keys())}")

        return response_data

    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error",
                }
            },
        )


@app.get("/stats")
async def get_statistics():
    """Get usage statistics."""
    stats = get_stats()
    return stats.get_statistics()


@app.post("/stats/reset")
async def reset_statistics():
    """Reset all statistics."""
    stats = get_stats()
    stats.reset()
    return {"message": "Statistics reset successfully"}


@app.get("/admin/config")
async def get_current_config():
    """Get current configuration (admin endpoint)."""
    config = get_config()
    return {
        "routing": {
            "strategy": config.routing.strategy,
            "simple_task_threshold": config.routing.simple_task_threshold,
            "complexity_keywords": config.routing.complexity_keywords,
        },
        "speculative_decoding": {
            "enabled": config.speculative_decoding.enabled,
            "draft_model": config.speculative_decoding.draft_model,
            "verify_model": config.speculative_decoding.verify_model,
            "max_draft_tokens": config.speculative_decoding.max_draft_tokens,
            "min_confidence": config.speculative_decoding.min_confidence,
            "parallel_enabled": config.speculative_decoding.parallel_enabled,
            "max_concurrent_drafts": config.speculative_decoding.max_concurrent_drafts,
            "draft_timeout": config.speculative_decoding.draft_timeout,
        },
        "models": {
            "local": {
                "enabled": config.models["local"].enabled,
                "name": config.models["local"].name,
                "model": config.models["local"].model,
                "endpoint": config.models["local"].endpoint,
            },
            "cerebras": {
                "enabled": config.models["cerebras"].enabled,
                "name": config.models["cerebras"].name,
                "model": config.models["cerebras"].model,
            },
        },
    }


@app.post("/admin/config/reload")
async def reload_configuration():
    """Reload configuration from config.yaml file (admin endpoint)."""
    try:
        # Reload the router configuration (which also reloads the global config)
        router.reload_config()
        logger.info("Configuration reloaded successfully")
        return {
            "message": "Configuration reloaded successfully",
            "details": {
                "routing_strategy": router.config.routing.strategy,
                "local_enabled": router.config.models["local"].enabled,
                "cerebras_enabled": router.config.models["cerebras"].enabled,
                "speculative_decoding_enabled": router.config.speculative_decoding.enabled,
            }
        }
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to reload config: {str(e)}"}}
        )


# Cache admin endpoints

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get cache statistics (admin endpoint)."""
    return router.cache.get_statistics()


@app.get("/admin/cache/entries")
async def get_cache_entries(limit: int = 100):
    """Get cache entries (admin endpoint).

    Args:
        limit: Maximum number of entries to return (default: 100)
    """
    return {
        "entries": router.cache.get_entries(limit=limit),
        "total": len(router.cache._cache),
    }


@app.post("/admin/cache/clear")
async def clear_cache():
    """Clear all cache entries (admin endpoint)."""
    router.cache.clear()
    return {"message": "Cache cleared successfully"}


@app.post("/admin/cache/invalidate/{model}")
async def invalidate_cache_model(model: str):
    """Invalidate all cache entries for a specific model (admin endpoint).

    Args:
        model: Model name to invalidate (e.g., "local", "cerebras")
    """
    router.cache.invalidate_model(model)
    return {"message": f"Cache entries for model '{model}' invalidated successfully"}


@app.post("/admin/cache/cleanup")
async def cleanup_expired_cache():
    """Remove all expired entries from the cache (admin endpoint)."""
    removed = router.cache.cleanup_expired()
    return {
        "message": f"Cleaned up {removed} expired cache entries",
        "removed_count": removed
    }


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    logger.warning(f"=== GLM TOKEN SAVER API STARTING ===")
    logger.warning(f"Server: {config.server.host}:{config.server.port}")
    logger.warning(f"Routing strategy: {config.routing.strategy}")
    logger.warning(
        f"Local model: {config.models['local'].name} ({config.models['local'].model})"
    )
    logger.warning(f"Local endpoint: {config.models['local'].endpoint}")
    logger.warning(
        f"Cerebras model: {config.models['cerebras'].name} ({config.models['cerebras'].model})"
    )
    logger.warning(f"Cerebras endpoint: {config.models['cerebras'].endpoint}")
    logger.warning(f"Speculative decoding: {config.speculative_decoding.enabled}")
    logger.warning(f"========================================")

    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )
