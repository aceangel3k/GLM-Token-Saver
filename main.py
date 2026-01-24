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


async def stream_response(response: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Generate streaming response in OpenAI-compatible SSE format.

    This converts a complete response into a stream of SSE chunks.
    """
    try:
        choices = response.get("choices", [])
        usage = response.get("usage", {})
        model = response.get("model", "glm-4.7")
        created = response.get("created", 0)
        response_id = response.get("id", "chatcmpl-unknown")

        # Send each choice as a chunk
        for i, choice in enumerate(choices):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": i,
                        "delta": {
                            "role": "assistant",
                            "content": choice.get("message", {}).get("content", ""),
                        },
                        "finish_reason": choice.get("finish_reason", "stop"),
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final usage chunk if usage is available
        if usage:
            usage_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in stream_response: {e}", exc_info=True)
        # Send error chunk
        error_chunk = {
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

        # Route request through smart router
        response = await router.route(
            messages=messages,
            temperature=chat_request.temperature or 0.9,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p,
            frequency_penalty=chat_request.frequency_penalty,
            presence_penalty=chat_request.presence_penalty,
        )

        # If streaming is requested, return streaming response
        if stream:
            logger.info("=== STREAMING RESPONSE ===")
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
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
