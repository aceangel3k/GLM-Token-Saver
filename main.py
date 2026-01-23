from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler
import os
from router import SmartRouter
from config import get_config

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.logging.file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add rotating file handler
    file_handler = RotatingFileHandler(
        config.logging.file,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    logging.getLogger().addHandler(file_handler)
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="GLM Token Saver API",
    description="Speculative decoding API for GLM models with smart routing",
    version="1.0.0"
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
router = SmartRouter()


# Pydantic models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model name (optional, will be auto-routed)")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    model_used: Optional[str] = None


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
            "models": "/v1/models"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "local": get_config().models["local"].enabled,
            "cerebras": get_config().models["cerebras"].enabled
        }
    }


@app.get("/v1/models")
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
                "parent": None
            },
            {
                "id": config.models["cerebras"].model,
                "object": "model",
                "owned_by": "cerebras",
                "permission": [],
                "root": config.models["cerebras"].model,
                "parent": None
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI-compatible).
    
    This endpoint automatically routes requests to the appropriate model
    based on task complexity and configuration.
    """
    try:
        # Convert messages to dict format
        messages = [msg.model_dump() for msg in request.messages]
        
        logger.info(f"Received chat completion request with {len(messages)} messages")
        
        # Route request through smart router
        response = await router.route(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # Extract response data
        choices = response.get("choices", [])
        usage = response.get("usage", {})
        model_used = response.get("model_used", "unknown")
        
        # Log which model was used
        logger.info(f"Request completed using model: {model_used}")
        
        # Return OpenAI-compatible response
        return ChatCompletionResponse(
            id=response.get("id", "chatcmpl-unknown"),
            created=response.get("created", 0),
            model=response.get("model", "glm-4.7"),
            choices=choices,
            usage=usage,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )


@app.get("/stats")
async def get_stats():
    """Get usage statistics."""
    return {
        "message": "Statistics endpoint - to be implemented with token tracking",
        "features": {
            "smart_routing": True,
            "speculative_decoding": get_config().speculative_decoding.enabled,
            "cost_tracking": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    
    logger.info(f"Starting GLM Token Saver API on {config.server.host}:{config.server.port}")
    logger.info(f"Routing strategy: {config.routing.strategy}")
    logger.info(f"Local model: {config.models['local'].name}")
    logger.info(f"Cerebras model: {config.models['cerebras'].name}")
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level
    )
