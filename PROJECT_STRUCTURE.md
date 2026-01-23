# Project Structure

```
GLM_Token_Saver/
├── main.py                 # FastAPI application and endpoints
├── config.py               # Configuration management
├── config.yaml             # Configuration file (endpoints, API keys, settings)
├── router.py               # Smart routing and task complexity classifier
├── requirements.txt        # Python dependencies
├── test_api.py            # Test suite for the API
├── README.md              # Full documentation
├── START.md               # Quick start guide
├── .gitignore             # Git ignore rules
│
├── models/                # Model clients
│   ├── __init__.py
│   ├── base.py           # Base model client class
│   ├── local.py          # Local llama.cpp client
│   └── cerebras.py       # Cerebras API client
│
└── logs/                  # Log files (created at runtime)
    └── api_server.log
```

## File Descriptions

### Core Files

- **main.py**: FastAPI application with OpenAI-compatible endpoints
  - `/v1/chat/completions` - Main chat endpoint
  - `/health` - Health check
  - `/v1/models` - List available models
  - `/stats` - Usage statistics

- **config.py**: Configuration management using Pydantic
  - Loads settings from config.yaml
  - Provides type-safe configuration access

- **config.yaml**: Configuration file
  - Model endpoints and API keys
  - Routing strategy settings
  - Complexity keywords
  - Cost tracking limits

- **router.py**: Smart routing logic
  - TaskComplexityClassifier: Analyzes prompt complexity
  - SmartRouter: Routes requests to appropriate model
  - Handles fallbacks and errors

### Model Clients

- **models/base.py**: Abstract base class for model clients
  - Common HTTP request handling
  - Response parsing
  - Error handling

- **models/local.py**: Client for local llama.cpp model
  - Connects to your DGX Spark
  - Handles local model requests

- **models/cerebras.py**: Client for Cerebras API
  - Connects to Cerebras GLM 4.7
  - Handles cloud model requests

### Testing & Documentation

- **test_api.py**: Comprehensive test suite
  - Tests all endpoints
  - Verifies routing logic
  - Checks model selection

- **README.md**: Complete documentation
  - Features and architecture
  - Installation and configuration
  - API usage examples
  - Troubleshooting guide

- **START.md**: Quick start guide
  - Step-by-step setup
  - Basic testing
  - Cline integration

## Key Components

### 1. Smart Routing
- Analyzes task complexity
- Routes to appropriate model
- Automatic fallback on errors

### 2. Task Complexity Classifier
- Keyword detection
- Token count analysis
- Configurable thresholds

### 3. Model Clients
- Unified interface for both models
- Async HTTP requests
- Error handling and retries

### 4. OpenAI-Compatible API
- Standard request/response format
- Works with any OpenAI client
- Easy integration with Cline

## Configuration Flow

```
config.yaml → config.py → main.py
                    ↓
                 router.py
                    ↓
              models/
              local.py & cerebras.py
```

## Request Flow

```
Cline → FastAPI (main.py)
         ↓
    SmartRouter (router.py)
         ↓
    TaskComplexityClassifier
         ↓
    Local Model OR Cerebras
         ↓
    Response with model_used field
```

## Features Implemented

✅ Smart routing based on task complexity
✅ Automatic fallback on errors
✅ Rate limit handling
✅ Speculative decoding framework
✅ Token usage tracking
✅ Cost monitoring
✅ Comprehensive logging
✅ OpenAI-compatible API
✅ Configuration management
✅ Test suite
✅ Documentation

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Review and update `config.yaml` if needed
3. Start the server: `python main.py`
4. Test the API: `python test_api.py`
5. Configure Cline to use `http://localhost:8000/v1`
