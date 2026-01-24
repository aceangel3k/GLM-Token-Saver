# GLM Token Saver API

A local API server that implements speculative decoding and smart routing for GLM models, helping you save on API costs and credits while maintaining high-quality responses.

## Features

- **Smart Routing**: Automatically routes requests to the appropriate model based on task complexity
- **Speculative Decoding**: Uses local model for draft generation, verifies with cloud model
- **Cost Savings**: Routes simple tasks to local model (free), complex tasks to Cerebras (when needed)
- **Rate Limit Handling**: Automatic fallback to local model when Cerebras hits rate limits
- **OpenAI-Compatible**: Works with any OpenAI-compatible client, including Cline
- **Token Tracking**: Monitor usage and costs

## Architecture

```
Cline (AI Assistant)
    ↓
Local API Server (FastAPI)
    ↓
┌─────────────────────────────────────┐
│  Smart Router                       │
│  - Task complexity classifier       │
│  - Automatic fallback               │
│  - Rate limit detection             │
└─────────────────────────────────────┘
    ↓                    ↓
Local GLM-4.7-Flash    Cerebras GLM 4.7
(llama.cpp)            (355B, 131k ctx)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your endpoints in `config.yaml`:
   - Local model endpoint (llama.cpp)
   - Cerebras API endpoint and key

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  local:
    enabled: true
    endpoint: "http://spark0.tail1f104f.ts.net:41447/v1/chat/completions"
    model: "unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-Q4_K_M"
  
  cerebras:
    enabled: true
    endpoint: "https://api.cerebras.ai/v1/chat/completions"
    api_key: "your-api-key"
    model: "glm-4.7-355b"

routing:
  strategy: "smart_routing"  # Options: smart_routing, speculative_decoding, always_local, always_cerebras
  simple_task_threshold: 1000
  complexity_keywords: ["code", "debug", "architecture", ...]
```

## Running the Server

Start the API server:

```bash
python main.py
```

Or use uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## Using with Cline

Configure Cline to use this API:

1. Set API Base URL: `http://localhost:8000/v1`
2. Set API Key: `any-string` (optional, not used for authentication)
3. Set Model: Leave blank for auto-routing, or specify a model name

The API will automatically:
- Route simple tasks to local model (free)
- Route complex tasks to Cerebras (when needed)
- Fallback to local model on rate limits
- Track token usage and costs

## API Endpoints

### `POST /v1/chat/completions`
OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.9,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "glm-4.7",
  "choices": [...],
  "usage": {...},
  "model_used": "local"
}
```

### `GET /health`
Health check endpoint.

### `GET /v1/models`
List available models.

### `GET /stats`
Get usage statistics.

## Routing Strategies

### `smart_routing` (Default)
- Classifies task complexity
- Simple tasks → Local model
- Complex tasks → Cerebras
- Automatic fallback on errors

### `speculative_decoding`
- Generates draft from local model
- Verifies with Cerebras
- Only sends corrections when needed

### `always_local`
- Always uses local model
- No API costs

### `always_cerebras`
- Always uses Cerebras
- Best quality, highest cost

## Task Complexity Classification

Tasks are classified as complex if they:
- Contain complexity keywords (code, debug, architecture, etc.)
- Exceed token threshold
- Are detected as requiring advanced reasoning

## Logging

Logs are written to `logs/api_server.log` with rotation.

## Cost Tracking

Monitor your Cerebras API usage:
- Set budget limits in `config.yaml`
- Track token usage
- Get alerts when approaching limits

## Troubleshooting

### Connection Issues
- Check that llama.cpp is running on your DGX Spark
- Verify the endpoint URLs in `config.yaml`
- Check network connectivity

### Rate Limits
- The API automatically falls back to local model
- Adjust routing strategy if needed

### Performance
- Local model is faster for simple tasks
- Cerebras is slower but more capable
- Adjust `simple_task_threshold` to balance speed vs. quality

## License

MIT License
