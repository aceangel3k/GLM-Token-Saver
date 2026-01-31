# GLM Token Saver API

A local API server that implements speculative decoding and smart routing for GLM models, helping you save on API costs and credits while maintaining high-quality responses.

[▶ Watch the demo video](https://x.com/aceangel/status/2015122694094967017?s=20)

## What is Speculative Decoding?

Speculative decoding is an optimization technique that dramatically speeds up LLM inference while maintaining high quality. Here's how it works:

### How It Works

1. **Draft Generation**: A smaller, faster model (draft model) generates a draft response
2. **Verification**: A larger, more accurate model (target model) verifies the draft
3. **Acceptance/Correction**: 
   - If the target model agrees with the draft → Use the draft (saves time and money)
   - If the target model disagrees → Use the corrected version (ensures quality)

### Why Use It?

**Speed**: 
- The draft model generates tokens much faster than the target model
- Most tokens are accepted without needing the expensive target model
- Typical speedup: 2-4x faster than using only the target model

**Cost Savings**:
- Local draft model = FREE
- Cloud target model = Paid (Cerebras, OpenAI, etc.)
- Only pay for target model when verification is needed
- Can reduce API costs by 60-80%

**Quality Preservation**:
- Target model ensures accuracy and quality
- Only accepts high-quality predictions from draft
- Automatic fallback to target model for complex tasks

### In This Project

- **Draft Model**: Local GLM-4.7-Flash (free, fast enough, runs on your hardware)
- **Target Model**: Cerebras GLM 4.7 (paid, much faster, more capable)
- **Process**:
  - Generate draft locally (free)
  - Verify with Cerebras (paid, only when needed)
  - Use whichever response is better based on similarity score

### Example

```
User asks: "Explain quantum computing"

1. Local model generates draft (0.5s, free):
   "Quantum computing uses quantum bits..."

2. Cerebras verifies draft (2.0s, paid):
   "Quantum computing uses quantum bits (qubits) to perform calculations..."

3. Compare similarity: 85%
   - Since 85% > 70% threshold → Use draft (FREE!)
   - Result: Fast response, high quality, zero cost
```

## Features

- **Speculative Decoding**: Uses local model for draft generation, verifies with cloud model
- **Parallel Speculative Decoding**: Launches multiple draft requests concurrently for faster response times
- **Dynamic Rate-Aware Routing**: Automatically shifts requests to local model when approaching Cerebras rate limits
- **Smart Routing**: Automatically routes requests to the appropriate model based on task complexity
- **Adaptive Cerebras Routing**: Maximizes Cerebras usage while automatically respecting rate limits
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
    endpoint: "http://<your-local-api-server-ip>:41447/v1/chat/completions"
    model: "unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-Q4_K_M"
  
  cerebras:
    enabled: true
    endpoint: "https://api.cerebras.ai/v1/chat/completions"
    api_key: "your-api-key"
    model: "zai-glm-4.7"

routing:
  strategy: "smart_routing"  # Options: smart_routing, smart_speculative, speculative_decoding, parallel_speculative, adaptive_cerebras, always_local, always_cerebras
  simple_task_threshold: 1000
  complexity_keywords: ["code", "debug", "architecture", ...]
  adaptive_cerebras:
    cool_down_duration: 60  # seconds
    rate_limit_threshold: 0.2  # 20% remaining

speculative_decoding:
  enabled: true
  draft_model: "local"
  verify_model: "cerebras"
  max_draft_tokens: 100      # Max tokens in draft before verification
  min_confidence: 0.7        # Similarity threshold (0.0-1.0)
  parallel_enabled: true     # Enable parallel speculative decoding
  max_concurrent_drafts: 3   # Maximum concurrent drafts
  draft_timeout: 5           # Timeout for draft generation (seconds)
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
Get usage statistics and configuration.

**Response:**
```json
{
  "uptime": "2h 15m",
  "requests": {
    "total": 150,
    "by_model": {
      "local": 120,
      "cerebras": 30
    }
  },
  "tokens": {
    "total": 45000,
    "prompt": 15000,
    "completion": 30000,
    "by_model": {
      "local": {"prompt": 10000, "completion": 20000, "total": 30000},
      "cerebras": {"prompt": 5000, "completion": 10000, "total": 15000}
    }
  },
  "costs": {
    "total_cost_usd": 0.03,
    "cost_saved_usd": 0.02,
    "cerebras_cost_per_1k_tokens": 0.002
  },
  "speculative_decoding": {
    "total_drafts": 50,
    "total_verifications": 20,
    "drafts_accepted": 35,
    "drafts_rejected": 15,
    "acceptance_rate_percent": 70.0,
    "total_draft_tokens": 5000,
    "total_verification_tokens": 3000,
    "tokens_saved": 3000,
    "tokens_saved_percent": 6.67
  },
  "errors": {
    "total": 5,
    "by_model": {
      "local": 2,
      "cerebras": 3
    }
  },
  "rate_limits": {
    "total_hits": 10
  },
  "performance": {
    "avg_tokens_per_request": 300
  }
}
```

### `POST /stats/reset`
Reset all statistics to zero.

## Routing Strategies

### `smart_routing` (Default)
- Classifies task complexity
- Simple tasks → Local model
- Complex tasks → Cerebras
- Automatic fallback on errors

### `speculative_decoding`
- Generates draft from local model with limited tokens
- Verifies draft with Cerebras for completeness
- Uses similarity score to decide which response to use
- Only sends to Cerebras when needed (incomplete or uncertain drafts)
- Balances speed (local) and quality (Cerebras)

**Configuration:**
```yaml
speculative_decoding:
  enabled: true
  draft_model: "local"
  verify_model: "cerebras"
  max_draft_tokens: 100      # Max tokens in draft before verification
  min_confidence: 0.7        # Similarity threshold (0.0-1.0)
  parallel_enabled: true     # Enable parallel speculative decoding
  max_concurrent_drafts: 3   # Maximum concurrent drafts
  draft_timeout: 5           # Timeout for draft generation (seconds)
```

### `parallel_speculative`
- Launches multiple draft requests to local model concurrently
- Selects the best response based on quality metrics (length, token count)
- Verifies selected draft with Cerebras only when needed
- Falls back to sequential mode if parallel generation fails
- **Benefits:**
  - Significantly faster response times
  - Higher quality drafts through multiple attempts
  - Graceful degradation to sequential mode

**Configuration:**
```yaml
speculative_decoding:
  parallel_enabled: true
  max_concurrent_drafts: 3   # Number of concurrent drafts
  draft_timeout: 5           # Timeout per draft (seconds)
  max_draft_tokens: 100      # Max tokens per draft
  min_confidence: 0.7        # Similarity threshold for verification
```

**How it Works:**
1. Launches N concurrent draft requests to local model
2. Waits for all drafts to complete or timeout
3. Selects the best draft based on quality metrics
4. Verifies selected draft with Cerebras if needed
5. Uses verified response or falls back to sequential mode

**Tuning Tips:**
- Increase `max_concurrent_drafts` for better quality (slower)
- Decrease `draft_timeout` for faster responses (may miss some drafts)
- Adjust `min_confidence` to control verification strictness

**How to Tune:**

- **For more cost savings**: Increase `max_draft_tokens` (more tokens from local model)
- **For higher quality**: Decrease `max_draft_tokens` (more verification with Cerebras)
- **To accept more drafts**: Decrease `min_confidence` (more lenient with quality)
- **To be more strict**: Increase `min_confidence` (require higher similarity)

**Example Scenarios:**

1. **Simple question** ("What is 2+2?"):
   - Draft: "2+2 equals 4" (5 tokens)
   - Draft < max_draft_tokens → No verification needed
   - **Result: FREE, FAST**

2. **Complex explanation** ("Explain quantum computing"):
   - Draft: "Quantum computing uses..." (100 tokens, hit limit)
   - Verify with Cerebras
   - Similarity: 85% > 70% threshold → Use draft
   - **Result: FREE, HIGH QUALITY**

3. **Code generation** ("Write a binary search"):
   - Draft: "def binary_search..." (100 tokens, hit limit)
   - Verify with Cerebras
   - Similarity: 30% < 70% threshold → Use Cerebras response
   - **Result: PAID, HIGHEST QUALITY**

### `always_local`
- Always uses local model
- No API costs

### `always_cerebras`
- Always uses Cerebras
- Best quality, highest cost

### `smart_speculative`
- Combines smart routing with speculative decoding
- Uses speculative decoding for complex tasks
- Falls back to local model for simple tasks
- Automatic rate limit handling

### `adaptive_cerebras`
An intelligent routing strategy that maximizes Cerebras API usage while automatically respecting rate limits. It seamlessly switches between Cerebras and local inference based on real-time rate limit information from the API.

**How It Works:**

1. **Primary Mode: Cerebras** - By default, all requests are routed to Cerebras for best quality and fastest response times

2. **Rate Limit Monitoring** - Monitors rate limit headers from Cerebras API:
   - `x-ratelimit-limit-requests-day` - Maximum requests per day
   - `x-ratelimit-limit-tokens-minute` - Maximum tokens per minute
   - `x-ratelimit-remaining-requests-day` - Requests remaining today
   - `x-ratelimit-remaining-tokens-minute` - Tokens remaining this minute
   - `x-ratelimit-reset-requests-day` - Seconds until daily reset
   - `x-ratelimit-reset-tokens-minute` - Seconds until minute reset

3. **Automatic Fallback** - When remaining tokens or requests drop below threshold (default: 20%):
   - Switches to local inference
   - Logs the reason for fallback
   - Waits for cool-down period (default: 60 seconds)
   - Checks if rate limits have reset

4. **Automatic Recovery** - After cool-down period:
   - Checks if new rate limit data shows capacity is available
   - If available, switches back to Cerebras
   - If still near limits, continues using local model

**Configuration:**
```yaml
routing:
  strategy: "adaptive_cerebras"
  adaptive_cerebras:
    cool_down_duration: 60  # seconds
    rate_limit_threshold: 0.2  # 20% remaining
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cool_down_duration` | int | 60 | How long to wait on local model before checking Cerebras again (seconds) |
| `rate_limit_threshold` | float | 0.2 | Percentage threshold to trigger fallback (0.0-1.0) |

**Example Configurations:**

**Conservative (avoid rate limits entirely):**
```yaml
routing:
  strategy: "adaptive_cerebras"
  adaptive_cerebras:
    cool_down_duration: 60
    rate_limit_threshold: 0.3  # 30% remaining
```

**Aggressive (use Cerebras until very close to limit):**
```yaml
routing:
  strategy: "adaptive_cerebras"
  adaptive_cerebras:
    cool_down_duration: 30
    rate_limit_threshold: 0.1  # 10% remaining
```

**Quick Recovery (check back frequently):**
```yaml
routing:
  strategy: "adaptive_cerebras"
  adaptive_cerebras:
    cool_down_duration: 15  # Check every 15 seconds
    rate_limit_threshold: 0.2
```

**Benefits:**

- **Maximum Cost Efficiency**: Uses Cerebras whenever possible for best quality, automatically avoids rate limit errors, no wasted API calls
- **Continuous Operation**: Seamless fallback to local model, no service interruption for users, automatic recovery when capacity returns
- **No Manual Monitoring**: Automatically tracks rate limits from API headers, no need to manually monitor usage, self-managing system
- **Optimal Resource Utilization**: Maximizes Cerebras usage within limits, minimizes local fallback usage (only when necessary), balances quality and availability

**Use Cases:**

- **High-Volume Production**: Continuous operation under heavy load, automatic handling of burst traffic, maintains service availability
- **Shared API Keys**: Multiple applications sharing one API key, automatic load balancing across rate limits, prevents one app from consuming all quota
- **Cost-Conscious Operations**: Avoids wasted API calls, maximizes value from Cerebras subscription, automatic cost optimization
- **Development & Testing**: Test with real Cerebras responses, automatic fallback if limits hit during testing, no need to switch strategies manually

**Comparison with Other Strategies:**

| Strategy | Cerebras Usage | Rate Limit Handling | Cost | Quality |
|----------|---------------|---------------------|------|---------|
| `adaptive_cerebras` | Maximum | Automatic fallback | Optimal | Best |
| `always_cerebras` | 100% | Manual (fails on limit) | Highest | Best |
| `smart_routing` | Variable | Basic error handling | Low | Variable |
| `always_local` | 0% | N/A | Free | Lower |

## Task Complexity Classification

Tasks are classified as complex if they:
- Contain complexity keywords (code, debug, architecture, etc.)
- Exceed token threshold
- Are detected as requiring advanced reasoning

## Monitoring Speculative Decoding

When using `speculative_decoding` strategy, responses include metadata about the process:

**Response Metadata:**
```json
{
  "model_used": "local",
  "speculative_decoding": {
    "draft_tokens": 50,
    "verified_tokens": 0,
    "accepted": true,
    "spilled_over": false,
    "similarity": 0.0
  }
}
```

**What the metrics mean:**

- `draft_tokens`: Tokens generated by local draft model
- `verified_tokens`: Tokens from Cerebras verification
- `accepted`: Whether draft was accepted without verification
- `spilled_over`: Whether Cerebras response was used instead of draft
- `similarity`: Similarity score between draft and verification (0.0 to 1.0)

**Interpreting Results:**

1. **`accepted: true, spilled_over: false`**
   - Draft was good enough, no verification needed
   - **Cost: FREE**
   - **Quality: Good**

2. **`accepted: false, spilled_over: true, similarity: 0.8`**
   - Draft was verified, but similarity was high
   - Used Cerebras for better quality
   - **Cost: PAID**
   - **Quality: High**

3. **`accepted: false, spilled_over: true, similarity: 0.2`**
   - Draft was significantly different from Cerebras
   - Used Cerebras response
   - **Cost: PAID**
   - **Quality: Highest**

**Optimization Tips:**

- High `spilled_over` rate → Decrease `max_draft_tokens` to generate shorter drafts
- High similarity but still spilling over → Decrease `min_confidence`
- Too many errors → Increase `min_confidence` or decrease `max_draft_tokens`

## Logging

Logs are written to `logs/api_server.log` with rotation.

## Statistics Tracking

The API automatically tracks usage statistics including:

- **Requests**: Total requests and breakdown by model
- **Tokens**: Prompt/completion tokens and totals per model
- **Costs**: Total costs and savings from speculative decoding
- **Speculative Decoding**: Draft acceptance rates, tokens saved
- **Errors**: Error counts per model
- **Rate Limits**: Number of rate limit hits
- **Performance**: Average tokens per request

**View Statistics:**
```bash
curl http://localhost:8000/stats
```

**Reset Statistics:**
```bash
curl -X POST http://localhost:8000/stats/reset
```

**Key Metrics:**

- `acceptance_rate_percent`: Percentage of drafts accepted without verification (higher = more savings)
- `tokens_saved`: Total tokens saved by accepting drafts
- `cost_saved_usd`: Money saved through speculative decoding
- `tokens_saved_percent`: Percentage of total tokens saved

## Cost Tracking

Monitor your Cerebras API usage:
- Set budget limits in `config.yaml`
- Track token usage via `/stats` endpoint
- Get alerts when approaching limits (TODO)

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

### Adaptive Cerebras Issues

**Issue: Always using local model**
- **Possible causes**: Rate limit threshold too high, cool-down duration too long, rate limit headers not arriving
- **Solutions**:
  - Lower `rate_limit_threshold` in config
  - Reduce `cool_down_duration`
  - Check logs for rate limit info
  - Verify Cerebras API responses include headers

**Issue: Frequent switching between models**
- **Possible causes**: Rate limit threshold too close to current usage, cool-down duration too short
- **Solutions**:
  - Increase `cool_down_duration` to 60+ seconds
  - Adjust `rate_limit_threshold` to have buffer
  - Check logs for switching reasons

**Issue: Rate limit errors still occur**
- **Possible causes**: Threshold too low (triggering too late), burst traffic depleting limits between checks
- **Solutions**:
  - Increase `rate_limit_threshold` (be more conservative)
  - Reduce `cool_down_duration` (check more frequently)
  - Consider additional rate limiting at application level

## Testing Speculative Decoding

Run the test script to verify speculative decoding is working:

```bash
python test_speculative_decoding.py
```

This will run 3 tests:
1. Simple request (should use draft only)
2. Complex explanation (should verify with Cerebras)
3. Code generation (should verify with Cerebras)

**Expected Output:**
```
Test 1: SUCCESS
  Model used: local
  Draft tokens: 50
  Spilled over: False

Test 2: SUCCESS
  Model used: cerebras
  Draft tokens: 100
  Verified tokens: 500
  Spilled over: True
  Similarity: 0.45
```

## Testing Adaptive Cerebras Strategy

Run the adaptive cerebras test script to verify the strategy is working correctly:

```bash
python test_adaptive_cerebras.py
```

Tests cover:
- Rate limit header parsing
- Near-limit detection
- Percentage calculations
- Reset time calculations
- Edge cases
- Integration scenarios

## Statistics Module

The `statistics.py` module provides comprehensive tracking of API usage:

**Features:**
- Thread-safe statistics tracking with locks
- Request and token counting per model
- Speculative decoding metrics (drafts, verifications, acceptance rates)
- Cost calculations and savings
- Error and rate limit tracking
- Statistics persistence (save/load to JSON)

**Key Classes:**

- `StatisticsTracker`: Main tracking class
  - `record_request()`: Record a request and its token usage
  - `record_speculative_decoding()`: Record speculative decoding stats
  - `record_error()`: Record an error
  - `record_rate_limit()`: Record a rate limit hit
  - `get_statistics()`: Get current statistics as dict
  - `reset()`: Reset all statistics
  - `save_to_file()`: Save statistics to JSON file
  - `load_from_file()`: Load statistics from JSON file

**Example Usage:**
```python
from statistics import get_stats

# Get the global statistics instance
stats = get_stats()

# Record a request
stats.record_request('local', {
    'usage': {
        'prompt_tokens': 50,
        'completion_tokens': 100,
        'total_tokens': 150
    }
})

# Record speculative decoding
stats.record_speculative_decoding(
    draft_tokens=100,
    verified_tokens=150,
    accepted=True
)

# Get statistics
statistics = stats.get_statistics()
print(f"Total requests: {statistics['requests']['total']}")
print(f"Tokens saved: {statistics['speculative_decoding']['tokens_saved']}")
```

## License

MIT License 
