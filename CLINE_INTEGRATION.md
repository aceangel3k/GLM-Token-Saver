# Cline Integration Guide

This guide explains how to configure Cline to use the GLM Token Saver API.

## Configuration

### Step 1: Start the API Server

Make sure the API server is running:

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Step 2: Configure Cline

In Cline's settings, configure the following:

#### API Base URL
```
http://localhost:8000/v1
```
**Note:** The API supports both `/v1` and non-`/v1` prefixes. You can use either:
- `http://localhost:8000/v1` (recommended, OpenAI-compatible)
- `http://localhost:8000` (also works)

#### API Key
```
any-string
```
(Note: The API key is optional and not used for authentication in the current implementation)

#### Model
```
Leave blank for auto-routing
```
Or specify one of:
- `unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-Q4_K_M` (local model)
- `zai-glm-4.7` (Cerebras model)

## How It Works

When you use Cline with this API:

1. **Simple Tasks** (questions, explanations, basic requests)
   - Automatically routed to local GLM-4.7-Flash
   - Fast and free
   - No API costs

2. **Complex Tasks** (coding, debugging, architecture)
   - Automatically routed to Cerebras GLM 4.7
   - Higher quality responses
   - Uses API credits

3. **Rate Limits**
   - If Cerebras hits rate limits, automatically falls back to local model
   - Ensures continuous operation

4. **Cost Savings**
   - Simple tasks use local model (free)
   - Only complex tasks use Cerebras (paid)
   - Significant cost reduction

## Example Usage

### Simple Question (Uses Local Model)
```
User: What is Python?
Response: [Fast, free response from local model]
```

### Complex Coding Task (Uses Cerebras)
```
User: Write a function to implement a binary search algorithm
Response: [High-quality response from Cerebras]
```

### Debugging Task (Uses Cerebras)
```
User: Debug this code: [code snippet]
Response: [Detailed analysis from Cerebras]
```

## Monitoring

### Check Which Model Was Used

The API response includes a `model_used` field:
- `"local"` - Used local model
- `"cerebras"` - Used Cerebras model

### View Logs

Check `logs/api_server.log` to see:
- Which model was used for each request
- Task complexity classification
- Any errors or fallbacks

### Check Statistics

Visit `http://localhost:8000/stats` to see:
- Current routing strategy
- Enabled features
- Usage statistics

## Customization

### Adjust Routing Strategy

Edit `config.yaml`:

```yaml
routing:
  strategy: "smart_routing"  # Options:
                              # - smart_routing (default)
                              # - speculative_decoding
                              # - always_local
                              # - always_cerebras
```

### Modify Complexity Keywords

Add or remove keywords that trigger Cerebras usage:

```yaml
routing:
  complexity_keywords:
    - "code"
    - "debug"
    - "architecture"
    - "implement"
    - "optimize"
    # Add your own keywords here
```

### Adjust Token Threshold

Change when tasks are considered complex based on length:

```yaml
routing:
  simple_task_threshold: 1000  # tokens
```

## Troubleshooting

### Cline Can't Connect
- Make sure the API server is running: `python main.py`
- Check that port 8000 is not blocked
- Verify the API Base URL is correct

### All Requests Use Cerebras
- Check the complexity keywords in `config.yaml`
- Adjust the `simple_task_threshold`
- Review logs to see why tasks are classified as complex

### All Requests Use Local Model
- Verify Cerebras API key is correct in `config.yaml`
- Check network connectivity to Cerebras
- Review logs for connection errors

### Slow Responses
- Local model should be fast (< 2 seconds)
- Cerebras may be slower (5-10 seconds)
- Check network latency to your DGX Spark
- Consider using `always_local` strategy if speed is critical

## Best Practices

1. **Start with smart_routing**: This provides the best balance of cost and quality
2. **Monitor costs**: Check logs to see how often Cerebras is used
3. **Adjust thresholds**: Fine-tune complexity keywords for your workflow
4. **Use local for simple tasks**: Saves costs and is faster
5. **Use Cerebras for complex tasks**: Ensures high-quality responses

## Advanced Configuration

### Force Specific Model

If you want to force Cline to use a specific model:

1. Set routing strategy in `config.yaml`:
   ```yaml
   routing:
     strategy: "always_local"  # or "always_cerebras"
   ```

2. Or specify the model in Cline's settings

### Enable Speculative Decoding

For advanced users, enable speculative decoding:

```yaml
routing:
  strategy: "speculative_decoding"

speculative_decoding:
  enabled: true
  draft_model: "local"
  verify_model: "cerebras"
  max_draft_tokens: 10
```

This will:
1. Generate draft from local model
2. Verify with Cerebras
3. Only send corrections when needed
4. Save on Cerebras API costs

## Support

For issues or questions:
1. Check `logs/api_server.log` for errors
2. Review the main README.md
3. Run `python test_api.py` to verify the API is working
4. Check that your DGX Spark and Cerebras endpoints are accessible
