# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Verify Configuration

Check that `config.yaml` has your correct endpoints:
- Local model: `http://<local_domain_or_ip>:41447/v1/chat/completions`
- Cerebras: `https://api.cerebras.ai/v1/chat/completions`

## 3. Start the Server

```bash
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 4. Test the API

In a new terminal, run:

```bash
python test_api.py
```

This will test:
- Health endpoint
- Models list
- Simple chat completion (local model)
- Complex chat completion (Cerebras)
- Stats endpoint

## 5. Use with Cline

Configure Cline with:
- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: `any-string` (optional)
- **Model**: Leave blank for auto-routing

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed

### Connection errors
- Make sure llama.cpp is running on your DGX Spark
- Check network connectivity to `<local_domain_or_ip>`
- Verify Cerebras API key is valid

### Tests fail
- Ensure the server is running before running tests
- Check logs in `logs/api_server.log` for errors

## Next Steps

- Customize routing strategy in `config.yaml`
- Adjust complexity keywords for your use case
- Monitor costs via the `/stats` endpoint
- Review logs to see which model is being used
