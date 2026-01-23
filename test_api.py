"""Test script for GLM Token Saver API."""
import asyncio
import httpx
import json


async def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def test_models():
    """Test models endpoint."""
    print("\n=== Testing Models Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/v1/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def test_chat_completion_simple():
    """Test simple chat completion (should use local model)."""
    print("\n=== Testing Simple Chat Completion ===")
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model used: {data.get('model_used', 'unknown')}")
        print(f"Tokens used: {data.get('usage', {})}")
        print(f"Response: {data.get('choices', [{}])[0].get('message', {}).get('content', '')[:200]}...")


async def test_chat_completion_complex():
    """Test complex chat completion (should use Cerebras)."""
    print("\n=== Testing Complex Chat Completion ===")
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "messages": [
                {"role": "user", "content": "Write a Python function to implement a binary search algorithm with proper error handling and documentation."}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model used: {data.get('model_used', 'unknown')}")
        print(f"Tokens used: {data.get('usage', {})}")
        print(f"Response: {data.get('choices', [{}])[0].get('message', {}).get('content', '')[:300]}...")


async def test_stats():
    """Test stats endpoint."""
    print("\n=== Testing Stats Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("GLM Token Saver API - Test Suite")
    print("=" * 60)
    
    try:
        await test_health()
        await test_models()
        await test_chat_completion_simple()
        await test_chat_completion_complex()
        await test_stats()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to the API server.")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
