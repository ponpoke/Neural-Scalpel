import asyncio
from multiprocessing import Process
import time
import uvicorn
from fastapi import FastAPI
import httpx

from neural_scalpel.serving.proxy_engine import ProxyServingEngine
from neural_scalpel.serving.backend_registry import BackendRegistry


# 1. Simple Mock Backend Server
mock_backend = FastAPI()

@mock_backend.post("/v1/completions")
async def completions(req: dict):
    # Simulate a vLLM-style response
    return {
        "id": "chat-123",
        "choices": [{"text": f"Echo: {req.get('prompt')}"}],
        "usage": {"total_tokens": 10}
    }

def run_backend():
    uvicorn.run(mock_backend, host="127.0.0.1", port=8001, log_level="warning")


async def wait_for_backend(url: str, timeout: float = 5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                # Try to ping the backend
                resp = await client.post(url, json={"prompt": "ping"})
                if resp.status_code < 500:
                    return
            except Exception:
                await asyncio.sleep(0.2)

    raise RuntimeError(f"Backend did not become ready: {url}")


async def main():
    print("\n--- Starting Enhanced Proxy Forwarding Smoke Test ---")
    p = Process(target=run_backend)
    p.start()

    engine = None
    try:
        backend_url = "http://127.0.0.1:8001/v1/completions"
        print(f"Waiting for backend at {backend_url}...")
        await wait_for_backend(backend_url)

        # 2. Setup ProxyServingEngine
        registry = BackendRegistry()
        registry.register_backend("alpaca", backend_url)
        engine = ProxyServingEngine(registry)

        # 3. Simulate Request and Tenant Context
        class MockRequest:
            route_id = "alpaca"
            prompt = "Live Smoke Test"
            max_tokens = 16
            temperature = 0.0
        
        class MockTenant:
            tenant_id = "t-smoke"

        print(f"Resolving route 'alpaca' to {registry.resolve_backend('alpaca')}")
        
        result = await engine.infer(MockRequest(), MockTenant(), "audit-smoke")
        
        print(f"[SMOKE TEST RESULT]: {result}")
        assert "Echo: Live Smoke Test" in result["choices"][0]["text"]
        print("\nPASS: Live proxy forwarding to local HTTP backend verified.")

    except Exception as e:
        print(f"\nFAIL: Smoke test failed with error: {str(e)}")
        raise
    finally:
        print("Cleaning up resources...")
        if engine is not None:
            await engine.close()
        
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            print("Force killing backend process...")
            p.kill()
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main())
