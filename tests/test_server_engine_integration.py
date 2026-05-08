import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from neural_scalpel.serving.server import create_app
from neural_scalpel.serving.engine import ServingEngine

class MockEngine(ServingEngine):
    def __init__(self):
        self.health = {"status": "healthy", "engine_type": "mock"}

    async def infer(self, req, tenant_ctx, audit_ref):
        return {"choices": [{"text": f"Mock output for {req.route_id}"}]}

    def get_health(self):
        return self.health

@pytest.fixture
def client():
    runtime = MagicMock()
    registry = MagicMock()
    # Mock registry behavior
    registry.get_route.return_value = {"tenant_id": "t1"}
    registry.get_route_status.return_value = MagicMock(value="READY")
    
    engine = MockEngine()
    app = create_app(runtime=runtime, registry=registry, engine=engine)
    return TestClient(app)

def test_infer_with_engine(client):
    payload = {
        "request_id": "req-1",
        "tenant_id": "t1",
        "route_id": "alpaca",
        "prompt": "Hello"
    }
    response = client.post("/v1/infer", json=payload)
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert "Mock output for alpaca" in data["output"]["choices"][0]["text"]

def test_healthz_with_engine(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_server_infer_with_proxy_engine_forwards_to_backend():
    from neural_scalpel.serving.proxy_engine import ProxyServingEngine
    from neural_scalpel.serving.backend_registry import BackendRegistry
    import httpx
    
    # 1. Setup real ProxyServingEngine
    registry = BackendRegistry()
    registry.register_backend("alpaca", "http://real-backend:8000/v1/completions")
    engine = ProxyServingEngine(registry)
    
    # 2. Setup RouteRegistry and App
    runtime = MagicMock()
    route_reg = MagicMock()
    route_reg.get_route.return_value = {"tenant_id": "t1"}
    route_reg.get_route_status.return_value = MagicMock(value="READY")
    
    app = create_app(runtime=runtime, registry=route_reg, engine=engine)
    new_client = TestClient(app)
    
    # 3. Mock httpx.AsyncClient.post
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"text": "Real Proxy Output"}]}
    
    payload = {
        "request_id": "req-proxy-1",
        "tenant_id": "t1",
        "route_id": "alpaca",
        "prompt": "Hello Proxy"
    }
    
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        
        # We need to run this in an event loop because it's async, but TestClient is sync.
        # Actually, TestClient handles the loop if the handler is async.
        response = new_client.post("/v1/infer", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["output"]["choices"][0]["text"] == "Real Proxy Output"
        
        # Verify forwarding to the correct backend URL
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://real-backend:8000/v1/completions"
    
    await engine.close()
