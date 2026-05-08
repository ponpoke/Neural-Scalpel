import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
from neural_scalpel.serving.proxy_engine import ProxyServingEngine
from neural_scalpel.serving.backend_registry import BackendRegistry

@pytest.fixture
def registry():
    reg = BackendRegistry()
    reg.register_backend("alpaca", "http://backend-a:8000")
    return reg

@pytest.mark.asyncio
async def test_proxy_forward_success(registry):
    engine = ProxyServingEngine(registry)
    
    # Mock request and tenant context
    mock_req = MagicMock()
    mock_req.route_id = "alpaca"
    mock_req.prompt = "Hello"
    mock_req.max_tokens = 50
    mock_req.temperature = 0.0
    
    mock_tenant = MagicMock()
    mock_tenant.tenant_id = "t1"
    
    # Mock httpx.AsyncClient.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"text": "Success"}]}
    
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        result = await engine.infer(mock_req, mock_tenant, "audit-123")
        
        assert result["choices"][0]["text"] == "Success"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://backend-a:8000"
        assert kwargs["json"]["prompt"] == "Hello"
    
    await engine.close()

@pytest.mark.asyncio
async def test_proxy_rejects_unknown_route(registry):
    engine = ProxyServingEngine(registry)
    mock_req = MagicMock()
    mock_req.route_id = "unknown"
    
    with pytest.raises(RuntimeError, match="Could not resolve backend"):
        await engine.infer(mock_req, MagicMock(), "audit-123")
    
    await engine.close()

@pytest.mark.asyncio
async def test_proxy_handles_timeout(registry):
    engine = ProxyServingEngine(registry)
    mock_req = MagicMock()
    mock_req.route_id = "alpaca"
    
    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.TimeoutException("Timeout", request=None)):
        with pytest.raises(RuntimeError, match="Backend timeout"):
            await engine.infer(mock_req, MagicMock(), "audit-123")
            
    # Backend should be marked as unhealthy
    assert registry.resolve_backend("alpaca") is None
    
    await engine.close()

@pytest.mark.asyncio
async def test_proxy_handles_5xx(registry):
    engine = ProxyServingEngine(registry)
    mock_req = MagicMock()
    mock_req.route_id = "alpaca"
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=None, response=mock_response)
    
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(RuntimeError, match="Backend error 500"):
            await engine.infer(mock_req, MagicMock(), "audit-123")
    
    await engine.close()

def test_proxy_engine_health(registry):
    engine = ProxyServingEngine(registry)
    health = engine.get_health()
    assert health["engine_type"] == "external_proxy"
    assert health["total_routes"] == 1
    assert health["ready_routes"] == 1
    assert health["status"] == "healthy"
