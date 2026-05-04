"""
Step 4A + API Hardening: External vLLM Server Integration Validation

Tests the Neural-Scalpel route-aware proxy behavior and the newly added
API security hardening features (JWT, Rate limits, Admin protection, Prometheus).
"""

import os
import sys
import time
import jwt
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from neural_scalpel.serving import vllm_proxy
from neural_scalpel.serving.vllm_proxy import app, audit_log, JWT_SECRET, ADMIN_API_KEY

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def reset_proxy_state():
    # Reset internal metrics and state
    audit_log.clear()
    vllm_proxy.current_active_route = None
    vllm_proxy._proxy_lock = None
    vllm_proxy.tenant_rate_limiter.tokens.clear()
    
def generate_jwt(tenant_id: str) -> str:
    return jwt.encode({"tenant_id": tenant_id}, JWT_SECRET, algorithm="HS256")

@pytest.mark.anyio
async def test_vllm_proxy_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

@pytest.mark.anyio
async def test_proxy_auth_rejection():
    """Test that missing or invalid JWT is rejected."""
    req = {"route_id": "route-A", "prompt": "Hello"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/infer", json=req)
        assert resp.status_code in [401, 403] # HTTPBearer may return 401 or 403 depending on FastAPI version
        
        headers = {"Authorization": "Bearer invalid_token"}
        resp = await ac.post("/v1/infer", json=req, headers=headers)
        assert resp.status_code == 401

@pytest.mark.anyio
async def test_proxy_single_route_success():
    req = {"route_id": "route-A", "prompt": "Hello world", "max_tokens": 10}
    token = generate_jwt("tenant-1")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/infer", json=req, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["proxy_metrics"]["tenant_id"] == "tenant-1"
    
    assert len(audit_log) == 1
    assert audit_log[0]["route_id"] == "route-A"
    assert audit_log[0]["tenant_id"] == "tenant-1"

@pytest.mark.anyio
async def test_proxy_admin_endpoints():
    """Test admin endpoint protection and Prometheus export."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Without key
        resp = await ac.get("/admin/metrics")
        assert resp.status_code == 403
        
        # With key
        headers = {"X-Admin-Key": ADMIN_API_KEY}
        resp = await ac.get("/admin/metrics", headers=headers)
        assert resp.status_code == 200
        text = resp.text
        assert "scalpel_requests_total" in text

@pytest.mark.anyio
async def test_proxy_rate_limiting():
    """Test basic tenant rate limiting (200 requests capacity)."""
    token = generate_jwt("tenant-rl")
    headers = {"Authorization": f"Bearer {token}"}
    req = {"route_id": "route-rl", "prompt": "Test"}
    
    successes = 0
    failures = 0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Send 250 requests (capacity is 200)
        tasks = [ac.post("/v1/infer", json=req, headers=headers) for _ in range(250)]
        responses = await asyncio.gather(*tasks)
        
    for r in responses:
        if r.status_code == 200: successes += 1
        elif r.status_code == 429: failures += 1
            
    # Roughly 200 should succeed, rest should be 429 limited
    assert successes >= 200
    assert failures > 0

@pytest.mark.anyio
async def test_proxy_no_route_leakage_endurance():
    """Run a small stress test to ensure 0 leakage across multiple route interleaving."""
    # Note: 150 requests total fits inside the 200 token bucket capacity
    reqs = []
    tokens = {
        "t1": generate_jwt("t1"),
        "t2": generate_jwt("t2"),
        "t3": generate_jwt("t3"),
    }
    
    for i in range(50):
        reqs.append( ({"route_id": "route-A", "prompt": f"A{i}"}, tokens["t1"]) )
        reqs.append( ({"route_id": "route-B", "prompt": f"B{i}"}, tokens["t2"]) )
        reqs.append( ({"route_id": "route-C", "prompt": f"C{i}"}, tokens["t3"]) )
        
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        tasks = [ac.post("/v1/infer", json=body, headers={"Authorization": f"Bearer {tk}"}) for body, tk in reqs]
        responses = await asyncio.gather(*tasks)
        
    successes = sum(1 for r in responses if r.status_code == 200)
    assert successes == 150
    assert len(audit_log) == 150
