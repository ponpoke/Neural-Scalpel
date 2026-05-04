"""
Neural-Scalpel vLLM Proxy (Prototype)

Step 4A + API Hardening: External vLLM Server Integration
Features:
- Route-aware temporal isolation
- JWT-based Tenant Authorization
- Admin endpoint protection via API Keys
- Basic Rate Limiting
- Payload Size Limits
- Prometheus Metrics Export
"""

import os
import asyncio
import time
import jwt
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import httpx

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Neural-Scalpel vLLM Gateway")

# ── Configuration ──────────────────────────────────────────────
VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://localhost:8000/v1/completions")
MOCK_BACKEND = os.getenv("VLLM_BACKEND_URL") is None
MAX_BATCH_SIZE = 32
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-eval-key-32-bytes-long")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin-secret-key")
MAX_REQUEST_SIZE_BYTES = 1024 * 512  # 512 KB

# ── Prometheus Metrics ─────────────────────────────────────────
REQ_TOTAL = Counter('scalpel_requests_total', 'Total requests', ['tenant_id', 'route_id'])
REJECTS_TOTAL = Counter('scalpel_rejected_total', 'Rejected requests', ['reason'])
ROUTE_SWAPS = Counter('scalpel_route_swaps_total', 'Number of VRAM route swaps')
REQ_LATENCY = Histogram('scalpel_request_latency_seconds', 'Latency of requests')
ACTIVE_ROUTE = Gauge('scalpel_active_route_info', 'Currently active route', ['route_id'])

# ── State ──────────────────────────────────────────────────────
class InferRequest(BaseModel):
    route_id: str
    prompt: str
    max_tokens: int = 50

class RouteQueue:
    def __init__(self):
        self.requests = []
        self.futures = []

queues: Dict[str, RouteQueue] = {}
current_active_route: Optional[str] = None
_proxy_lock: Optional[asyncio.Lock] = None

audit_log = []

def get_proxy_lock() -> asyncio.Lock:
    global _proxy_lock
    if _proxy_lock is None:
        _proxy_lock = asyncio.Lock()
    return _proxy_lock

# ── Rate Limiting (Simple Token Bucket per tenant) ─────────────
class RateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}

    def consume(self, key: str, tokens: int = 1) -> bool:
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.capacity
            self.last_refill[key] = now

        # Refill
        elapsed = now - self.last_refill[key]
        self.tokens[key] = min(self.capacity, self.tokens[key] + elapsed * self.refill_rate)
        self.last_refill[key] = now

        if self.tokens[key] >= tokens:
            self.tokens[key] -= tokens
            return True
        return False

# Allow 100 requests per second per tenant
tenant_rate_limiter = RateLimiter(capacity=200, refill_rate=100.0)

# ── Middleware & Security ──────────────────────────────────────
security_bearer = HTTPBearer()
api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    # Request size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE_BYTES:
        REJECTS_TOTAL.labels(reason="payload_too_large").inc()
        return PlainTextResponse("Payload too large", status_code=413)
    return await call_next(request)

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security_bearer)) -> str:
    """Verifies JWT and extracts tenant_id."""
    token = credentials.credentials
    try:
        # In production, verify issuer, expiry, signature algorithm
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            raise ValueError("Missing tenant_id in token")
        return tenant_id
    except Exception as e:
        REJECTS_TOTAL.labels(reason="invalid_jwt").inc()
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

def verify_admin(api_key: str = Security(api_key_header)):
    if api_key != ADMIN_API_KEY:
        REJECTS_TOTAL.labels(reason="unauthorized_admin").inc()
        raise HTTPException(status_code=403, detail="Admin API key required")
    return True

# ── Backend Processing ─────────────────────────────────────────

async def _forward_to_backend(prompt: str, max_tokens: int, route_id: str) -> dict:
    if MOCK_BACKEND:
        await asyncio.sleep(0.05)
        return {"choices": [{"text": f" [Mock Completion for {route_id}]"}], "usage": {"total_tokens": max_tokens}}
        
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                VLLM_BACKEND_URL,
                json={"model": "Qwen/Qwen2.5-0.5B", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

async def _process_route_batch(route_id: str):
    global current_active_route
    try:
        lock = get_proxy_lock()
        async with lock:
            if route_id not in queues or not queues[route_id].requests:
                return

            queue = queues[route_id]
            reqs_to_process = queue.requests[:MAX_BATCH_SIZE]
            futs_to_resolve = queue.futures[:MAX_BATCH_SIZE]
            
            queue.requests = queue.requests[MAX_BATCH_SIZE:]
            queue.futures = queue.futures[MAX_BATCH_SIZE:]

            if current_active_route != route_id:
                current_active_route = route_id
                ACTIVE_ROUTE.labels(route_id=route_id).set(1.0)
                ROUTE_SWAPS.inc()
                await asyncio.sleep(0.01)  # Simulated swap overhead

            for req, tenant_id in reqs_to_process:
                audit_log.append({
                    "timestamp": time.time(),
                    "route_id": route_id,
                    "tenant_id": tenant_id,
                    "event": "proxy_forward",
                })

            tasks = [_forward_to_backend(r.prompt, r.max_tokens, route_id) for r, t in reqs_to_process]
            results = await asyncio.gather(*tasks)

            for fut, res in zip(futs_to_resolve, results):
                if not fut.done():
                    fut.set_result(res)
    except Exception as e:
        print(f"Error in _process_route_batch: {e}")
        if 'futs_to_resolve' in locals():
            for fut in futs_to_resolve:
                if not fut.done():
                    fut.set_exception(e)

# ── API Endpoints ──────────────────────────────────────────────

@app.post("/v1/infer")
async def infer_endpoint(request: InferRequest, tenant_id: str = Depends(verify_jwt)):
    # 1. Rate Limiting
    if not tenant_rate_limiter.consume(tenant_id):
        REJECTS_TOTAL.labels(reason="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Too many requests")

    route_id = request.route_id
    REQ_TOTAL.labels(tenant_id=tenant_id, route_id=route_id).inc()
    
    if route_id not in queues:
        queues[route_id] = RouteQueue()
        
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    
    queues[route_id].requests.append((request, tenant_id))
    queues[route_id].futures.append(fut)
    
    asyncio.create_task(_process_route_batch(route_id))
    
    try:
        t0 = time.time()
        result = await fut
        latency = time.time() - t0
        REQ_LATENCY.observe(latency)
        
        result["proxy_metrics"] = {"latency_ms": round(latency * 1000, 2), "tenant_id": tenant_id}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def healthz():
    return {"status": "ok", "active_route": current_active_route, "mode": "MOCK" if MOCK_BACKEND else "LIVE"}


@app.get("/admin/metrics", dependencies=[Depends(verify_admin)])
def get_prometheus_metrics():
    """Prometheus export endpoint (Admin Only)"""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/audit", dependencies=[Depends(verify_admin)])
def get_audit_log():
    """Returns the internal audit log (Admin Only)"""
    return {"audit_entries": audit_log}
