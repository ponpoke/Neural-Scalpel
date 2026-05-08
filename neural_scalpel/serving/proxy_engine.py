import httpx
import time
from typing import Any, Optional
from .engine import ServingEngine
from .backend_registry import BackendRegistry

class ProxyServingEngine(ServingEngine):
    """
    Serving engine that forwards requests to external vLLM backends.
    
    This engine implements Mode B: External Proxy Fallback. It does not perform
    any local weight swapping. Instead, it uses a BackendRegistry to resolve
    which external process or instance should handle a specific route.
    """
    def __init__(
        self,
        registry: BackendRegistry,
        timeout_seconds: float = 30.0,
        max_retries: int = 0
    ):
        self.registry = registry
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def infer(self, req: Any, tenant_ctx: Any, audit_ref: str) -> Any:
        """
        Forwards the inference request to the resolved backend.
        
        The 'req' is expected to have a 'route_id' and 'prompt' (or similar payload).
        We resolve the backend URL via the registry.
        """
        route_id = getattr(req, "route_id", None)
        if not route_id:
            raise ValueError("Request must contain a route_id")

        backend_url = self.registry.resolve_backend(route_id)
        if not backend_url:
            status = self.registry.get_route_status(route_id)
            raise RuntimeError(f"Could not resolve backend for route '{route_id}': status={status}")

        # Forwarding logic
        # Note: In a real implementation, we would extract the payload from 'req'
        # and forward it to the OpenAI-compatible vLLM endpoint.
        # For Phase C, we assume a standard vLLM /v1/completions or /v1/chat/completions structure.
        
        # We'll use a simplified forwarding for the prototype/tests.
        tenant_id = getattr(tenant_ctx, "tenant_id", None)
        payload = {
            "model": "neural-scalpel-routed", # The backend might ignore this or use it for routing
            "prompt": getattr(req, "prompt", ""),
            "max_tokens": getattr(req, "max_tokens", 50),
            "temperature": getattr(req, "temperature", 0.0),
            "extra_args": {"audit_ref": audit_ref, "tenant_id": tenant_id}
        }

        try:
            resp = await self._client.post(backend_url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            self.registry.update_health(backend_url, is_healthy=False)
            raise RuntimeError(f"Backend timeout for route '{route_id}' at {backend_url}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                self.registry.update_health(backend_url, is_healthy=False)
            raise RuntimeError(f"Backend error {exc.response.status_code} for route '{route_id}'")
        except Exception as e:
            raise RuntimeError(f"Proxy forwarding failed: {str(e)}")

    def get_health(self) -> dict:
        """Returns health status including registry summary."""
        routes = self.registry.list_routes()
        ready_count = sum(1 for r in routes if self.registry.get_route_status(r) == "READY")
        return {
            "engine_type": "external_proxy",
            "total_routes": len(routes),
            "ready_routes": ready_count,
            "status": "healthy" if ready_count > 0 or not routes else "degraded"
        }

    async def close(self):
        await self._client.aclose()
