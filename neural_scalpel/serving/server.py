"""
Neural-Scalpel Pilot API Server

FastAPI-based HTTP server exposing the Neural-Scalpel Hot-Swap Runtime
as a network service. This is the Production Integration Prototype (Block D),
designed to validate:

  1. Route-isolated inference over HTTP
  2. Tenant access control at the API boundary
  3. Audit traceability for every request
  4. Metrics/observability endpoint for SRE dashboards
  5. Integration with the RouteAwareScheduler and HotSwapRuntime

The server does NOT implement streaming, authentication tokens, or TLS.
Those are Phase 6+ concerns. This prototype focuses exclusively on
correctness of route isolation and fail-close behavior.

Usage:
  from neural_scalpel.serving.server import create_app
  app = create_app(runtime=..., registry=..., audit_logger=...)
  uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from neural_scalpel.serving.schemas import (
    InferRequest,
    InferResponse,
    RouteRegisterRequest,
    RouteRegisterResponse,
    RouteListEntry,
    MetricsResponse,
)
from neural_scalpel.serving.metrics import MetricsCollector
from neural_scalpel.route.policy import RouteStatus


class PilotServer:
    """
    Encapsulates the serving state and provides route handlers.

    By separating server state from the FastAPI app factory, we gain:
      - Testability: inject mock runtimes, registries, and loggers
      - Lifecycle control: explicit startup/shutdown
      - Single Responsibility: this class owns the request→runtime pipeline
    """

    def __init__(
        self,
        runtime,  # HotSwapRuntime instance (or mock)
        registry,  # RouteRegistry instance (or mock)
        audit_logger=None,  # AuditLogger instance (optional)
        metrics: Optional[MetricsCollector] = None,
    ):
        self.runtime = runtime
        self.registry = registry
        self.audit_logger = audit_logger
        self.metrics = metrics or MetricsCollector()

    def _generate_audit_ref(self, request_id: str) -> str:
        """Creates a unique, traceable audit reference for a request."""
        return f"audit-{request_id}-{uuid.uuid4().hex[:8]}"

    def _log_audit(
        self,
        request_id: str,
        tenant_id: str,
        route_id: str,
        event: str,
        status: str,
        latency_ms: float = 0.0,
        **kwargs,
    ):
        """Safely logs an audit event; no-ops if no audit logger is configured."""
        if self.audit_logger:
            self.audit_logger.log_event(
                request_id, tenant_id, route_id, event, status, latency_ms, **kwargs
            )

    # ── Inference Handler ──────────────────────────────────────

    async def handle_infer(self, req: InferRequest) -> InferResponse:
        """
        Core inference pipeline:
          1. Validate tenant access to the requested route
          2. Execute inference through the HotSwapRuntime
          3. Record metrics and audit events
          4. Return structured response with audit reference
        """
        audit_ref = self._generate_audit_ref(req.request_id)
        self.metrics.record_request()
        t_start = time.perf_counter()

        # ── Step 1: Route existence and status check ──
        route_data = self.registry.get_route(req.route_id)
        if route_data is None:
            self.metrics.record_rejection(reason="route_not_found")
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_rejected", "failure", failure_reason="route_not_found",
            )
            raise HTTPException(status_code=404, detail=f"Route '{req.route_id}' not found")

        route_status = self.registry.get_route_status(req.route_id)
        if route_status in (RouteStatus.REVOKED, RouteStatus.QUARANTINED):
            self.metrics.record_rejection(reason=f"route_{route_status.value.lower()}")
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_rejected", "failure", failure_reason=f"route_{route_status.value}",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Route '{req.route_id}' is {route_status.value} and cannot be executed",
            )

        # ── Step 2: Tenant access verification ──
        route_tenant = route_data.get("tenant_id")
        if route_tenant and route_tenant != req.tenant_id:
            self.metrics.record_rejection(reason="tenant_mismatch")
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_rejected", "failure", failure_reason="tenant_mismatch",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Tenant '{req.tenant_id}' is not authorized for route '{req.route_id}'",
            )

        # ── Step 3: Execute inference ──
        try:
            from neural_scalpel.route.tenant import TenantContext

            tenant_ctx = TenantContext(req.tenant_id)

            def mock_inference():
                """
                Prototype inference function. In production, this would be
                the actual model forward pass. Here we return a deterministic
                marker that proves which route was active during execution.
                """
                return f"[route:{req.route_id}] Output for: {req.prompt[:50]}"

            output = self.runtime.infer(
                route_id=req.route_id,
                current_tenant=tenant_ctx,
                request_id=req.request_id,
                inference_func=mock_inference,
            )

            e2e_ms = (time.perf_counter() - t_start) * 1000
            swap_ms = self.runtime.last_timings.get("swap_latency", 0) * 1000
            rollback_ms = self.runtime.last_timings.get("rollback_latency", 0) * 1000

            self.metrics.record_success(
                swap_ms=swap_ms,
                rollback_ms=rollback_ms,
                e2e_ms=e2e_ms,
            )
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_completed", "success", e2e_ms,
                audit_ref=audit_ref,
            )

            return InferResponse(
                request_id=req.request_id,
                route_id=req.route_id,
                status="success",
                output=output,
                latency_ms=round(e2e_ms, 4),
                audit_ref=audit_ref,
            )

        except PermissionError as e:
            self.metrics.record_rejection(reason="permission_error")
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_rejected", "failure", failure_reason=str(e),
            )
            raise HTTPException(status_code=403, detail=str(e))

        except RuntimeError as e:
            err_msg = str(e)
            if "QUARANTINED" in err_msg:
                self.metrics.set_quarantined(True)
                self.metrics.record_rollback_failure()
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_error", "failure", failure_reason=err_msg,
            )
            raise HTTPException(status_code=500, detail=err_msg)

        except Exception as e:
            self._log_audit(
                req.request_id, req.tenant_id, req.route_id,
                "infer_error", "failure", failure_reason=str(e),
            )
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # ── Route Management Handlers ──────────────────────────────

    async def handle_register_route(self, req: RouteRegisterRequest) -> RouteRegisterResponse:
        """Registers a new route manifest into the registry."""
        try:
            route_id = self.registry.register_route(req.filepath)
            self._log_audit(
                "system", "system", route_id,
                "route_registered", "success",
            )
            return RouteRegisterResponse(route_id=route_id, status="registered")
        except Exception as e:
            self._log_audit(
                "system", "system", "unknown",
                "route_registration_failed", "failure", failure_reason=str(e),
            )
            raise HTTPException(status_code=400, detail=str(e))

    async def handle_list_routes(self) -> list:
        """Lists all registered routes with their statuses."""
        entries = []
        for route_id in self.registry.list_routes():
            route_data = self.registry.get_route(route_id)
            status = self.registry.get_route_status(route_id)
            entries.append(
                RouteListEntry(
                    route_id=route_id,
                    status=status.value if status else "UNKNOWN",
                    tenant_id=route_data.get("tenant_id") if route_data else None,
                )
            )
        return entries

    async def handle_revoke_route(self, route_id: str):
        """Revokes a route, preventing further inference."""
        try:
            self.registry.revoke_route(route_id)
            self._log_audit(
                "system", "system", route_id,
                "route_revoked", "success",
            )
            return {"route_id": route_id, "status": "revoked"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ── Metrics Handler ────────────────────────────────────────

    async def handle_metrics(self) -> MetricsResponse:
        """Returns a point-in-time snapshot of runtime health metrics."""
        snapshot = self.metrics.snapshot()
        return MetricsResponse(**snapshot)


# ── FastAPI App Factory ────────────────────────────────────────


def create_app(
    runtime,
    registry,
    audit_logger=None,
    metrics: Optional[MetricsCollector] = None,
) -> FastAPI:
    """
    Factory function that creates a fully wired FastAPI application.

    Args:
        runtime: HotSwapRuntime instance (or compatible mock)
        registry: RouteRegistry instance
        audit_logger: Optional AuditLogger for structured event logging
        metrics: Optional MetricsCollector; created automatically if not provided

    Returns:
        A configured FastAPI app ready for uvicorn.run()
    """
    server = PilotServer(
        runtime=runtime,
        registry=registry,
        audit_logger=audit_logger,
        metrics=metrics,
    )

    app = FastAPI(
        title="Neural-Scalpel Pilot API",
        description="Production Integration Prototype for route-isolated Hot-Swap inference",
        version="0.1.0-prototype",
    )

    # Store server reference for test access
    app.state.server = server

    @app.post("/v1/infer", response_model=InferResponse)
    async def infer(req: InferRequest):
        return await server.handle_infer(req)

    @app.post("/v1/routes/register", response_model=RouteRegisterResponse)
    async def register_route(req: RouteRegisterRequest):
        return await server.handle_register_route(req)

    @app.get("/v1/routes")
    async def list_routes():
        return await server.handle_list_routes()

    @app.post("/v1/routes/{route_id}/revoke")
    async def revoke_route(route_id: str):
        return await server.handle_revoke_route(route_id)

    @app.get("/v1/metrics", response_model=MetricsResponse)
    async def metrics_endpoint():
        return await server.handle_metrics()

    @app.get("/healthz")
    async def healthz():
        quarantined = server.metrics.snapshot()["runtime_quarantined"]
        # Check runtime state machine if available
        runtime_healthy = True
        if hasattr(server.runtime, "is_healthy"):
            runtime_healthy = server.runtime.is_healthy
        if hasattr(server.runtime, "can_accept_requests"):
            runtime_healthy = runtime_healthy and server.runtime.can_accept_requests

        is_healthy = runtime_healthy and not quarantined
        status_code = 200 if is_healthy else 503
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={
                "status": "ok" if is_healthy else "unhealthy",
                "quarantined": quarantined or not runtime_healthy,
                "accepting_requests": is_healthy,
            },
            status_code=status_code,
        )

    return app
