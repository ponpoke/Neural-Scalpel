"""
Neural-Scalpel Serving API Schemas

Pydantic models defining the HTTP API contract for the Pilot API.
These models enforce strict type validation at the network boundary,
ensuring that malformed or incomplete requests are rejected before
reaching any runtime logic.

Design notes:
  - All string fields use `min_length=1` to reject empty values
  - `request_id` is optional on input; the server generates one if absent
  - `audit_ref` in responses links back to the structured audit log
"""

from typing import Optional, Any
from pydantic import BaseModel, Field
import uuid


class InferRequest(BaseModel):
    """A single inference request scoped to a tenant and route."""
    tenant_id: str = Field(..., min_length=1, description="Tenant identifier for access control")
    route_id: str = Field(..., min_length=1, description="Route to apply during inference")
    prompt: str = Field(..., min_length=1, description="Input text for the model")
    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}",
        description="Unique request identifier; auto-generated if omitted",
    )


class InferResponse(BaseModel):
    """Response from a completed inference request."""
    request_id: str
    route_id: str
    status: str = Field(..., description="'success' | 'error' | 'rejected'")
    output: Any = Field(default="", description="Model output text or structured response")
    latency_ms: float = Field(default=0.0, description="End-to-end request latency in ms")
    audit_ref: str = Field(default="", description="Reference to the audit log event")
    error_detail: Optional[str] = Field(default=None, description="Error message on failure")


class RouteRegisterRequest(BaseModel):
    """Request to register a new .scalpel_route manifest file."""
    filepath: str = Field(..., min_length=1, description="Absolute path to the route manifest JSON")


class RouteRegisterResponse(BaseModel):
    """Confirmation of route registration."""
    route_id: str
    status: str = Field(..., description="'registered' | 'rejected'")
    error_detail: Optional[str] = Field(default=None, description="Reason for rejection")


class RouteListEntry(BaseModel):
    """Summary of a single route in the registry."""
    route_id: str
    status: str
    tenant_id: Optional[str] = None


class MetricsResponse(BaseModel):
    """
    System health and observability snapshot.
    This is the primary interface for SRE dashboards and automated alerting.
    """
    requests_total: int = 0
    requests_success: int = 0
    requests_rejected: int = 0
    runtime_quarantined: bool = False
    swap_latency_p99_ms: float = 0.0
    rollback_latency_p99_ms: float = 0.0
    route_leakage_count: int = 0
    rollback_failure_count: int = 0
