"""
Neural-Scalpel Prometheus Metrics

Production-grade Prometheus metrics export for the Neural-Scalpel runtime.
Implements the complete metric set required by P5 of the production readiness roadmap.

Metrics exported:
  scalpel_requests_total{tenant,route,status}
  scalpel_forward_total{route}
  scalpel_swap_total{route}
  scalpel_rollback_total{route}
  scalpel_rollback_failure_total{route}
  scalpel_route_violation_total
  scalpel_quarantine_total{scope,reason}
  scalpel_swap_latency_seconds
  scalpel_rollback_latency_seconds
  scalpel_payload_load_latency_seconds
  scalpel_scheduler_shelving_total
  scalpel_active_route
  scalpel_worker_health
"""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class PrometheusMetrics:
    """
    Centralized Prometheus metric registry for Neural-Scalpel.

    If prometheus_client is not installed, all methods become no-ops,
    allowing the runtime to function without the observability dependency.
    """

    def __init__(self, registry: Optional[object] = None):
        if not HAS_PROMETHEUS:
            self._enabled = False
            return

        self._enabled = True
        self._registry = registry or CollectorRegistry()

        # ── Request Counters ───────────────────────────────────
        self.requests_total = Counter(
            "scalpel_requests_total",
            "Total inference requests",
            ["tenant", "route", "status"],
            registry=self._registry,
        )
        self.forward_total = Counter(
            "scalpel_forward_total",
            "Total forward passes executed",
            ["route"],
            registry=self._registry,
        )

        # ── Swap / Rollback Counters ───────────────────────────
        self.swap_total = Counter(
            "scalpel_swap_total",
            "Total weight swaps performed",
            ["route"],
            registry=self._registry,
        )
        self.rollback_total = Counter(
            "scalpel_rollback_total",
            "Total rollbacks performed",
            ["route"],
            registry=self._registry,
        )
        self.rollback_failure_total = Counter(
            "scalpel_rollback_failure_total",
            "Total rollback failures (critical)",
            ["route"],
            registry=self._registry,
        )

        # ── Safety Counters ────────────────────────────────────
        self.route_violation_total = Counter(
            "scalpel_route_violation_total",
            "Total route isolation violations detected",
            registry=self._registry,
        )
        self.quarantine_total = Counter(
            "scalpel_quarantine_total",
            "Total quarantine events",
            ["scope", "reason"],
            registry=self._registry,
        )
        self.scheduler_shelving_total = Counter(
            "scalpel_scheduler_shelving_total",
            "Total requests shelved by scheduler for route isolation",
            registry=self._registry,
        )

        # ── Latency Histograms ─────────────────────────────────
        latency_buckets = (
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        )
        self.swap_latency = Histogram(
            "scalpel_swap_latency_seconds",
            "Latency of weight swap operations",
            buckets=latency_buckets,
            registry=self._registry,
        )
        self.rollback_latency = Histogram(
            "scalpel_rollback_latency_seconds",
            "Latency of rollback operations",
            buckets=latency_buckets,
            registry=self._registry,
        )
        self.payload_load_latency = Histogram(
            "scalpel_payload_load_latency_seconds",
            "Latency of payload file loading",
            buckets=latency_buckets,
            registry=self._registry,
        )

        # ── Gauges ─────────────────────────────────────────────
        self.active_route = Gauge(
            "scalpel_active_route",
            "Currently active route in VRAM",
            ["route"],
            registry=self._registry,
        )
        self.worker_health = Gauge(
            "scalpel_worker_health",
            "Worker health status (1=healthy, 0=quarantined)",
            registry=self._registry,
        )
        self.worker_health.set(1)

    # ── Recording Methods ──────────────────────────────────────

    def record_request(self, tenant: str, route: str, status: str) -> None:
        if self._enabled:
            self.requests_total.labels(tenant=tenant, route=route, status=status).inc()

    def record_forward(self, route: str) -> None:
        if self._enabled:
            self.forward_total.labels(route=route).inc()

    def record_swap(self, route: str, latency_seconds: float) -> None:
        if self._enabled:
            self.swap_total.labels(route=route).inc()
            self.swap_latency.observe(latency_seconds)

    def record_rollback(self, route: str, latency_seconds: float) -> None:
        if self._enabled:
            self.rollback_total.labels(route=route).inc()
            self.rollback_latency.observe(latency_seconds)

    def record_rollback_failure(self, route: str) -> None:
        if self._enabled:
            self.rollback_failure_total.labels(route=route).inc()

    def record_route_violation(self) -> None:
        if self._enabled:
            self.route_violation_total.inc()

    def record_quarantine(self, scope: str, reason: str) -> None:
        if self._enabled:
            self.quarantine_total.labels(scope=scope, reason=reason).inc()

    def record_payload_load(self, latency_seconds: float) -> None:
        if self._enabled:
            self.payload_load_latency.observe(latency_seconds)

    def record_scheduler_shelving(self) -> None:
        if self._enabled:
            self.scheduler_shelving_total.inc()

    def set_active_route(self, route: str) -> None:
        if self._enabled:
            self.active_route.labels(route=route).set(1)

    def set_worker_unhealthy(self) -> None:
        if self._enabled:
            self.worker_health.set(0)

    # ── Export ─────────────────────────────────────────────────

    def export(self) -> bytes:
        """Returns Prometheus exposition format bytes."""
        if not self._enabled:
            return b""
        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST
