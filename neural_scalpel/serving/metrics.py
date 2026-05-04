"""
Neural-Scalpel Serving Metrics Collector

Thread-safe, lock-free metrics collection for the Pilot API.
Tracks request counts, latencies, route leakage, and rollback failures
to support the /v1/metrics endpoint and production readiness assessment.

Design decisions:
  - Uses collections.deque for bounded latency windows (O(1) append)
  - Percentile calculation is done on-demand from the window, not streaming
  - All counters are monotonically increasing; resets require explicit action
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# Maximum number of latency samples retained for percentile calculations.
# This bounds memory usage while providing sufficient statistical resolution.
_LATENCY_WINDOW_SIZE = 10_000


@dataclass
class _LatencyWindow:
    """A bounded circular buffer of latency samples (milliseconds)."""
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW_SIZE))

    def record(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """
        Returns the p-th percentile (0-100) of recorded samples.
        Returns 0.0 if no samples have been recorded.
        """
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]


class MetricsCollector:
    """
    Central observability sink for the Neural-Scalpel Serving layer.

    All mutating methods are thread-safe via a single lock.
    Read methods acquire the same lock to ensure a consistent snapshot.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._requests_total: int = 0
        self._requests_success: int = 0
        self._requests_rejected: int = 0
        self._runtime_quarantined: bool = False
        self._route_leakage_count: int = 0
        self._rollback_failure_count: int = 0
        self._swap_latency = _LatencyWindow()
        self._rollback_latency = _LatencyWindow()
        self._e2e_latency = _LatencyWindow()

    # ── Recording ──────────────────────────────────────────────

    def record_request(self) -> None:
        with self._lock:
            self._requests_total += 1

    def record_success(self, swap_ms: float = 0.0, rollback_ms: float = 0.0, e2e_ms: float = 0.0) -> None:
        with self._lock:
            self._requests_success += 1
            if swap_ms > 0:
                self._swap_latency.record(swap_ms)
            if rollback_ms > 0:
                self._rollback_latency.record(rollback_ms)
            if e2e_ms > 0:
                self._e2e_latency.record(e2e_ms)

    def record_rejection(self, reason: str = "") -> None:
        with self._lock:
            self._requests_rejected += 1

    def record_route_leakage(self) -> None:
        with self._lock:
            self._route_leakage_count += 1

    def record_rollback_failure(self) -> None:
        with self._lock:
            self._rollback_failure_count += 1

    def set_quarantined(self, quarantined: bool = True) -> None:
        with self._lock:
            self._runtime_quarantined = quarantined

    # ── Snapshot ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Returns a consistent point-in-time snapshot of all metrics.
        The returned dict matches the MetricsResponse schema exactly.
        """
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "requests_success": self._requests_success,
                "requests_rejected": self._requests_rejected,
                "runtime_quarantined": self._runtime_quarantined,
                "swap_latency_p99_ms": round(self._swap_latency.percentile(99), 4),
                "rollback_latency_p99_ms": round(self._rollback_latency.percentile(99), 4),
                "route_leakage_count": self._route_leakage_count,
                "rollback_failure_count": self._rollback_failure_count,
            }
