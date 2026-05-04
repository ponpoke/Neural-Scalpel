"""
Neural-Scalpel Route-Aware Request Scheduler

This scheduler enforces the fundamental safety invariant of Neural-Scalpel serving:
**No two different route_ids may ever coexist in a single VRAM swap cycle.**

Architecture:
  - Incoming requests are queued by route_id into independent per-route buffers
  - `fetch_next_safe_batch()` drains at most `max_batch_size` requests from a
    single route, guaranteeing homogeneity
  - A background drain loop periodically flushes ready batches to a provided
    executor callback, respecting `max_wait_time` before forcing a flush

Design invariants:
  - The scheduler NEVER creates a batch containing mixed route_ids
  - If a defaultdict race condition were to somehow create a corrupted batch,
    `validate_batch_homogeneity()` performs a secondary assertion before execution
  - All queue operations are serialized behind `asyncio.Lock`
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field


@dataclass
class ScheduledBatch:
    """
    A homogeneous micro-batch of requests sharing exactly one route_id.
    The scheduler guarantees that all entries in `requests` share the same route.
    """
    route_id: str
    requests: List[dict] = field(default_factory=list)
    futures: List[asyncio.Future] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        return len(self.requests)


class RouteAwareScheduler:
    """
    A production-prototype scheduler designed to prevent route contamination.
    It enforces that only requests belonging to the exact same `route_id`
    can be grouped together in a single VRAM swap cycle (micro-batching).
    """

    def __init__(self, max_batch_size: int = 16, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        # Per-route queues: each key maps to a ScheduledBatch accumulator
        self._queues: Dict[str, ScheduledBatch] = {}
        self._lock = asyncio.Lock()
        self._drain_task: Optional[asyncio.Task] = None
        self._executor: Optional[Callable[[ScheduledBatch], Awaitable[None]]] = None

    # ── Request Submission ─────────────────────────────────────

    async def submit_request(
        self,
        request_id: str,
        tenant_id: str,
        route_id: str,
        payload: Any,
    ) -> Any:
        """
        Enqueues a request and returns a Future that will be resolved
        when the batch containing this request is executed.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "route_id": route_id,
            "payload": payload,
        }

        async with self._lock:
            if route_id not in self._queues:
                self._queues[route_id] = ScheduledBatch(route_id=route_id)

            batch = self._queues[route_id]
            batch.requests.append(request)
            batch.futures.append(future)

        return await future

    # ── Batch Extraction ───────────────────────────────────────

    async def fetch_next_safe_batch(self) -> Optional[ScheduledBatch]:
        """
        Retrieves the next batch of homogeneous requests.
        Guarantees that NO two different route_ids are ever mixed in the returned batch.
        Returns None if all queues are empty.
        """
        async with self._lock:
            if not self._queues:
                return None

            # Prioritize the oldest waiting batch (FIFO fairness across routes)
            oldest_route = min(
                self._queues.keys(),
                key=lambda k: self._queues[k].created_at,
            )
            source = self._queues[oldest_route]

            # Extract up to max_batch_size requests into a new safe batch
            safe_batch = ScheduledBatch(route_id=source.route_id)
            safe_batch.requests = source.requests[: self.max_batch_size]
            safe_batch.futures = source.futures[: self.max_batch_size]

            # Trim the source queue
            source.requests = source.requests[self.max_batch_size :]
            source.futures = source.futures[self.max_batch_size :]

            # Clean up empty queues to prevent unbounded key growth
            if not source.requests:
                del self._queues[oldest_route]

            return safe_batch

    # ── Safety Assertion ───────────────────────────────────────

    @staticmethod
    def validate_batch_homogeneity(batch: ScheduledBatch) -> bool:
        """
        Secondary safety assertion: verifies that every request in the batch
        carries the same route_id as the batch header. This is a defense-in-depth
        check — the scheduler's queue structure should make this impossible to fail,
        but we assert it anyway because route contamination is a critical safety violation.

        Raises RuntimeError if contamination is detected.
        """
        if not batch or not batch.requests:
            return True

        route_ids = {req["route_id"] for req in batch.requests}
        if len(route_ids) > 1:
            raise RuntimeError(
                f"CRITICAL: Route contamination detected in scheduled batch. "
                f"Expected homogeneous route_id='{batch.route_id}', "
                f"but found {len(route_ids)} distinct routes: {route_ids}"
            )
        if route_ids and batch.route_id not in route_ids:
            raise RuntimeError(
                f"CRITICAL: Batch header route_id='{batch.route_id}' does not match "
                f"request route_ids: {route_ids}"
            )
        return True

    # ── Background Drain Loop ──────────────────────────────────

    def start_drain_loop(self, executor: Callable[[ScheduledBatch], Awaitable[None]]) -> None:
        """
        Starts a background asyncio task that periodically drains ready batches
        and passes them to the provided executor coroutine.
        """
        self._executor = executor
        self._drain_task = asyncio.ensure_future(self._drain_loop())

    async def _drain_loop(self) -> None:
        """
        Internal loop that flushes batches at `max_wait_time` intervals.
        Designed to be cancelled gracefully via `stop_drain_loop()`.
        """
        while True:
            await asyncio.sleep(self.max_wait_time)
            batch = await self.fetch_next_safe_batch()
            if batch is not None and self._executor is not None:
                self.validate_batch_homogeneity(batch)
                await self._executor(batch)

    async def stop_drain_loop(self) -> None:
        """Cancels the background drain task."""
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    # ── Introspection ──────────────────────────────────────────

    async def pending_count(self) -> int:
        """Returns the total number of requests waiting across all route queues."""
        async with self._lock:
            return sum(len(batch) for batch in self._queues.values())

    async def active_routes(self) -> List[str]:
        """Returns the list of route_ids that currently have pending requests."""
        async with self._lock:
            return list(self._queues.keys())
