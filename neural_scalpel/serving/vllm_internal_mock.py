"""
Neural-Scalpel: vLLM Internal Integration Mock (Step 4B)

This module mocks the internal architecture of vLLM (Scheduler, BlockAllocator, ModelRunner)
to prove that Route-Homogeneous Batching and KV Cache Tagging can be safely implemented.
"""

from typing import List, Dict, Optional, Tuple
import time

class UnsafeMixedRouteBatchError(Exception):
    """Raised when the ModelRunner receives a batch with multiple active routes."""
    pass

class RouteLeakageError(Exception):
    """Raised when KV cache is accessed across different routes."""
    pass

# ── Mock Request ──────────────────────────────────────────────────────────

class MockRequest:
    def __init__(self, request_id: str, prompt: str, route_id: Optional[str] = None):
        self.request_id = request_id
        self.prompt = prompt
        self.route_id = route_id
        self.status = "PENDING"
        self.kv_blocks: List[int] = []

# ── Mock KV Cache Allocator ───────────────────────────────────────────────

class RouteTaggedKVBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.route_id: Optional[str] = None
        self.ref_count = 0

class RouteAwareBlockAllocator:
    def __init__(self, num_blocks: int = 100):
        self.blocks = {i: RouteTaggedKVBlock(i) for i in range(num_blocks)}
        self.free_blocks = list(range(num_blocks))

    def allocate(self, route_id: Optional[str]) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of KV Cache blocks")
        block_id = self.free_blocks.pop(0)
        block = self.blocks[block_id]
        block.route_id = route_id
        block.ref_count = 1
        return block_id

    def verify_access(self, block_id: int, request_route_id: Optional[str]):
        block = self.blocks[block_id]
        if block.route_id != request_route_id:
            raise RouteLeakageError(
                f"KV Cache Leakage! Request route '{request_route_id}' attempted to "
                f"access block {block_id} tagged with route '{block.route_id}'"
            )

# ── Mock Scheduler ────────────────────────────────────────────────────────

class RouteAwareScheduler:
    def __init__(self, max_batch_size: int = 4):
        self.pending_requests: List[MockRequest] = []
        self.max_batch_size = max_batch_size

    def add_request(self, request: MockRequest):
        self.pending_requests.append(request)

    def schedule_next_batch(self) -> List[MockRequest]:
        """
        Strategy A: Route-Homogeneous Batching
        Groups requests only if they share the exact same route_id.
        """
        if not self.pending_requests:
            return []

        # Group by route_id
        route_groups: Dict[Optional[str], List[MockRequest]] = {}
        for req in self.pending_requests:
            if req.route_id not in route_groups:
                route_groups[req.route_id] = []
            route_groups[req.route_id].append(req)

        # Pick the largest group to maximize throughput, or just the first
        # For simplicity, pick the first route_id we see
        target_route = self.pending_requests[0].route_id
        group = route_groups[target_route]

        batch = group[:self.max_batch_size]
        
        # Remove scheduled requests from pending
        for req in batch:
            self.pending_requests.remove(req)

        return batch

# ── Mock Model Runner ─────────────────────────────────────────────────────

class RouteAwareModelRunner:
    def __init__(self, allocator: RouteAwareBlockAllocator):
        self.allocator = allocator
        self.metrics = {
            "route_leakage_events": 0,
            "rollback_failures": 0,
            "swaps_performed": 0
        }
        self.active_weights_route: Optional[str] = None
        self._is_quarantined = False

    def _swap_weights(self, route_id: Optional[str]):
        if self.active_weights_route == route_id:
            return
        # Simulated HotSwapRuntime.swap()
        time.sleep(0.01) # Simulate overhead
        self.active_weights_route = route_id
        self.metrics["swaps_performed"] += 1

    def _rollback_weights(self):
        # Simulated HotSwapRuntime.rollback()
        time.sleep(0.005) # Simulate overhead
        self.active_weights_route = None

    def execute_forward_pass(self, batch: List[MockRequest]):
        if self._is_quarantined:
            raise RuntimeError("Model is in QUARANTINE due to previous failure.")
            
        if not batch:
            return

        # 1. Safety Check: Ensure batch is homogeneous
        target_route = batch[0].route_id
        for req in batch:
            if req.route_id != target_route:
                raise UnsafeMixedRouteBatchError(
                    f"Mixed route batch detected! Expected {target_route}, found {req.route_id}"
                )

        # 2. Pre-forward Swap
        self._swap_weights(target_route)

        try:
            # 3. Verify KV Cache access
            for req in batch:
                for block_id in req.kv_blocks:
                    try:
                        self.allocator.verify_access(block_id, req.route_id)
                    except RouteLeakageError:
                        self.metrics["route_leakage_events"] += 1
                        raise

                # Simulate allocating a new block for the generated token
                new_block = self.allocator.allocate(req.route_id)
                req.kv_blocks.append(new_block)
                req.status = "COMPLETED"

        except Exception as e:
            # Simulated failure during forward
            try:
                self._rollback_weights()
            except Exception:
                self.metrics["rollback_failures"] += 1
                self._is_quarantined = True
            raise e

        # 4. Post-forward Rollback (Optional depending on strategy, but safe)
        self._rollback_weights()
