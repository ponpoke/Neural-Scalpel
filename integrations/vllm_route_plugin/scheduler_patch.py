"""
Route-Aware Scheduler Patch for vLLM V1.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict

def inject_route_aware_scheduler():
    """
    Monkey patch vLLM's Scheduler to enforce Route-Homogeneous Batching.
    """
    import vllm.v1.core.sched.scheduler as vllm_scheduler
    from vllm.v1.request import Request
    
    # Store original methods
    original_schedule = vllm_scheduler.Scheduler.schedule
    original_init = vllm_scheduler.Scheduler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.active_route_id = None
        self.route_window_tokens_left = 0
        self.MAX_ROUTE_WINDOW_TOKENS = 8192  # Configurable fairness threshold

    def _unwrap_request(obj):
        """
        vLLM version compatibility:
        - Some queues contain Request directly.
        - Some queues contain RequestState-like objects with .request.
        """
        return getattr(obj, "request", obj)

    def _get_request_route_id(req_or_state) -> str:
        req = _unwrap_request(req_or_state)
        return getattr(req, "route_id", "__base__")

    def patched_schedule(self):
        """
        Custom schedule() that enforces Route-Homogeneous Batching.
        1. All running requests must share the same route_id.
        2. We can only schedule waiting requests that match the active_route_id.
        3. If running queue is empty, pick the next route_id using round-robin.
        """
        # If there are running requests, the active route is strictly their route
        if self.running:
            self.active_route_id = _get_request_route_id(self.running[0])
            # Ensure all running are indeed the active route (Fail-Close)
            for running_req in self.running:
                req_route = _get_request_route_id(running_req)
                if req_route != self.active_route_id:
                    raise RuntimeError(f"Mixed routes in running queue! Expected {self.active_route_id}, got {req_route}")
        else:
            # If nothing is running, we can pick a new active route
            # For simplicity, we pick the route of the oldest waiting request
            if self.waiting:
                self.active_route_id = _get_request_route_id(self.waiting[0])
                self.route_window_tokens_left = self.MAX_ROUTE_WINDOW_TOKENS
            else:
                self.active_route_id = None

        if self.active_route_id is None:
            return original_schedule(self)

        # Separate waiting requests into matching and non-matching routes
        matching_waiting = []
        non_matching_waiting = []
        
        for req_state in self.waiting:
            req_route = _get_request_route_id(req_state)
            if req_route == self.active_route_id:
                matching_waiting.append(req_state)
            else:
                non_matching_waiting.append(req_state)

        # Temporarily hide non-matching requests from the original scheduler
        self.waiting = matching_waiting
        
        try:
            # Call the original scheduler with only matching requests
            output = original_schedule(self)
            # Record metrics / Validate homogeneity
            if hasattr(output, "scheduled_new_reqs") and output.scheduled_new_reqs:
                from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
                for scheduled_req in output.scheduled_new_reqs:
                    req_route = _get_request_route_id(scheduled_req)
                    if req_route != self.active_route_id:
                        RoutePluginMetrics.record_violation()
                        raise RuntimeError(f"CRITICAL: Unsafe mixed-route batch detected during scheduling! "
                                         f"Expected {self.active_route_id}, got {req_route}")
                        
            return output
        finally:
            # Restore the non-matching requests back to the waiting queue
            self.waiting.extend(non_matching_waiting)
            
            # Sort waiting queue by priority/arrival time again since we messed with the order
            def _sort_key(req_or_state):
                req = _unwrap_request(req_or_state)
                return (
                    getattr(req, "priority", 0),
                    getattr(req, "arrival_time", 0),
                    getattr(req, "request_id", ""),
                )
            self.waiting.sort(key=_sort_key)
            
        return output

    # Apply patches
    vllm_scheduler.Scheduler.__init__ = patched_init
    vllm_scheduler.Scheduler.schedule = patched_schedule

