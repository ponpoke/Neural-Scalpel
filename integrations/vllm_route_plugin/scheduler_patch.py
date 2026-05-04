"""
Route-Aware Scheduler Patch for vLLM V1.

Phase 7E safe version:
- Does NOT replace self.waiting / self.running queues (avoiding 'list' vs 'RequestQueue' errors).
- Lets vLLM's native scheduler run.
- Validates the emitted SchedulerOutput.
- Fails closed if a mixed-route batch is detected.
"""

def inject_route_aware_scheduler():
    import vllm.v1.core.sched.scheduler as vllm_scheduler

    if getattr(vllm_scheduler.Scheduler, "_scalpel_scheduler_patched", False):
        return

    original_schedule = vllm_scheduler.Scheduler.schedule
    original_init = vllm_scheduler.Scheduler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.active_route_id = None
        self.MAX_ROUTE_WINDOW_TOKENS = 8192

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

    def _iter_scheduled_reqs(output):
        """
        vLLM version compatibility:
        SchedulerOutput may expose different fields depending on version.
        """
        candidate_attrs = (
            "scheduled_new_reqs",
            "scheduled_resumed_reqs",
            "scheduled_running_reqs",
        )

        for attr in candidate_attrs:
            items = getattr(output, attr, None)
            if not items:
                continue

            # Some attrs may be dict-like, some list-like.
            if isinstance(items, dict):
                iterable = items.values()
            else:
                iterable = items

            for item in iterable:
                yield item

    def patched_schedule(self):
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics

        # Let the original scheduler decide what to batch
        output = original_schedule(self)

        # Validate the batch for route homogeneity
        routes = []
        for scheduled_req in _iter_scheduled_reqs(output):
            routes.append(_get_request_route_id(scheduled_req))

        unique_routes = set(routes)

        if len(unique_routes) > 1:
            RoutePluginMetrics.record_violation()
            raise RuntimeError(
                f"CRITICAL: Unsafe mixed-route batch detected by Neural-Scalpel. "
                f"routes={sorted(unique_routes)}"
            )

        # Determine the active route for this forward pass
        if len(unique_routes) == 1:
            active_route = next(iter(unique_routes))
        else:
            # Empty batch or only base
            active_route = "__base__"

        self.active_route_id = active_route
        
        # Set the active route globally for the ModelRunner hook to pick up
        RoutePluginMetrics.set_active_route(active_route)

        # Attach route to SchedulerOutput so ModelRunner can see it (if it uses SchedulerOutput)
        try:
            setattr(output, "active_route_id", active_route)
        except Exception:
            pass

        return output

    # Apply patches
    vllm_scheduler.Scheduler.__init__ = patched_init
    vllm_scheduler.Scheduler.schedule = patched_schedule
    vllm_scheduler.Scheduler._scalpel_scheduler_patched = True
