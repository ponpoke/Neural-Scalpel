"""
Route-Aware Scheduler Patch for vLLM V1.

Phase 7F-2 implementation:
- Active route-homogeneous scheduling via "Shelving" strategy.
- Temporarily moves non-matching requests to avoid mixed-route batches.
- Preserves vLLM V1 internal structures and expectations (no None returns).
"""

def inject_route_aware_scheduler():
    import vllm.v1.core.sched.scheduler as vllm_scheduler
    from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics

    if getattr(vllm_scheduler.Scheduler, "_scalpel_scheduler_patched", False):
        return

    original_schedule = vllm_scheduler.Scheduler.schedule
    original_init = vllm_scheduler.Scheduler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.active_route_id = None

    def _unwrap_request(obj):
        return getattr(obj, "request", obj)

    def _get_request_route_id(req_or_state) -> str:
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        req = _unwrap_request(req_or_state)
        route_id = getattr(req, "route_id", None)
        if route_id:
            return route_id
        request_id = getattr(req, "request_id", getattr(req, "req_id", None))
        if request_id is not None:
            return RoutePluginMetrics.get_route_for_request_id(request_id)
        if isinstance(req, str):
            return RoutePluginMetrics.get_route_for_request_id(req)
        return "__base__"

    def _iter_scheduled_reqs(output):
        candidate_attrs = ("scheduled_new_reqs", "scheduled_resumed_reqs", "scheduled_running_reqs")
        for attr in candidate_attrs:
            items = getattr(output, attr, None)
            if not items: continue
            iterable = items.values() if isinstance(items, dict) else items
            for item in iterable: yield item

    def patched_schedule(self):
        # 1. Determine target route for this batch
        target_route = None
        if hasattr(self, "running") and self.running:
            target_route = _get_request_route_id(self.running[0])
        elif hasattr(self, "waiting") and self.waiting:
            req = self.waiting.peek_request()
            if req:
                target_route = _get_request_route_id(req)
        
        target_route = target_route or "__base__"
        
        # 2. Shelve non-matching requests from waiting queues
        # We use a list to store them temporarily.
        shelved_waiting = []
        shelved_skipped = []
        
        # Process self.waiting
        while self.waiting:
            req = self.waiting.peek_request()
            if _get_request_route_id(req) == target_route:
                break # Keep this and everything behind it might be fine, 
                      # but vLLM might pick deeper ones. To be safe, 
                      # we should ideally shelve ALL non-matching.
            shelved_waiting.append(self.waiting.pop_request())
            
        # Process self.skipped_waiting
        while self.skipped_waiting:
            req = self.skipped_waiting.peek_request()
            if _get_request_route_id(req) == target_route:
                break
            shelved_skipped.append(self.skipped_waiting.pop_request())

        # 3. Execute original scheduler
        # It now only sees requests matching target_route at the front of the queues.
        output = original_schedule(self)

        # 4. Restore shelved requests
        # Use prepend_request to put them back at the front in original order (reverse pop)
        for req in reversed(shelved_waiting):
            self.waiting.prepend_request(req)
        for req in reversed(shelved_skipped):
            self.skipped_waiting.prepend_request(req)

        # 5. Post-validation (Fail-Close is still our safety net)
        routes = [_get_request_route_id(r) for r in _iter_scheduled_reqs(output)]
        unique_routes = set(routes)
        if len(unique_routes) > 1:
            RoutePluginMetrics.record_violation()
            raise RuntimeError(f"CRITICAL: Unsafe mixed-route batch detected despite shelving! routes={sorted(unique_routes)}")

        # 6. Update active route state
        active_route = next(iter(unique_routes)) if unique_routes else target_route
        self.active_route_id = active_route
        RoutePluginMetrics.set_active_route(active_route)
        try:
            setattr(output, "active_route_id", active_route)
        except Exception: pass

        return output

    vllm_scheduler.Scheduler.__init__ = patched_init
    vllm_scheduler.Scheduler.schedule = patched_schedule
    vllm_scheduler.Scheduler._scalpel_scheduler_patched = True
