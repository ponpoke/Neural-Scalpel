"""
Route-Aware Scheduler Patch for vLLM V1.

Phase 7F-2 implementation (Robust Shelving Strategy):
- Scans the entire waiting queue to extract non-matching requests.
- Uses try/finally to ensure requests are ALWAYS restored to the queue.
- API-compatible with vLLM V1 RequestQueue (pop_request/prepend_request).
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
        return "__base__"

    def _iter_scheduled_reqs(output):
        candidate_attrs = ("scheduled_new_reqs", "scheduled_resumed_reqs", "scheduled_running_reqs")
        for attr in candidate_attrs:
            items = getattr(output, attr, None)
            if not items: continue
            iterable = items.values() if isinstance(items, dict) else items
            for item in iterable: yield item

    def _shelve_queue(queue, target_route):
        """
        Pops ALL requests from the queue and puts back only those matching target_route.
        Returns the non-matching requests for later restoration.
        """
        matching = []
        non_matching = []
        while queue:
            req = queue.pop_request()
            if _get_request_route_id(req) == target_route:
                matching.append(req)
            else:
                non_matching.append(req)
        
        # Put matching back in original order (add_request usually appends for FCFS)
        for req in matching:
            queue.add_request(req)
            
        return non_matching

    def _restore_queue(queue, shelved):
        """
        Restores shelved requests to the front of the queue in their original order.
        """
        for req in reversed(shelved):
            queue.prepend_request(req)

    def patched_schedule(self):
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics

        # 1. Determine target route for this batch
        target_route = None
        if hasattr(self, "running") and self.running:
            target_route = _get_request_route_id(self.running[0])
        elif hasattr(self, "waiting") and self.waiting:
            try:
                req = self.waiting.peek_request()
                target_route = _get_request_route_id(req)
            except IndexError:
                pass
        
        target_route = target_route or "__base__"
        
        # 2. Shelve non-matching requests from waiting queues
        shelved_waiting = []
        shelved_skipped = []
        
        if hasattr(self, "waiting"):
            shelved_waiting = _shelve_queue(self.waiting, target_route)
        if hasattr(self, "skipped_waiting"):
            shelved_skipped = _shelve_queue(self.skipped_waiting, target_route)

        output = None
        try:
            # 3. Execute original scheduler with homogeneous queues
            output = original_schedule(self)
        finally:
            # 4. CRITICAL: Restore shelved requests even if schedule() fails
            if hasattr(self, "waiting"):
                _restore_queue(self.waiting, shelved_waiting)
            if hasattr(self, "skipped_waiting"):
                _restore_queue(self.skipped_waiting, shelved_skipped)

        if output is None:
            # Should only happen if original_schedule failed and finally block restored reqs
            return None

        # 5. Post-validation (Safety Net)
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
