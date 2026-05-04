"""
ModelRunner Swap/Rollback Hook for vLLM internal integration.
"""

def inject_model_runner_hook():
    """
    Monkey patch vLLM's GPUModelRunner to perform Hot-Swap and Rollback
    around the core execute_model pass.
    """
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    
    if getattr(gpu_model_runner.GPUModelRunner, "_scalpel_model_runner_patched", False):
        return

    original_execute_model = gpu_model_runner.GPUModelRunner.execute_model

    def _unwrap_request(obj):
        return getattr(obj, "request", obj)

    def _get_request_route_id(req_or_state) -> str:
        req = _unwrap_request(req_or_state)
        return getattr(req, "route_id", "__base__")

    def _iter_scheduled_reqs(output):
        for attr in ("scheduled_new_reqs", "scheduled_resumed_reqs", "scheduled_running_reqs"):
            items = getattr(output, attr, None)
            if not items:
                continue

            if isinstance(items, dict):
                iterable = items.values()
            else:
                iterable = items

            for item in iterable:
                yield item

    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        """
        Intercepts execute_model to apply the route payload before forward pass,
        and roll it back afterwards.
        """
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        
        # 1. Determine active route and validate homogeneity
        routes = [_get_request_route_id(req) for req in _iter_scheduled_reqs(scheduler_output)]
        unique_routes = set(routes)

        if len(unique_routes) > 1:
            RoutePluginMetrics.record_violation()
            raise RuntimeError(
                f"CRITICAL: Mixed-route batch leaked into ModelRunner. "
                f"routes={sorted(unique_routes)}"
            )

        # Use the route ID attached by the scheduler if available
        active_route = getattr(scheduler_output, "active_route_id", None)
        if active_route is None:
            if len(unique_routes) == 1:
                active_route = next(iter(unique_routes))
            else:
                active_route = "__base__"
        
        # 2. Swap before forward pass
        is_swapped = False
        if active_route != "__base__":
            # Placeholder for HotSwapRuntime.atomic_swap(active_route)
            is_swapped = True
            RoutePluginMetrics.record_swap()

        try:
            # 3. Execute original forward pass
            output = original_execute_model(self, scheduler_output, *args, **kwargs)
            return output
            
        except Exception as e:
            # Failure Handling: Ensure rollback even on crash
            raise e
            
        finally:
            # 4. Rollback
            if is_swapped:
                # Placeholder for HotSwapRuntime.atomic_rollback()
                RoutePluginMetrics.record_rollback()

    gpu_model_runner.GPUModelRunner.execute_model = patched_execute_model
    gpu_model_runner.GPUModelRunner._scalpel_model_runner_patched = True
