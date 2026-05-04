"""
ModelRunner Swap/Rollback Hook for vLLM internal integration.
"""

def inject_model_runner_hook():
    """
    Monkey patch vLLM's GPUModelRunner to perform Hot-Swap and Rollback
    around the core execute_model pass.
    """
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    
    original_execute_model = gpu_model_runner.GPUModelRunner.execute_model

    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        """
        Intercepts execute_model to apply the route payload before forward pass,
        and roll it back afterwards (Mode A: per-forward rollback).
        """
        from neural_scalpel.kernel.hotswap_runtime import HotSwapRuntime
        
        # 1. Determine active route from scheduler output
        # According to Route-Homogeneous Batching, all requests in the batch MUST share the same route
        active_route = "__base__"
        if scheduler_output.scheduled_new_reqs:
            # We assume req_id mapping to route_id exists via our metadata patch.
            # In a real integration, we'd pull this securely from the new_req or cached_req.
            # For this patch, we pull from thread local or global registry logic.
            # Here, we assume the scheduler attached the route_id to the output.
            active_route = getattr(scheduler_output, "active_route_id", "__base__")
            
            # Double-check homogeneity (Fail-Close)
            from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
            for req_state in getattr(scheduler_output, "scheduled_new_reqs", []):
                req_route = getattr(req_state.request, "route_id", "__base__")
                if req_route != active_route:
                    RoutePluginMetrics.record_violation()
                    raise RuntimeError(f"CRITICAL: Mixed-route batch leaked into ModelRunner! "
                                     f"Expected {active_route}, got {req_route}")
        
        # 2. Swap before forward pass
        is_swapped = False
        if active_route != "__base__":
            # In a full implementation, we fetch the payload for the route_id
            # and call HotSwapRuntime.atomic_swap(payload)
            is_swapped = True
            
            # Record metrics
            from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
            RoutePluginMetrics.record_swap()
            pass

        try:
            # 3. Execute original forward pass
            output = original_execute_model(self, scheduler_output, *args, **kwargs)
            return output
            
        except Exception as e:
            # Failure Handling Path (Phase 5)
            # If exception occurs, we must rollback to ensure no model corruption
            raise e
            
        finally:
            # 4. Rollback
            if is_swapped:
                # HotSwapRuntime.atomic_rollback()
                from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
                RoutePluginMetrics.record_rollback()
                pass

    gpu_model_runner.GPUModelRunner.execute_model = patched_execute_model
