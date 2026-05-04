"""
ModelRunner Swap/Rollback Hook for vLLM internal integration.
"""

def inject_model_runner_hook():
    """
    Monkey patch vLLM's GPUModelRunner to perform Hot-Swap and Rollback
    around the core _model_forward pass.
    """
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    
    if getattr(gpu_model_runner.GPUModelRunner, "_scalpel_forward_patched", False):
        return

    # In vLLM V1, _model_forward is the specific helper that calls the model.
    # It is cleaner to wrap this than the high-level execute_model.
    original_model_forward = gpu_model_runner.GPUModelRunner._model_forward

    def patched_model_forward(self, *args, **kwargs):
        """
        Intercepts _model_forward to apply the route payload before forward pass,
        and roll it back afterwards.
        """
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        
        # Determine active route from global metrics (set by Scheduler patch)
        active_route = RoutePluginMetrics.get_active_route()
        
        # 2. Swap before forward pass
        is_swapped = False
        if active_route != "__base__":
            # Placeholder for HotSwapRuntime.atomic_swap(active_route)
            is_swapped = True
            RoutePluginMetrics.record_swap()

        try:
            # 3. Execute original model forward
            return original_model_forward(self, *args, **kwargs)
            
        except Exception as e:
            # Failure Handling: Ensure rollback even on crash
            raise e
            
        finally:
            # 4. Rollback
            if is_swapped:
                # Placeholder for HotSwapRuntime.atomic_rollback()
                RoutePluginMetrics.record_rollback()

    gpu_model_runner.GPUModelRunner._model_forward = patched_model_forward
    gpu_model_runner.GPUModelRunner._scalpel_forward_patched = True
