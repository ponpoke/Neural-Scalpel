"""
ModelRunner Swap/Rollback Hook for vLLM internal integration.
Phase 7G: Real Payload Integration with Robust Layer Verification.
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
    original_model_forward = gpu_model_runner.GPUModelRunner._model_forward

    def patched_model_forward(self, *args, **kwargs):
        """
        Intercepts _model_forward to apply the route payload before forward pass,
        and roll it back afterwards.
        """
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        from integrations.vllm_route_plugin.runtime_context import get_vllm_runtime
        
        # Track that _model_forward was actually entered
        RoutePluginMetrics.record_forward()
        
        # 1. Determine active route
        route_id = RoutePluginMetrics.get_active_route()
        
        # 2. Get Runtime instance (lazily initialized with self.model)
        runtime = get_vllm_runtime(self.model)
        
        # Debug: Print projection layer names once to verify structure
        if not hasattr(self, "_scalpel_names_logged"):
            names = [name for name, _ in self.model.named_parameters()]
            print("[Neural-Scalpel] q_proj/v_proj candidates:")
            print([n for n in names if "q_proj" in n or "v_proj" in n][:20])
            self._scalpel_names_logged = True
        
        is_swapped = False
        if route_id != "__base__":
            # Look up route definition in registry (with fallback to direct dict access)
            route_data = runtime.registry.get_route(route_id)
            if route_data is None and hasattr(runtime.registry, "routes"):
                route_data = runtime.registry.routes.get(route_id)
                
            if route_data:
                try:
                    # Perform Atomic Swap
                    runtime.capture_and_verify(route_data)
                    runtime.swap(route_data)
                    is_swapped = True
                    RoutePluginMetrics.record_swap()
                except Exception as e:
                    print(f"[Neural-Scalpel] Swap failed for {route_id}: {e}")
                    # In production this would trigger Fail-Close. 
                    # For Phase 7G we allow the crash to see the stack trace.
                    raise RuntimeError(f"Neural-Scalpel Swap Failure: {e}")
            else:
                raise RuntimeError(f"Route not found in vLLM runtime registry: {route_id}")

        try:
            # 3. Execute original model forward
            return original_model_forward(self, *args, **kwargs)
            
        finally:
            # 4. Atomic Rollback
            if is_swapped:
                try:
                    runtime.rollback()
                    RoutePluginMetrics.record_rollback()
                except Exception as e:
                    print(f"[Neural-Scalpel] CRITICAL: Rollback failed! {e}")
                    raise RuntimeError(f"Neural-Scalpel CRITICAL Rollback Failure: {e}")

    gpu_model_runner.GPUModelRunner._model_forward = patched_model_forward
    gpu_model_runner.GPUModelRunner._scalpel_forward_patched = True
