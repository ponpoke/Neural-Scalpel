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
        
        Hardened prototype path for Production Candidate evaluation:
          - Pre-swap validation using PayloadValidator
          - Failure policy enforcement (fail-close / quarantine)
          - Formal state transitions
        """
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        from integrations.vllm_route_plugin.runtime_context import get_vllm_runtime
        from neural_scalpel.route.tenant import TenantContext
        
        # Track that _model_forward was actually entered
        RoutePluginMetrics.record_forward()
        
        # 1. Determine active route requested by the scheduler/sampler
        route_id = RoutePluginMetrics.get_active_route()
        
        # 2. Get Runtime instance (lazily initialized with self.model)
        runtime = get_vllm_runtime(self.model)
        
        try:
            # Phase 5-C: Route-Window Optimization
            # ensure_route only swaps/rollbacks if the route_id has changed.
            # This keeps the weights persistent across decoding steps.
            tenant = TenantContext(tenant_id="vllm-tenant")
            request_id = f"vllm-req-{id(args)}"
            
            runtime.ensure_route(route_id, tenant, request_id)
            
            # 3. Proceed with inference (using persistent weights if swapped)
            return original_model_forward(self, *args, **kwargs)
            
        except Exception as e:
            # Emergency rollback on error to keep the worker safe
            print(f"[Neural-Scalpel ERROR] Lifecycle exception: {e}")
            if hasattr(runtime, "clear_active_route"):
                runtime.clear_active_route()
            raise



    gpu_model_runner.GPUModelRunner._model_forward = patched_model_forward
    gpu_model_runner.GPUModelRunner._scalpel_forward_patched = True

