import os
import logging
from typing import Optional, Any, Callable
from .engine import ServingMode, ServingEngine
from .mode_selector import select_from_environment
from .backend_registry import BackendRegistry
from .proxy_engine import ProxyServingEngine
from .startup_selftest import run_self_tests

logger = logging.getLogger(__name__)

class InternalPluginEngine(ServingEngine):
    """
    Adapter that wraps the internal HotSwapRuntime to match the ServingEngine interface.
    """
    def __init__(self, runtime: Any):
        self.runtime = runtime

    async def infer(self, req: Any, tenant_ctx: Any, audit_ref: str) -> Any:
        """
        Executes inference via the internal weight-swapping runtime.
        """
        # Note: In a real implementation, we would extract the inference function
        # from the vLLM model executor.
        def _inference_placeholder():
            return f"Internal output for: {getattr(req, 'prompt', '')[:20]}..."

        return self.runtime.infer(
            route_id=getattr(req, "route_id"),
            current_tenant=tenant_ctx,
            request_id=getattr(req, "request_id", audit_ref),
            inference_func=_inference_placeholder
        )

    def get_health(self) -> dict:
        is_healthy = True
        if hasattr(self.runtime, "is_healthy"):
            is_healthy = self.runtime.is_healthy
            
        return {
            "engine_type": "internal_plugin",
            "status": "healthy" if is_healthy else "unhealthy"
        }

class EngineFactory:
    """
    Factory for instantiating the appropriate ServingEngine based on environment
    and system compatibility.
    """
    @staticmethod
    def create_engine(
        registry_dir: str,
        payload_dir: str,
        backend_registry: Optional[BackendRegistry] = None,
        internal_runtime_factory: Optional[Callable[[], Any]] = None,
    ) -> ServingEngine:
        """
        Performs self-tests and selects the best available serving engine.
        
        Returns:
            An initialized ServingEngine (Internal or Proxy).
        
        Raises:
            RuntimeError if no valid serving mode can be established.
        """
        # 1. Run Internal Compatibility Self-Tests
        self_test_report = run_self_tests(registry_dir=registry_dir, payload_dir=payload_dir)
        internal_compatible = self_test_report.all_passed
        
        # 2. Check Proxy Configuration
        external_proxy_configured = backend_registry is not None and len(backend_registry.list_routes()) > 0
        
        # 3. Select Mode based on environment and compatibility
        selection = select_from_environment(
            internal_compatible=internal_compatible,
            external_proxy_configured=external_proxy_configured
        )
        
        logger.info(f"Serving mode selection: requested={selection.requested_mode}, "
                    f"selected={selection.selected_mode}, reason={selection.reason}")
        
        if not selection.should_start or selection.selected_mode is None:
            # Audit log would record this failure in a real system
            raise RuntimeError(f"Engine failed to start: {selection.reason}")

        # 4. Instantiate Selected Engine
        if selection.selected_mode == ServingMode.INTERNAL:
            if internal_runtime_factory is None:
                raise RuntimeError("Internal mode selected but no internal_runtime_factory provided")
            
            logger.info("Initializing InternalPluginEngine (Mode A)")
            runtime = internal_runtime_factory()
            return InternalPluginEngine(runtime)
            
        if selection.selected_mode == ServingMode.EXTERNAL_PROXY:
            logger.info("Initializing ProxyServingEngine (Mode B)")
            return ProxyServingEngine(backend_registry)
            
        raise RuntimeError(f"Unsupported serving mode selected: {selection.selected_mode}")
