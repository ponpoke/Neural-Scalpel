"""
Runtime context manager for Neural-Scalpel vLLM integration.
Maintains a singleton HotSwapRuntime instance for the engine.
"""
import os
from typing import Optional
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.audit import AuditLogger

_GLOBAL_RUNTIME: Optional[HotSwapRuntime] = None

def get_vllm_runtime(model) -> HotSwapRuntime:
    """
    Returns (and initializes if necessary) the global HotSwapRuntime for vLLM.
    """
    global _GLOBAL_RUNTIME
    if _GLOBAL_RUNTIME is None:
        # Initialize Registry
        # In a real production setup, this would load from a shared manifest.
        # For our Phase 7G validation, we'll initialize an empty registry 
        # and allow tests to register routes dynamically.
        registry = RouteRegistry()
        
        # Initialize Audit Logger
        audit_log_path = os.path.join(os.getcwd(), "vllm_scalpel_audit.jsonl")
        audit_logger = AuditLogger(log_path=audit_log_path)
        
        # Create Runtime
        # We use a dummy model hash for local validation.
        _GLOBAL_RUNTIME = HotSwapRuntime(
            target_model=model,
            registry=registry,
            runtime_model_hash="vllm-opt-125m-hash",
            audit_logger=audit_logger
        )
        print(f"[Neural-Scalpel] HotSwapRuntime initialized for vLLM. Audit log: {audit_log_path}")
        
    return _GLOBAL_RUNTIME
