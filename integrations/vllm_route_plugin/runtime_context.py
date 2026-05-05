"""
Runtime context manager for Neural-Scalpel vLLM integration.
Separates Registry lifecycle from Runtime initialization.
"""
import os
from typing import Optional
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.experimental.audit import AuditLogger

_GLOBAL_RUNTIME: Optional[HotSwapRuntime] = None
_GLOBAL_REGISTRY: Optional[RouteRegistry] = None
_GLOBAL_AUDIT_LOGGER: Optional[AuditLogger] = None

def get_vllm_registry() -> RouteRegistry:
    """Returns the global registry for route registration."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        # RouteSigner expects a dictionary of secret_keys
        signer = RouteSigner(secret_keys={"vllm-test-key": "vllm-test-secret"})
        storage_dir = os.path.join(os.getcwd(), "vllm_registry_storage")
        _GLOBAL_REGISTRY = RouteRegistry(storage_dir=storage_dir, signer=signer)
    return _GLOBAL_REGISTRY

def get_vllm_runtime(model) -> HotSwapRuntime:
    """
    Returns (and initializes with the real model) the global HotSwapRuntime.
    """
    global _GLOBAL_RUNTIME, _GLOBAL_AUDIT_LOGGER
    
    if model is None:
        raise ValueError("get_vllm_runtime requires a real model instance.")

    if _GLOBAL_AUDIT_LOGGER is None:
        audit_log_path = os.path.join(os.getcwd(), "vllm_scalpel_audit.jsonl")
        _GLOBAL_AUDIT_LOGGER = AuditLogger(log_path=audit_log_path)
        
    if _GLOBAL_RUNTIME is None:
        # Initialize the Runtime with the actual model provided by vLLM ModelRunner
        _GLOBAL_RUNTIME = HotSwapRuntime(
            target_model=model,
            registry=get_vllm_registry(),
            runtime_model_hash="vllm-opt-125m-hash",
            audit_logger=_GLOBAL_AUDIT_LOGGER
        )
        print(f"[Neural-Scalpel] HotSwapRuntime initialized for vLLM with model={type(model).__name__}")
        
    return _GLOBAL_RUNTIME
