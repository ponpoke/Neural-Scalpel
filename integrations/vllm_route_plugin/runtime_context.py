"""
Runtime context manager for Neural-Scalpel vLLM integration.
Separates Registry lifecycle from Runtime initialization.
"""
import os
import json
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
        
        # Use environment variables with sensible defaults for cross-process consistency
        base_dir = os.environ.get("SCALPEL_HOME", os.getcwd())
        storage_dir = os.environ.get(
            "SCALPEL_VLLM_REGISTRY_DIR",
            os.path.join(base_dir, "vllm_registry_storage")
        )
        print(f"[Neural-Scalpel] Initializing Registry with storage_dir: {storage_dir}")
        
        _GLOBAL_REGISTRY = RouteRegistry(storage_dir=storage_dir, signer=signer)
        
        # PROTOTYPE HACK: Auto-load routes from storage to sync across vLLM processes
        if os.path.exists(storage_dir):
            for f in os.listdir(storage_dir):
                if f.endswith(".scalpel_route"):
                    try:
                        path = os.path.join(storage_dir, f)
                        with open(path, "r") as rf:
                            data = json.load(rf)
                            rid = data.get("route_id")
                            if rid:
                                _GLOBAL_REGISTRY.routes[rid] = data
                                from neural_scalpel.route.policy import RouteStatus
                                _GLOBAL_REGISTRY.statuses[rid] = RouteStatus.PRODUCTION
                                print(f"[Neural-Scalpel] Auto-loaded route '{rid}' from {f}")
                    except Exception as e:
                        print(f"[Neural-Scalpel] Failed to auto-load route {f}: {e}")
        else:
            print(f"[Neural-Scalpel] Registry storage dir NOT FOUND: {storage_dir}")
                        
    return _GLOBAL_REGISTRY

def get_vllm_runtime(model) -> HotSwapRuntime:
    """
    Returns (and initializes with the real model) the global HotSwapRuntime.
    """
    global _GLOBAL_RUNTIME, _GLOBAL_AUDIT_LOGGER
    
    if model is None:
        raise ValueError("get_vllm_runtime requires a real model instance.")

    if _GLOBAL_AUDIT_LOGGER is None:
        base_dir = os.environ.get("SCALPEL_HOME", os.getcwd())
        audit_log_path = os.environ.get(
            "SCALPEL_AUDIT_LOG",
            os.path.join(base_dir, "vllm_scalpel_audit.jsonl")
        )
        _GLOBAL_AUDIT_LOGGER = AuditLogger(log_file_path=audit_log_path)
        
    if _GLOBAL_RUNTIME is None:
        # Environment-variable-ize runtime hash for evaluation flexibility
        runtime_hash = os.environ.get("SCALPEL_RUNTIME_MODEL_HASH", "qwen2.5-0.5b-vllm-hash")
        
        # Initialize the Runtime with the actual model provided by vLLM ModelRunner
        _GLOBAL_RUNTIME = HotSwapRuntime(
            target_model=model,
            registry=get_vllm_registry(),
            runtime_model_hash=runtime_hash,
            audit_logger=_GLOBAL_AUDIT_LOGGER
        )
        print(f"[Neural-Scalpel] HotSwapRuntime initialized for vLLM with model={type(model).__name__}")
        
    return _GLOBAL_RUNTIME

def get_current_vllm_runtime() -> Optional[HotSwapRuntime]:
    """Returns the currently initialized HotSwapRuntime without requiring a model instance."""
    global _GLOBAL_RUNTIME
    return _GLOBAL_RUNTIME
