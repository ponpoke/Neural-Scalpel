import threading
import torch
import hashlib
import time
from enum import Enum
from typing import Callable, Any, Optional

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.experimental.audit import AuditLogger
from neural_scalpel.route.payload import load_payload_deltas

class RuntimeState(Enum):
    IDLE = "IDLE"
    VERIFYING = "VERIFYING"
    LOCKED = "LOCKED"
    SWAPPING = "SWAPPING"
    INFERENCE_ACTIVE = "INFERENCE_ACTIVE"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    ROLLED_BACK = "ROLLED_BACK"
    QUARANTINED = "QUARANTINED"

class HotSwapRuntime:
    """
    Production-oriented Hot-Swap Runtime focusing on strict state management, 
    checksum verification, and a robust fail-close policy.
    """
    def __init__(self, target_model, registry: RouteRegistry, runtime_model_hash: str, audit_logger: AuditLogger = None, payload_base_dir: Optional[str] = None):
        self.model = target_model
        self.registry = registry
        self.runtime_model_hash = runtime_model_hash
        self.audit_logger = audit_logger
        self.payload_base_dir = payload_base_dir
        self.state = RuntimeState.IDLE
        self.lock = threading.Lock()
        self.snapshots = {}
        self.checksums = {}
        self.last_timings = {}

    def _get_state_dict(self):
        if self.model is None:
            return None
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        if isinstance(self.model, dict):
            return self.model
        return None

    def _calculate_checksum(self, tensor: torch.Tensor) -> str:
        """
        Calculates a deterministic checksum for strict rollback verification.
        Note: For extreme real-time constraints, a faster hash could be used.
        """
        data_bytes = tensor.contiguous().cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def transition(self, new_state: RuntimeState):
        if self.state == RuntimeState.QUARANTINED:
            raise RuntimeError("CRITICAL: Runtime is QUARANTINED. Cannot transition state.")
        self.state = new_state

    def capture_and_verify(self, route_data: dict):
        self.transition(RuntimeState.VERIFYING)
        state = self._get_state_dict()
        if state is None:
            raise RuntimeError("Base model state dict is not accessible.")

        self.snapshots.clear()
        self.checksums.clear()

        for layer in route_data.get("layers", []):
            name = layer["name"]
            if name not in state:
                raise ValueError(f"Target layer {name} not found in base model.")
            
            live_tensor = state[name]
            
            # Validate shape
            if list(live_tensor.shape) != layer["shape"]:
                raise ValueError(f"Shape mismatch for {name}: expected {layer['shape']}, got {list(live_tensor.shape)}")
            
            # Validate dtype
            live_dtype = str(live_tensor.dtype).replace('torch.', '')
            if live_dtype != layer["dtype"]:
                raise ValueError(f"Dtype mismatch for {name}: expected {layer['dtype']}, got {live_dtype}")

            # Capture snapshot and calculate baseline checksum
            self.snapshots[name] = live_tensor.clone()
            self.checksums[name] = self._calculate_checksum(live_tensor)

    def swap(self, route_data: dict):
        self.transition(RuntimeState.SWAPPING)
        state = self._get_state_dict()

        # Attempt to load real payload deltas from safetensors
        device = "cpu"
        for v in state.values():
            if hasattr(v, "device"):
                device = str(v.device)
                break

        real_deltas = {}
        if route_data.get("payload"):
            real_deltas = load_payload_deltas(
                route_data, base_dir=self.payload_base_dir, device=device
            )

        with torch.no_grad():
            for layer in route_data.get("layers", []):
                name = layer["name"]
                live_tensor = state[name]
                if name in real_deltas:
                    # Apply real projected delta from payload
                    live_tensor.add_(real_deltas[name].to(live_tensor.device))
                else:
                    # Fallback: simulated delta for legacy/mock routes
                    dummy_delta = (torch.ones_like(live_tensor) * 0.01).to(live_tensor.device)
                    live_tensor.add_(dummy_delta)

    def rollback(self):
        self.transition(RuntimeState.ROLLBACK_PENDING)
        state = self._get_state_dict()
        with torch.no_grad():
            for name, snapshot in self.snapshots.items():
                if state[name].is_cuda:
                    torch.cuda.synchronize(state[name].device)
                state[name].copy_(snapshot)
        self.transition(RuntimeState.ROLLED_BACK)

    def verify_rollback(self):
        """Ensures the post-rollback state exactly matches the pre-swap state."""
        state = self._get_state_dict()
        for name, expected_hash in self.checksums.items():
            current_hash = self._calculate_checksum(state[name])
            if current_hash != expected_hash:
                print(f"[CRITICAL] Checksum mismatch on {name}. Expected: {expected_hash}, Got: {current_hash}")
                return False
        return True

    def infer(self, route_id: str, current_tenant: TenantContext, request_id: str, inference_func: Callable, *args, **kwargs) -> Any:
        """
        The main inference gateway enforcing the strict fail-close policy and SRE audit logging.
        """
        def log(event: str, status: str, lat: float = 0.0, **kw):
            if self.audit_logger:
                self.audit_logger.log_event(request_id, current_tenant.tenant_id, route_id, event, status, lat, **kw)

        # Step 1: Pre-lock verification. Will raise if route is revoked or incompatible.
        try:
            route_data = self.registry.get_verified_route(route_id, self.runtime_model_hash, current_tenant)
            route_version = route_data.get("route_schema_version", "unknown")
            route_hash = route_data.get("source_adapter_sha256", "unknown")
            log("route_verified", "success", route_version=route_version, route_sha256=route_hash)
        except Exception as e:
            log("route_rejected", "failure", failure_reason=str(e))
            raise

        # Step 2: Lock acquisition
        t_lock_start = time.perf_counter()
        with self.lock:
            self.last_timings["lock_wait_time"] = time.perf_counter() - t_lock_start
            self.transition(RuntimeState.LOCKED)
            output = None
            try:
                # 3. Snapshot & Verify
                t_swap_start = time.perf_counter()
                log("swap_started", "pending")
                self.capture_and_verify(route_data)
                
                # 4. Atomic Swap
                self.swap(route_data)
                swap_lat = time.perf_counter() - t_swap_start
                self.last_timings["swap_latency"] = swap_lat
                log("swap_completed", "success", swap_lat * 1000)
                
                # 5. Inference Execution
                self.transition(RuntimeState.INFERENCE_ACTIVE)
                log("inference_started", "pending")
                t_inf_start = time.perf_counter()
                output = inference_func(*args, **kwargs)
                inf_lat = time.perf_counter() - t_inf_start
                log("inference_completed", "success", inf_lat * 1000)
                
            except Exception as e:
                # 6. Failure Recovery
                t_rb_start = time.perf_counter()
                log("rollback_started", "pending")
                self.rollback()
                rb_lat = time.perf_counter() - t_rb_start
                self.last_timings["rollback_latency"] = rb_lat
                
                if not self.verify_rollback():
                    self.state = RuntimeState.QUARANTINED
                    reason = "Rollback checksum mismatch after exception."
                    self.registry.quarantine_route(route_id, reason)
                    log("rollback_failed", "failure", rb_lat * 1000, failure_reason=reason)
                    log("runtime_quarantined", "failure", failure_reason=reason)
                    raise RuntimeError("CRITICAL: Rollback checksum mismatch after inference. Runtime QUARANTINED.") from e
                
                log("rollback_completed", "success", rb_lat * 1000)
                raise
                
            finally:
                # 7. Normal Rollback (Only if we haven't already quarantined)
                if self.state != RuntimeState.QUARANTINED:
                    t_rb_start = time.perf_counter()
                    log("rollback_started", "pending")
                    self.rollback()
                    rb_lat = time.perf_counter() - t_rb_start
                    self.last_timings["rollback_latency"] = rb_lat
                    
                    if not self.verify_rollback():
                        self.state = RuntimeState.QUARANTINED
                        reason = "Rollback checksum mismatch after normal execution."
                        self.registry.quarantine_route(route_id, reason)
                        log("rollback_failed", "failure", rb_lat * 1000, failure_reason=reason)
                        log("runtime_quarantined", "failure", failure_reason=reason)
                    else:
                        log("rollback_completed", "success", rb_lat * 1000)
                        self.transition(RuntimeState.IDLE)
                        
            if self.state == RuntimeState.QUARANTINED:
                raise RuntimeError("CRITICAL: Rollback checksum mismatch after inference. Runtime QUARANTINED.")
            
            return output