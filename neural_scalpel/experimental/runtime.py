"""
Neural-Scalpel Hot-Swap Runtime (Production Candidate)

Orchestrates the complete swap/rollback lifecycle with formal state machine
transitions, failure policy enforcement, and comprehensive audit logging.

This is the central production runtime that integrates:
  - RuntimeStateMachine for formal state transitions
  - FailurePolicy for fail-close decision making
  - PayloadValidator for pre-swap integrity checks
  - AuditLogger for structured event tracing
  - PrometheusMetrics for observability

Lifecycle:
  READY → VERIFYING → SNAPSHOT_CAPTURED → SWAPPING → SWAPPED
  → FORWARDING → ROLLING_BACK → READY

Failure paths:
  Any phase → ROUTE_QUARANTINED (recoverable)
  ROLLING_BACK → WORKER_QUARANTINED (terminal, requires restart)
"""

import threading
import time
import os
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
import hashlib

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.experimental.audit import AuditLogger
from neural_scalpel.experimental.runtime_state import (
    RuntimeStateMachine,
    RuntimePhase,
    InvalidStateTransition,
)
from neural_scalpel.experimental.failure_policy import (
    FailureClass,
    evaluate_failure,
    FailureVerdict,
)
from neural_scalpel.route.payload_validator import (
    validate_payload,
    PayloadValidationError,
)
from neural_scalpel.route.payload import load_payload_deltas


# ── Backward Compatibility ─────────────────────────────────────────────
# The old RuntimeState enum is preserved as an alias so that existing
# tests and benchmarks can continue to import it without changes.

class RuntimeState(Enum):
    """Legacy state enum — maps to RuntimePhase for backward compatibility."""
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
    Hardened prototype path for Production Candidate evaluation:
    Orchestrates the complete swap/rollback lifecycle with formal state machine
    transitions, failure policy enforcement, and comprehensive audit logging.
    """

    def __init__(
        self,
        target_model,
        registry: RouteRegistry,
        runtime_model_hash: str,
        audit_logger: Optional[AuditLogger] = None,
        payload_base_dir: Optional[str] = None,
        max_payload_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        self.model = target_model
        self.registry = registry
        self.runtime_model_hash = runtime_model_hash
        self.audit_logger = audit_logger
        self.payload_base_dir = payload_base_dir
        self.max_payload_bytes = max_payload_bytes

        self._state_machine = RuntimeStateMachine()
        self._lock = threading.Lock()
        self._snapshots: Dict[str, torch.Tensor] = {}
        self._checksums: Dict[str, str] = {}
        self._name_mapping: Dict[str, str] = {}
        self._validated_delta_cache: Dict[str, Dict[str, torch.Tensor]] = {} # Phase 5-C: Delta cache
        
        self.active_route_id: str = "__base__" # Phase 5-C: Persistence tracking
        self.swap_count: int = 0       # Phase 5-C: Direct counters (process-safe)
        self.rollback_count: int = 0
        self.last_timings: Dict[str, float] = {}

    # ── State Properties ───────────────────────────────────────

    @property
    def state(self):
        """Backward-compatible state access."""
        return self._state_machine.phase

    @state.setter
    def state(self, value):
        """Backward-compatible state setter for tests."""
        self._state_machine.phase = value

    @property
    def is_quarantined(self) -> bool:
        return self._state_machine.is_quarantined

    @property
    def is_healthy(self) -> bool:
        return not self._state_machine.is_terminal

    @property
    def can_accept_requests(self) -> bool:
        return self._state_machine.can_accept_requests

    # ── Internal Helpers ───────────────────────────────────────

    def _get_state_dict(self) -> Optional[dict]:
        if self.model is None:
            return None
        if hasattr(self.model, "state_dict"):
            return self.model.state_dict()
        if isinstance(self.model, dict):
            return self.model
        return None

    def _calculate_checksum(self, tensor: torch.Tensor) -> str:
        data_bytes = tensor.contiguous().cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def _log(self, request_id: str, tenant_id: str, route_id: str,
             event: str, status: str, latency_ms: float = 0.0, **kwargs):
        if self.audit_logger:
            self.audit_logger.log_event(
                request_id, tenant_id, route_id, event, status, latency_ms, **kwargs
            )

    def _write_audit_event(self, event: str, route_id: str, latency_ms: float = 0.0, **kwargs):
        """Phase 5-C: Direct file write for cross-process audit reliability.
        Bypasses Python logging (which doesn't survive subprocess fork)."""
        import json as _json
        from datetime import datetime, timezone
        audit_path = os.environ.get("SCALPEL_AUDIT_LOG")
        if not audit_path:
            return
        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "route_id": route_id,
                "latency_ms": round(latency_ms, 2),
                "active_route": self.active_route_id,
                "swap_count": self.swap_count,
                "rollback_count": self.rollback_count,
            }
            payload.update(kwargs)
            entry = _json.dumps(payload)
            with open(audit_path, "a") as f:
                f.write(entry + "\n")
                f.flush()
        except Exception as e:
            print(f"[Neural-Scalpel AUDIT] Write failed: {e}")

    def _resolve_layer_name(self, name: str, state: dict) -> str:
        """
        Robustly resolves a manifest layer name to a base model layer name.
        Handles common prefix variations (model., model.model.layers., etc).
        """
        candidates = [name]

        # Model prefix variations
        if name.startswith("model."):
            candidates.append(name[len("model."):])
        else:
            candidates.append(f"model.{name}")

        # PEFT nesting variations
        if name.startswith("model.layers."):
            candidates.append(name.replace("model.layers.", "model.model.layers.", 1))
        if name.startswith("model.model.layers."):
            candidates.append(name.replace("model.model.layers.", "model.layers.", 1))
        
        # Absolute 'layers.' variations
        if name.startswith("layers."):
            candidates.append(f"model.layers.{name[len('layers.'):]}")
            candidates.append(f"model.model.layers.{name[len('layers.'):]}")

        # 1. Direct match
        for c in candidates:
            if c in state:
                return c

        # 2. Suffix match (only if unique)
        matches = [k for k in state.keys() if k.endswith(name)]
        if len(matches) == 1:
            return matches[0]

        matches = [k for k in state.keys() if name.endswith(k)]
        if len(matches) == 1:
            return matches[0]

        available_keys = list(state.keys())[:10]
        raise ValueError(
            f"Target layer {name} not found in base model. "
            f"Fuzzy candidates tried: {candidates}. "
            f"First 10 available keys: {available_keys}"
        )

    def _apply_verdict(self, verdict: FailureVerdict, route_id: str,
                       request_id: str, tenant_id: str):
        """Applies a failure verdict: quarantine route/worker as prescribed."""
        if verdict.quarantine_scope is not None:
            if verdict.target_phase == RuntimePhase.WORKER_QUARANTINED:
                try:
                    self._state_machine.transition(
                        RuntimePhase.WORKER_QUARANTINED, reason=verdict.message
                    )
                except InvalidStateTransition:
                    self._state_machine.phase = RuntimePhase.WORKER_QUARANTINED
                    self._state_machine.quarantine_reason = verdict.message
                self._log(request_id, tenant_id, route_id,
                          "worker_quarantined", "failure",
                          failure_reason=verdict.message)
            elif verdict.target_phase == RuntimePhase.ROUTE_QUARANTINED:
                self.registry.quarantine_route(route_id, verdict.message)
                try:
                    self._state_machine.transition(
                        RuntimePhase.ROUTE_QUARANTINED, reason=verdict.message
                    )
                except InvalidStateTransition:
                    pass
                self._log(request_id, tenant_id, route_id,
                          "route_quarantined", "failure",
                          failure_reason=verdict.message)

    # ── Core Lifecycle ─────────────────────────────────────────

    def capture_snapshot(self, route_data: dict):
        """Captures pre-swap snapshots and checksums for all target layers.
        
        Phase 5-C Optimization: Uses a 'golden snapshot' cache. The first snapshot
        taken for a given set of layers is preserved as the canonical baseline.
        Subsequent swaps reuse this golden snapshot instead of re-capturing from
        a potentially fp-drifted state after rollback. This ensures exact
        Leave-No-Trace restoration across multiple swap/rollback cycles.
        """
        state = self._get_state_dict()
        if state is None:
            raise RuntimeError("Base model state dict is not accessible.")

        # Phase 5-C: If we already have golden snapshots for these layers, reuse them.
        # This prevents fp drift from accumulating across swap/rollback cycles.
        route_layers = [layer["name"] for layer in route_data.get("layers", [])]
        if self._snapshots and all(name in self._snapshots for name in route_layers):
            return  # Golden snapshots already captured

        self._snapshots.clear()
        self._checksums.clear()
        self._name_mapping.clear()

        for layer in route_data.get("layers", []):
            name = layer["name"]
            target_name = self._resolve_layer_name(name, state)
            
            self._name_mapping[name] = target_name
            live_tensor = state[target_name]

            # Shape validation
            if list(live_tensor.shape) != layer["shape"]:
                raise ValueError(
                    f"Shape mismatch for {name} (resolved to {target_name}): "
                    f"expected {layer['shape']}, got {list(live_tensor.shape)}"
                )

            # Dtype validation
            live_dtype = str(live_tensor.dtype).replace("torch.", "")
            if live_dtype != layer["dtype"]:
                raise ValueError(
                    f"Dtype mismatch for {name} (resolved to {target_name}): "
                    f"expected {layer['dtype']}, got {live_dtype}"
                )

            self._snapshots[name] = live_tensor.clone()
            self._checksums[name] = self._calculate_checksum(live_tensor)

    def apply_swap(self, route_data: dict, validated_deltas: Optional[Dict[str, torch.Tensor]] = None):
        """Applies weight deltas to the live model."""
        state = self._get_state_dict()

        device = "cpu"
        for v in state.values():
            if hasattr(v, "device"):
                device = str(v.device)
                break

        # Use pre-validated deltas or fall back to loading
        deltas = validated_deltas or {}
        if not deltas and route_data.get("payload"):
            deltas = load_payload_deltas(
                route_data, base_dir=self.payload_base_dir, device=device
            )

        with torch.no_grad():
            for layer in route_data.get("layers", []):
                name = layer["name"]
                target_name = self._name_mapping.get(name, name)
                
                if target_name not in state:
                    continue # Should have been caught by capture_snapshot
                    
                live_tensor = state[target_name]
                
                # Resolve delta key (try manifest name then resolved target name)
                delta = None
                if name in deltas:
                    delta = deltas[name]
                elif target_name in deltas:
                    delta = deltas[target_name]

                if delta is not None:
                    # Enforce live tensor dtype for safety
                    live_tensor.add_(delta.to(device=live_tensor.device, dtype=live_tensor.dtype))
                else:
                    # Fail-close if it's a real payload route, fallback to dummy only for simulation
                    if route_data.get("payload"):
                        raise KeyError(
                            f"Validated delta for layer '{name}' (resolved to '{target_name}') not found in payload. "
                            "Failing swap to prevent inconsistent state."
                        )
                    # Simulation/Demo mode fallback
                    dummy = (torch.ones_like(live_tensor) * 0.01).to(live_tensor.device)
                    live_tensor.add_(dummy)

    def rollback(self):
        """Restores all modified layers to their pre-swap snapshot state."""
        state = self._get_state_dict()
        with torch.no_grad():
            for name, snapshot in self._snapshots.items():
                target_name = self._name_mapping.get(name, name)
                if state[target_name].is_cuda:
                    torch.cuda.synchronize(state[target_name].device)
                state[target_name].copy_(snapshot)

    def verify_rollback(self) -> bool:
        """Verifies post-rollback checksums match pre-swap checksums."""
        state = self._get_state_dict()
        for name, expected in self._checksums.items():
            target_name = self._name_mapping.get(name, name)
            actual = self._calculate_checksum(state[target_name])
            if actual != expected:
                return False
        return True

    # ── Main Inference Gateway ─────────────────────────────────

    def infer(
        self,
        route_id: str,
        current_tenant: TenantContext,
        request_id: str,
        inference_func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Production inference gateway with formal state machine transitions
        and fail-close policy enforcement.
        """
        tenant_id = current_tenant.tenant_id

        # ── Guard: reject if worker is quarantined ─────────────
        if self._state_machine.is_terminal:
            raise RuntimeError(
                "CRITICAL: Worker is QUARANTINED. Cannot accept requests. "
                "Restart required."
            )

        # ── Step 1: Route verification ─────────────────────────
        try:
            route_data = self.registry.get_verified_route(
                route_id, self.runtime_model_hash, current_tenant
            )
            self._log(request_id, tenant_id, route_id,
                      "route_verified", "success",
                      route_version=route_data.get("route_schema_version", "unknown"),
                      route_sha256=route_data.get("source_adapter_sha256", "unknown"))
        except Exception as e:
            self._log(request_id, tenant_id, route_id,
                      "route_rejected", "failure", failure_reason=str(e))
            raise

        # ── Step 2: Pre-swap payload validation ────────────────
        validated_deltas = {}
        try:
            t_validate = time.perf_counter()
            validated_deltas = validate_payload(
                route_data,
                base_dir=self.payload_base_dir,
                max_payload_bytes=self.max_payload_bytes,
            )
            validate_lat = time.perf_counter() - t_validate
            self.last_timings["payload_validate_latency"] = validate_lat
            if validated_deltas:
                self._log(request_id, tenant_id, route_id,
                          "payload_validated", "success", validate_lat * 1000)
        except PayloadValidationError as e:
            verdict = evaluate_failure(e.failure_class)
            self._apply_verdict(verdict, route_id, request_id, tenant_id)
            self._log(request_id, tenant_id, route_id,
                      "payload_rejected", "failure",
                      failure_reason=str(e),
                      failure_class=e.failure_class.value)
            raise RuntimeError(verdict.message) from e

        # ── Step 3: Locked swap/infer/rollback cycle ───────────
        t_lock_start = time.perf_counter()
        with self._lock:
            self.last_timings["lock_wait_time"] = time.perf_counter() - t_lock_start
            output = None

            try:
                # Transition: READY → VERIFYING
                self._state_machine.transition(RuntimePhase.VERIFYING, route_id=route_id)

                # Snapshot capture
                t_swap_start = time.perf_counter()
                self._log(request_id, tenant_id, route_id, "snapshot_started", "pending")
                self.capture_snapshot(route_data)
                self._state_machine.transition(RuntimePhase.SNAPSHOT_CAPTURED)
                self._log(request_id, tenant_id, route_id, "snapshot_captured", "success")

                # Swap
                self._state_machine.transition(RuntimePhase.SWAPPING)
                self._log(request_id, tenant_id, route_id, "swap_started", "pending")
                self.apply_swap(route_data, validated_deltas)
                swap_lat = time.perf_counter() - t_swap_start
                self.last_timings["swap_latency"] = swap_lat
                self._state_machine.transition(RuntimePhase.SWAPPED)
                self._log(request_id, tenant_id, route_id,
                          "swap_completed", "success", swap_lat * 1000)

                # Inference
                self._state_machine.transition(RuntimePhase.FORWARDING)
                self._log(request_id, tenant_id, route_id, "forward_started", "pending")
                t_inf_start = time.perf_counter()
                output = inference_func(*args, **kwargs)
                inf_lat = time.perf_counter() - t_inf_start
                self._log(request_id, tenant_id, route_id,
                          "forward_completed", "success", inf_lat * 1000)

            except Exception as e:
                # Exception during swap or inference → attempt rollback
                self._log(request_id, tenant_id, route_id,
                          "exception_during_lifecycle", "failure",
                          failure_reason=str(e))
                self._perform_rollback(request_id, tenant_id, route_id)
                raise

            else:
                # Normal rollback after successful inference
                self._perform_rollback(request_id, tenant_id, route_id)

        # Final guard
        if self._state_machine.is_terminal:
            raise RuntimeError(
                "CRITICAL: Rollback checksum mismatch. Worker QUARANTINED."
            )

        return output

    def ensure_route(
        self,
        route_id: str,
        current_tenant: TenantContext,
        request_id: str
    ) -> bool:
        """
        Phase 5-C: Persistently ensures the model is in the correct route state.
        Only performs rollback/swap if the route_id has changed.
        Returns True if a swap occurred, False if the route was already active.
        """
        if self._state_machine.is_terminal:
            raise RuntimeError("Worker is QUARANTINED.")

        # 1. Check if already active
        if self.active_route_id == route_id:
            return False

        tenant_id = current_tenant.tenant_id

        # 2. Rollback current route if active
        if self.active_route_id != "__base__":
            self._perform_rollback(request_id, tenant_id, self.active_route_id)
            self.active_route_id = "__base__"

        # 3. If target is base, we are done
        if route_id == "__base__":
            return True

        # 4. Prepare target route
        try:
            route_data = self.registry.get_verified_route(
                route_id, self.runtime_model_hash, current_tenant
            )
        except Exception as e:
            self._log(request_id, tenant_id, route_id, "route_rejected", "failure", failure_reason=str(e))
            raise

        # 5. Get or Validate Deltas (Cache optimized)
        if route_id not in self._validated_delta_cache:
            t_validate = time.perf_counter()
            deltas = validate_payload(
                route_data,
                base_dir=self.payload_base_dir,
                max_payload_bytes=self.max_payload_bytes,
            )
            self._validated_delta_cache[route_id] = deltas
            self._log(request_id, tenant_id, route_id, "payload_validated", "success", (time.perf_counter() - t_validate) * 1000)
        
        validated_deltas = self._validated_delta_cache[route_id]

        # 6. Atomic Swap Cycle
        with self._lock:
            try:
                self._state_machine.transition(RuntimePhase.VERIFYING, route_id=route_id)
                
                # Snapshot
                self.capture_snapshot(route_data)
                self._state_machine.transition(RuntimePhase.SNAPSHOT_CAPTURED)
                
                # Swap
                self._state_machine.transition(RuntimePhase.SWAPPING)
                t_swap = time.perf_counter()
                self.apply_swap(route_data, validated_deltas)
                swap_ms = (time.perf_counter() - t_swap) * 1000
                
                self._state_machine.transition(RuntimePhase.SWAPPED)
                self.active_route_id = route_id
                self.swap_count += 1
                
                self._log(request_id, tenant_id, route_id, "swap_completed", "success", swap_ms)
                self._write_audit_event("swap_completed", route_id, swap_ms)
                print(f"[Neural-Scalpel] SWAP #{self.swap_count}: {route_id} ({swap_ms:.1f}ms)")
                return True
            except Exception as e:
                self._log(request_id, tenant_id, route_id, "swap_failed", "failure", failure_reason=str(e))
                self._perform_rollback(request_id, tenant_id, route_id)
                self.active_route_id = "__base__"
                raise

    def clear_active_route(self) -> bool:
        """Phase 5-C: Force rollback to base state. Returns True if rollback was performed."""
        if self.active_route_id != "__base__":
            print(f"[Neural-Scalpel] Forced cleanup of route: {self.active_route_id}")
            self._perform_rollback("cleanup", "system", self.active_route_id)
            self.active_route_id = "__base__"
            return True
        return False

    def _perform_rollback(self, request_id: str, tenant_id: str, route_id: str):
        """Executes rollback with checksum verification and failure escalation."""
        try:
            self._state_machine.transition(RuntimePhase.ROLLING_BACK)
        except InvalidStateTransition:
            # Force transition for error recovery
            self._state_machine.phase = RuntimePhase.ROLLING_BACK

        t_rb = time.perf_counter()
        self._log(request_id, tenant_id, route_id, "rollback_started", "pending")

        try:
            self.rollback()
        except Exception as e:
            rb_lat = time.perf_counter() - t_rb
            self.last_timings["rollback_latency"] = rb_lat
            verdict = evaluate_failure(FailureClass.ROLLBACK_EXCEPTION)
            self._apply_verdict(verdict, route_id, request_id, tenant_id)
            self._log(request_id, tenant_id, route_id,
                      "rollback_failed", "failure", rb_lat * 1000,
                      failure_reason=str(e))
            return

        rb_lat = time.perf_counter() - t_rb
        self.last_timings["rollback_latency"] = rb_lat

        if not self.verify_rollback():
            verdict = evaluate_failure(FailureClass.ROLLBACK_CHECKSUM_MISMATCH)
            self._apply_verdict(verdict, route_id, request_id, tenant_id)
            self._log(request_id, tenant_id, route_id,
                      "rollback_failed", "failure", rb_lat * 1000,
                      failure_reason="Checksum mismatch after rollback")
        else:
            self.rollback_count += 1
            rb_ms = rb_lat * 1000
            self._log(request_id, tenant_id, route_id,
                      "rollback_completed", "success", rb_ms)
            self._write_audit_event("rollback_completed", route_id, rb_ms, rollback_verified=True)
            print(f"[Neural-Scalpel] ROLLBACK #{self.rollback_count}: {route_id} ({rb_ms:.1f}ms)")
            try:
                self._state_machine.transition(RuntimePhase.READY)
            except InvalidStateTransition:
                self._state_machine.phase = RuntimePhase.READY

    # ── Backward Compatibility ─────────────────────────────────

    def transition(self, new_state):
        """Legacy transition method for backward compatibility."""
        from neural_scalpel.experimental.runtime_state import RuntimePhase as RP
        # Map old RuntimeState enum values to new RuntimePhase
        state_map = {
            "IDLE": RP.READY,
            "VERIFYING": RP.VERIFYING,
            "LOCKED": RP.VERIFYING,
            "SWAPPING": RP.SWAPPING,
            "INFERENCE_ACTIVE": RP.FORWARDING,
            "ROLLBACK_PENDING": RP.ROLLING_BACK,
            "ROLLED_BACK": RP.READY,
            "QUARANTINED": RP.WORKER_QUARANTINED,
        }
        if hasattr(new_state, "value"):
            target = state_map.get(new_state.value, RP.READY)
        else:
            target = state_map.get(str(new_state), RP.READY)
        try:
            self._state_machine.transition(target)
        except InvalidStateTransition:
            self._state_machine.phase = target

    def capture_and_verify(self, route_data: dict):
        """Legacy method for backward compatibility. Now includes state transitions."""
        try:
            self._state_machine.transition(RuntimePhase.VERIFYING)
        except InvalidStateTransition:
            pass
        self.capture_snapshot(route_data)
        try:
            self._state_machine.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        except InvalidStateTransition:
            pass

    def swap(self, route_data: dict):
        """Legacy swap method for backward compatibility. Now includes state transitions."""
        try:
            self._state_machine.transition(RuntimePhase.SWAPPING)
        except InvalidStateTransition:
            pass
        self.apply_swap(route_data)
        try:
            self._state_machine.transition(RuntimePhase.SWAPPED)
        except InvalidStateTransition:
            pass

    @property
    def lock(self):
        return self._lock

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def checksums(self):
        return self._checksums