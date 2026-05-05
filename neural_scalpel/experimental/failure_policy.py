"""
Neural-Scalpel Failure Policy Engine

Defines the fail-close decision logic for every known failure mode.
Each failure class maps to a concrete remediation action with full audit metadata.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from neural_scalpel.experimental.runtime_state import RuntimePhase, QuarantineScope


class FailureClass(Enum):
    """Exhaustive catalogue of failure modes during hot-swap."""
    # P1-A: Corrupted safetensors
    PAYLOAD_FILE_NOT_FOUND = "payload_file_not_found"
    PAYLOAD_SHA256_MISMATCH = "payload_sha256_mismatch"
    PAYLOAD_HEADER_CORRUPTED = "payload_header_corrupted"
    PAYLOAD_TENSOR_KEY_MISSING = "payload_tensor_key_missing"
    PAYLOAD_SHAPE_MISMATCH = "payload_shape_mismatch"
    PAYLOAD_DTYPE_MISMATCH = "payload_dtype_mismatch"
    PAYLOAD_NAN_INF_DETECTED = "payload_nan_inf_detected"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    PAYLOAD_IO_ERROR = "payload_io_error"
    # P1-B: Swap failure
    SWAP_TARGET_LAYER_NOT_FOUND = "swap_target_layer_not_found"
    SWAP_LOCK_ACQUISITION_FAILED = "swap_lock_acquisition_failed"
    SWAP_VRAM_INSUFFICIENT = "swap_vram_insufficient"
    SWAP_DELTA_APPLICATION_ERROR = "swap_delta_application_error"
    SWAP_STATE_STUCK = "swap_state_stuck"
    # P1-C: Rollback failure
    ROLLBACK_CHECKSUM_MISMATCH = "rollback_checksum_mismatch"
    ROLLBACK_EXCEPTION = "rollback_exception"
    ROLLBACK_PARTIAL_RESTORE = "rollback_partial_restore"
    # Runtime
    INFERENCE_EXCEPTION = "inference_exception"
    STATE_TRANSITION_INVALID = "state_transition_invalid"


class RemediationAction(Enum):
    REJECT_REQUEST = "reject_request"
    QUARANTINE_ROUTE = "quarantine_route"
    QUARANTINE_WORKER = "quarantine_worker"
    ABORT_AND_ROLLBACK = "abort_and_rollback"


@dataclass(frozen=True)
class FailureVerdict:
    """Immutable outcome of evaluating a failure against the policy."""
    failure_class: FailureClass
    action: RemediationAction
    quarantine_scope: Optional[QuarantineScope]
    target_phase: RuntimePhase
    http_status: int
    message: str
    is_critical: bool = False


def _v(fc, action, scope, phase, status, msg, critical=False):
    """Shorthand factory for building the policy table."""
    return (fc, FailureVerdict(fc, action, scope, phase, status, msg, critical))


_entries = [
    _v(FailureClass.PAYLOAD_FILE_NOT_FOUND, RemediationAction.REJECT_REQUEST,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Payload file not found. Route quarantined."),
    _v(FailureClass.PAYLOAD_SHA256_MISMATCH, RemediationAction.QUARANTINE_ROUTE,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Payload SHA-256 integrity check failed. Route quarantined.", True),
    _v(FailureClass.PAYLOAD_HEADER_CORRUPTED, RemediationAction.QUARANTINE_ROUTE,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Safetensors header corrupted. Route quarantined."),
    _v(FailureClass.PAYLOAD_TENSOR_KEY_MISSING, RemediationAction.REJECT_REQUEST,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Required tensor key missing from payload. Route quarantined."),
    _v(FailureClass.PAYLOAD_SHAPE_MISMATCH, RemediationAction.REJECT_REQUEST,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Tensor shape mismatch in payload. Route quarantined."),
    _v(FailureClass.PAYLOAD_DTYPE_MISMATCH, RemediationAction.REJECT_REQUEST,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Tensor dtype mismatch in payload. Route quarantined."),
    _v(FailureClass.PAYLOAD_NAN_INF_DETECTED, RemediationAction.QUARANTINE_ROUTE,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "NaN/Inf detected in payload tensors. Route quarantined.", True),
    _v(FailureClass.PAYLOAD_TOO_LARGE, RemediationAction.REJECT_REQUEST,
       None, RuntimePhase.READY, 413,
       "Payload exceeds maximum allowed size."),
    _v(FailureClass.PAYLOAD_IO_ERROR, RemediationAction.REJECT_REQUEST,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "I/O error reading payload. Route quarantined."),
    _v(FailureClass.SWAP_TARGET_LAYER_NOT_FOUND, RemediationAction.QUARANTINE_ROUTE,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Target layer not found. Route quarantined."),
    _v(FailureClass.SWAP_LOCK_ACQUISITION_FAILED, RemediationAction.REJECT_REQUEST,
       None, RuntimePhase.READY, 503,
       "Failed to acquire swap lock. Retry later."),
    _v(FailureClass.SWAP_VRAM_INSUFFICIENT, RemediationAction.ABORT_AND_ROLLBACK,
       QuarantineScope.ROUTE, RuntimePhase.ROUTE_QUARANTINED, 503,
       "Insufficient VRAM for swap. Route quarantined."),
    _v(FailureClass.SWAP_DELTA_APPLICATION_ERROR, RemediationAction.ABORT_AND_ROLLBACK,
       QuarantineScope.ROUTE, RuntimePhase.ROLLING_BACK, 503,
       "Error applying delta during swap. Rolling back."),
    _v(FailureClass.SWAP_STATE_STUCK, RemediationAction.QUARANTINE_WORKER,
       QuarantineScope.WORKER, RuntimePhase.WORKER_QUARANTINED, 503,
       "Runtime state stuck. Worker quarantined.", True),
    _v(FailureClass.ROLLBACK_CHECKSUM_MISMATCH, RemediationAction.QUARANTINE_WORKER,
       QuarantineScope.WORKER, RuntimePhase.WORKER_QUARANTINED, 503,
       "CRITICAL: Rollback checksum mismatch. Worker quarantined.", True),
    _v(FailureClass.ROLLBACK_EXCEPTION, RemediationAction.QUARANTINE_WORKER,
       QuarantineScope.WORKER, RuntimePhase.WORKER_QUARANTINED, 503,
       "CRITICAL: Exception during rollback. Worker quarantined.", True),
    _v(FailureClass.ROLLBACK_PARTIAL_RESTORE, RemediationAction.QUARANTINE_WORKER,
       QuarantineScope.WORKER, RuntimePhase.WORKER_QUARANTINED, 503,
       "CRITICAL: Partial rollback. Worker quarantined.", True),
    _v(FailureClass.INFERENCE_EXCEPTION, RemediationAction.ABORT_AND_ROLLBACK,
       None, RuntimePhase.ROLLING_BACK, 500,
       "Inference failed. Rolling back."),
    _v(FailureClass.STATE_TRANSITION_INVALID, RemediationAction.QUARANTINE_WORKER,
       QuarantineScope.WORKER, RuntimePhase.WORKER_QUARANTINED, 503,
       "CRITICAL: Invalid state transition. Worker quarantined.", True),
]

_POLICY_TABLE = {fc: verdict for fc, verdict in _entries}

_UNKNOWN_VERDICT = FailureVerdict(
    failure_class=FailureClass.STATE_TRANSITION_INVALID,
    action=RemediationAction.QUARANTINE_WORKER,
    quarantine_scope=QuarantineScope.WORKER,
    target_phase=RuntimePhase.WORKER_QUARANTINED,
    http_status=503,
    message="Unknown failure. Worker quarantined as precaution.",
    is_critical=True,
)


def evaluate_failure(failure_class: FailureClass) -> FailureVerdict:
    """Evaluates a failure against the policy table. Sole entry point for failure handling."""
    return _POLICY_TABLE.get(failure_class, _UNKNOWN_VERDICT)
