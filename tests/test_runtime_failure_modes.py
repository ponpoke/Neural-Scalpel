"""
Neural-Scalpel Runtime Failure Mode Tests

Comprehensive tests for P1-A, P1-B, P1-C of the production readiness roadmap.
Validates that every known failure mode triggers the correct fail-close behavior.
"""

import os
import json
import struct
import tempfile
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from safetensors.torch import save_file

# ── P1 Modules ─────────────────────────────────────────────────────────

from neural_scalpel.experimental.runtime_state import (
    RuntimeStateMachine,
    RuntimePhase,
    QuarantineScope,
    InvalidStateTransition,
)
from neural_scalpel.experimental.failure_policy import (
    FailureClass,
    RemediationAction,
    evaluate_failure,
)
from neural_scalpel.route.payload_validator import (
    validate_payload,
    PayloadValidationError,
    DEFAULT_MAX_PAYLOAD_BYTES,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def valid_payload(tmp_dir):
    """Creates a valid safetensors payload and matching route_data."""
    layer_name = "model.layers.0.self_attn.q_proj.weight"
    tensor = torch.randn(64, 64, dtype=torch.float32)
    payload_path = os.path.join(tmp_dir, "delta.safetensors")
    save_file({layer_name: tensor}, payload_path)

    from neural_scalpel.route.payload import compute_file_sha256
    sha = compute_file_sha256(payload_path)

    route_data = {
        "payload": {"uri": payload_path, "sha256": sha},
        "layers": [
            {
                "name": layer_name,
                "payload_key": layer_name,
                "shape": [64, 64],
                "dtype": "float32",
                "delta_sha256": "placeholder",
            }
        ],
    }
    return route_data, payload_path, tensor


# ═══════════════════════════════════════════════════════════════════════
# P1-A: Corrupted Safetensors Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCorruptedSafetensors:
    """Tests for P1-A: corrupted safetensors fail-close."""

    def test_corrupted_safetensors_fails_closed(self, tmp_dir):
        """Payload with corrupted header must be rejected before swap."""
        corrupt_path = os.path.join(tmp_dir, "corrupt.safetensors")
        with open(corrupt_path, "wb") as f:
            f.write(b"\x00" * 100)  # invalid header

        route_data = {
            "payload": {"uri": corrupt_path},
            "layers": [{"name": "test", "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class in (
            FailureClass.PAYLOAD_HEADER_CORRUPTED,
            FailureClass.PAYLOAD_IO_ERROR,
        )

    def test_sha_mismatch_quarantines_route(self, valid_payload):
        """SHA-256 mismatch must quarantine the route."""
        route_data, path, _ = valid_payload
        route_data["payload"]["sha256"] = "a" * 64  # wrong hash
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_SHA256_MISMATCH
        verdict = evaluate_failure(exc_info.value.failure_class)
        assert verdict.quarantine_scope == QuarantineScope.ROUTE

    def test_missing_tensor_key_rejected_before_swap(self, tmp_dir):
        """Missing tensor key in payload must be rejected."""
        payload_path = os.path.join(tmp_dir, "partial.safetensors")
        save_file({"other_key": torch.randn(4, 4)}, payload_path)

        route_data = {
            "payload": {"uri": payload_path},
            "layers": [{"name": "expected_key", "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_TENSOR_KEY_MISSING

    def test_shape_mismatch_rejected_before_swap(self, tmp_dir):
        """Shape mismatch must be rejected."""
        key = "layer.weight"
        payload_path = os.path.join(tmp_dir, "wrong_shape.safetensors")
        save_file({key: torch.randn(8, 8)}, payload_path)

        route_data = {
            "payload": {"uri": payload_path},
            "layers": [{"name": key, "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_SHAPE_MISMATCH

    def test_nan_payload_rejected(self, tmp_dir):
        """NaN values in payload must be rejected."""
        key = "layer.weight"
        nan_tensor = torch.full((4, 4), float("nan"))
        payload_path = os.path.join(tmp_dir, "nan.safetensors")
        save_file({key: nan_tensor}, payload_path)

        route_data = {
            "payload": {"uri": payload_path},
            "layers": [{"name": key, "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_NAN_INF_DETECTED

    def test_large_payload_limit(self, tmp_dir):
        """Payload exceeding size limit must be rejected."""
        large_path = os.path.join(tmp_dir, "large.safetensors")
        save_file({"test": torch.randn(4, 4)}, large_path)

        route_data = {
            "payload": {"uri": large_path},
            "layers": [{"name": "test", "shape": [4, 4], "dtype": "float32"}],
        }
        # Set an artificially small limit
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data, max_payload_bytes=10)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_TOO_LARGE

    def test_file_not_found(self):
        """Non-existent payload file must be rejected."""
        route_data = {
            "payload": {"uri": "/nonexistent/path.safetensors"},
            "layers": [],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_FILE_NOT_FOUND

    def test_dtype_mismatch_rejected(self, tmp_dir):
        """Dtype mismatch must be rejected."""
        key = "layer.weight"
        payload_path = os.path.join(tmp_dir, "wrong_dtype.safetensors")
        save_file({key: torch.randn(4, 4, dtype=torch.float16)}, payload_path)

        route_data = {
            "payload": {"uri": payload_path},
            "layers": [{"name": key, "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_DTYPE_MISMATCH

    def test_inf_payload_rejected(self, tmp_dir):
        """Inf values in payload must be rejected."""
        key = "layer.weight"
        inf_tensor = torch.full((4, 4), float("inf"))
        payload_path = os.path.join(tmp_dir, "inf.safetensors")
        save_file({key: inf_tensor}, payload_path)

        route_data = {
            "payload": {"uri": payload_path},
            "layers": [{"name": key, "shape": [4, 4], "dtype": "float32"}],
        }
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(route_data)
        assert exc_info.value.failure_class == FailureClass.PAYLOAD_NAN_INF_DETECTED


# ═══════════════════════════════════════════════════════════════════════
# P1-B/C: Runtime State Machine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeStateMachine:
    """Tests for the formal state machine governing runtime transitions."""

    def test_valid_lifecycle(self):
        """Normal lifecycle: READY → VERIFYING → ... → READY."""
        sm = RuntimeStateMachine()
        assert sm.phase == RuntimePhase.READY

        sm.transition(RuntimePhase.VERIFYING, route_id="test_route")
        assert sm.phase == RuntimePhase.VERIFYING

        sm.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        sm.transition(RuntimePhase.SWAPPING)
        sm.transition(RuntimePhase.SWAPPED)
        sm.transition(RuntimePhase.FORWARDING)
        sm.transition(RuntimePhase.ROLLING_BACK)
        sm.transition(RuntimePhase.READY)
        assert sm.is_ready
        assert len(sm.history) == 7

    def test_invalid_transition_raises(self):
        """Illegal transitions must raise InvalidStateTransition."""
        sm = RuntimeStateMachine()
        with pytest.raises(InvalidStateTransition):
            sm.transition(RuntimePhase.SWAPPING)  # READY → SWAPPING is illegal

    def test_worker_quarantine_is_terminal(self):
        """WORKER_QUARANTINED is a terminal state with no successors."""
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING)
        sm.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        sm.transition(RuntimePhase.SWAPPING)
        sm.transition(RuntimePhase.ROLLING_BACK)
        sm.transition(RuntimePhase.WORKER_QUARANTINED, reason="checksum mismatch")
        assert sm.is_terminal
        with pytest.raises(InvalidStateTransition):
            sm.transition(RuntimePhase.READY)

    def test_route_quarantine_recovery(self):
        """Route quarantine can be recovered to READY."""
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING)
        sm.transition(RuntimePhase.ROUTE_QUARANTINED, reason="SHA mismatch")
        assert sm.is_quarantined
        assert sm.quarantine_scope == QuarantineScope.ROUTE
        sm.transition(RuntimePhase.READY)
        assert sm.is_ready

    def test_quarantine_metadata(self):
        """Quarantine records scope and reason."""
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING)
        sm.transition(RuntimePhase.ROUTE_QUARANTINED, reason="test reason")
        state = sm.to_dict()
        assert state["quarantine_scope"] == "ROUTE"
        assert state["quarantine_reason"] == "test reason"

    def test_rollback_failure_marks_worker_quarantined(self):
        """Rollback failure must escalate to WORKER_QUARANTINED."""
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING)
        sm.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        sm.transition(RuntimePhase.SWAPPING)
        sm.transition(RuntimePhase.ROLLING_BACK)
        sm.transition(RuntimePhase.WORKER_QUARANTINED, reason="checksum mismatch after rollback")
        assert sm.phase == RuntimePhase.WORKER_QUARANTINED
        assert sm.quarantine_scope == QuarantineScope.WORKER

    def test_worker_quarantine_rejects_future_requests(self):
        """Worker in quarantine cannot accept requests."""
        sm = RuntimeStateMachine()
        sm.transition(RuntimePhase.VERIFYING)
        sm.transition(RuntimePhase.SNAPSHOT_CAPTURED)
        sm.transition(RuntimePhase.SWAPPING)
        sm.transition(RuntimePhase.ROLLING_BACK)
        sm.transition(RuntimePhase.WORKER_QUARANTINED)
        assert not sm.can_accept_requests


# ═══════════════════════════════════════════════════════════════════════
# Failure Policy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFailurePolicy:
    """Tests for the failure policy decision engine."""

    def test_all_failure_classes_have_verdicts(self):
        """Every FailureClass must have a verdict in the policy table."""
        for fc in FailureClass:
            verdict = evaluate_failure(fc)
            assert verdict is not None
            assert verdict.http_status >= 400

    def test_rollback_failures_are_critical(self):
        """All rollback failures must be critical and quarantine worker."""
        rollback_failures = [
            FailureClass.ROLLBACK_CHECKSUM_MISMATCH,
            FailureClass.ROLLBACK_EXCEPTION,
            FailureClass.ROLLBACK_PARTIAL_RESTORE,
        ]
        for fc in rollback_failures:
            verdict = evaluate_failure(fc)
            assert verdict.is_critical, f"{fc} should be critical"
            assert verdict.quarantine_scope == QuarantineScope.WORKER
            assert verdict.action == RemediationAction.QUARANTINE_WORKER

    def test_payload_failures_quarantine_route(self):
        """Payload integrity failures must quarantine the route."""
        route_quarantine_failures = [
            FailureClass.PAYLOAD_SHA256_MISMATCH,
            FailureClass.PAYLOAD_NAN_INF_DETECTED,
            FailureClass.PAYLOAD_FILE_NOT_FOUND,
        ]
        for fc in route_quarantine_failures:
            verdict = evaluate_failure(fc)
            assert verdict.quarantine_scope == QuarantineScope.ROUTE

    def test_payload_too_large_does_not_quarantine(self):
        """Payload size limit does not quarantine — it's a per-request rejection."""
        verdict = evaluate_failure(FailureClass.PAYLOAD_TOO_LARGE)
        assert verdict.action == RemediationAction.REJECT_REQUEST
        assert verdict.quarantine_scope is None
        assert verdict.http_status == 413

    def test_healthz_unhealthy_after_rollback_failure(self):
        """Worker quarantine implies healthz should report unhealthy."""
        verdict = evaluate_failure(FailureClass.ROLLBACK_CHECKSUM_MISMATCH)
        assert verdict.target_phase == RuntimePhase.WORKER_QUARANTINED

    def test_audit_contains_critical_rollback_failure(self):
        """Critical failures must include descriptive messages for audit."""
        verdict = evaluate_failure(FailureClass.ROLLBACK_CHECKSUM_MISMATCH)
        assert "CRITICAL" in verdict.message
        assert verdict.is_critical


# ═══════════════════════════════════════════════════════════════════════
# Valid Payload Acceptance Test
# ═══════════════════════════════════════════════════════════════════════

class TestValidPayload:
    """Ensures valid payloads pass all validation checks."""

    def test_valid_payload_accepted(self, valid_payload):
        """A properly formed payload must pass validation."""
        route_data, _, _ = valid_payload
        deltas = validate_payload(route_data)
        assert len(deltas) == 1
        assert "model.layers.0.self_attn.q_proj.weight" in deltas

    def test_no_payload_block_returns_empty(self):
        """Routes without payload blocks return empty deltas (legacy support)."""
        route_data = {"layers": []}
        deltas = validate_payload(route_data)
        assert deltas == {}
