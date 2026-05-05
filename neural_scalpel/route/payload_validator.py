"""
Neural-Scalpel Payload Validator

Pre-swap validation of safetensors payloads. All checks run BEFORE any
weight modification occurs, enforcing the fail-close invariant:
  - If validation fails, NO swap happens
  - The route is quarantined
  - The request receives a 503

Checks performed:
  1. File existence and readability
  2. File size limit enforcement
  3. SHA-256 integrity verification
  4. Safetensors header parse (detects corruption)
  5. Per-tensor key presence
  6. Per-tensor shape validation
  7. Per-tensor dtype validation
  8. NaN / Inf detection
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from neural_scalpel.experimental.failure_policy import FailureClass
from neural_scalpel.route.payload import (
    resolve_payload_path,
    compute_file_sha256,
)


# Default maximum payload size: 2 GB
DEFAULT_MAX_PAYLOAD_BYTES: int = 2 * 1024 * 1024 * 1024


class PayloadValidationError(Exception):
    """Raised when payload validation fails. Carries the failure class for policy lookup."""

    def __init__(self, failure_class: FailureClass, detail: str):
        self.failure_class = failure_class
        self.detail = detail
        super().__init__(f"[{failure_class.value}] {detail}")


def validate_payload(
    route_data: dict,
    base_dir: Optional[str] = None,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Dict[str, torch.Tensor]:
    """
    Performs exhaustive pre-swap validation of a route's payload.

    Returns:
        Dict mapping layer names to validated delta tensors, ready for swap.

    Raises:
        PayloadValidationError with the appropriate FailureClass on any failure.
    """
    # ── Step 1: Resolve and check file existence ───────────────
    filepath = resolve_payload_path(route_data, base_dir)
    if filepath is None:
        # No payload block means legacy/simulated route — skip validation
        return {}

    path = Path(filepath)
    if not path.exists():
        raise PayloadValidationError(
            FailureClass.PAYLOAD_FILE_NOT_FOUND,
            f"Payload file does not exist: {filepath}",
        )

    # ── Step 2: File size limit ────────────────────────────────
    try:
        file_size = path.stat().st_size
    except OSError as e:
        raise PayloadValidationError(
            FailureClass.PAYLOAD_IO_ERROR,
            f"Cannot stat payload file: {e}",
        )

    if file_size > max_payload_bytes:
        raise PayloadValidationError(
            FailureClass.PAYLOAD_TOO_LARGE,
            f"Payload size {file_size} exceeds limit {max_payload_bytes}",
        )

    # ── Step 3: SHA-256 integrity ──────────────────────────────
    payload_block = route_data.get("payload", {})
    expected_sha = payload_block.get("sha256")
    if expected_sha:
        try:
            actual_sha = compute_file_sha256(filepath)
        except OSError as e:
            raise PayloadValidationError(
                FailureClass.PAYLOAD_IO_ERROR,
                f"I/O error computing SHA-256: {e}",
            )
        if actual_sha != expected_sha:
            raise PayloadValidationError(
                FailureClass.PAYLOAD_SHA256_MISMATCH,
                f"Expected SHA-256: {expected_sha}, actual: {actual_sha}",
            )

    # ── Step 4: Parse safetensors (detects header corruption) ──
    try:
        from safetensors.torch import load_file
        all_tensors = load_file(filepath, device="cpu")
    except Exception as e:
        err_msg = str(e).lower()
        if "header" in err_msg or "invalid" in err_msg or "corrupt" in err_msg:
            raise PayloadValidationError(
                FailureClass.PAYLOAD_HEADER_CORRUPTED,
                f"Safetensors header parse error: {e}",
            )
        raise PayloadValidationError(
            FailureClass.PAYLOAD_IO_ERROR,
            f"Failed to load safetensors file: {e}",
        )

    # ── Step 5-8: Per-tensor validation ────────────────────────
    deltas: Dict[str, torch.Tensor] = {}
    layers: List[dict] = route_data.get("layers", [])

    for layer_spec in layers:
        layer_name = layer_spec["name"]
        payload_key = layer_spec.get("payload_key", layer_name)
        expected_shape = layer_spec.get("shape")
        expected_dtype = layer_spec.get("dtype")

        # Step 5: Key presence
        if payload_key not in all_tensors:
            raise PayloadValidationError(
                FailureClass.PAYLOAD_TENSOR_KEY_MISSING,
                f"Key '{payload_key}' not in payload. Available: {list(all_tensors.keys())}",
            )

        delta = all_tensors[payload_key]

        # Step 6: Shape validation
        if expected_shape and list(delta.shape) != expected_shape:
            raise PayloadValidationError(
                FailureClass.PAYLOAD_SHAPE_MISMATCH,
                f"Layer '{layer_name}': expected shape {expected_shape}, got {list(delta.shape)}",
            )

        # Step 7: Dtype validation
        if expected_dtype:
            actual_dtype = str(delta.dtype).replace("torch.", "")
            if actual_dtype != expected_dtype:
                raise PayloadValidationError(
                    FailureClass.PAYLOAD_DTYPE_MISMATCH,
                    f"Layer '{layer_name}': expected dtype {expected_dtype}, got {actual_dtype}",
                )

        # Step 8: NaN / Inf detection
        if torch.isnan(delta).any():
            raise PayloadValidationError(
                FailureClass.PAYLOAD_NAN_INF_DETECTED,
                f"Layer '{layer_name}': NaN values detected in delta tensor",
            )
        if torch.isinf(delta).any():
            raise PayloadValidationError(
                FailureClass.PAYLOAD_NAN_INF_DETECTED,
                f"Layer '{layer_name}': Inf values detected in delta tensor",
            )

        deltas[layer_name] = delta

    return deltas
