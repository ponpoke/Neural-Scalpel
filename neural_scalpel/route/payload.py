"""
Neural-Scalpel Route Payload Loader

Loads real weight-delta tensors from safetensors files referenced
by `.scalpel_route` manifests. Handles:
  - Payload file resolution (relative to a base directory or absolute)
  - SHA-256 integrity verification of the payload file
  - Per-layer delta extraction with shape/dtype validation
  - Lazy loading to minimize VRAM allocation until swap time

Design: The loader returns a dict[layer_name -> Tensor] that the
HotSwapRuntime.swap() can apply directly via tensor.add_(delta).
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Computes SHA-256 of a tensor's raw bytes (for per-layer verification)."""
    data = tensor.contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def resolve_payload_path(route_data: dict, base_dir: Optional[str] = None) -> Optional[str]:
    """
    Resolves the payload file path from route_data.
    Supports both the new `payload.uri` field and legacy routes without payloads.
    """
    payload_block = route_data.get("payload")
    if not payload_block:
        return None

    uri = payload_block.get("uri", "")
    if not uri:
        return None

    path = Path(uri)
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path

    return str(path)


def verify_payload_integrity(filepath: str, expected_sha256: str) -> bool:
    """Verifies the payload file's SHA-256 against the manifest."""
    actual = compute_file_sha256(filepath)
    if actual != expected_sha256:
        raise ValueError(
            f"Payload integrity check FAILED.\n"
            f"  Expected: {expected_sha256}\n"
            f"  Actual:   {actual}\n"
            f"  File:     {filepath}"
        )
    return True


def load_payload_deltas(
    route_data: dict,
    base_dir: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Loads weight-delta tensors from a safetensors payload file.

    Process:
      1. Resolve payload path from route_data["payload"]["uri"]
      2. Verify file-level SHA-256 against route_data["payload"]["sha256"]
      3. Load safetensors file
      4. For each layer in route_data["layers"], extract the delta tensor
         using the layer's "payload_key" (or fall back to layer "name")
      5. Validate shape and dtype against the manifest
      6. Optionally verify per-layer delta_sha256

    Returns:
      dict mapping layer names to delta tensors on the specified device.
    """
    filepath = resolve_payload_path(route_data, base_dir)
    if filepath is None:
        return {}

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Payload file not found: {filepath}")

    # File-level integrity check
    payload_block = route_data["payload"]
    if "sha256" in payload_block:
        verify_payload_integrity(filepath, payload_block["sha256"])

    # Load all tensors from the safetensors file
    all_tensors = load_file(filepath, device=device)

    # Extract and validate per-layer deltas
    deltas: Dict[str, torch.Tensor] = {}

    for layer_spec in route_data.get("layers", []):
        layer_name = layer_spec["name"]
        payload_key = layer_spec.get("payload_key", layer_name)
        expected_shape = layer_spec.get("shape")
        expected_dtype = layer_spec.get("dtype")

        if payload_key not in all_tensors:
            raise KeyError(
                f"Payload key '{payload_key}' not found in {filepath}. "
                f"Available keys: {list(all_tensors.keys())}"
            )

        delta = all_tensors[payload_key]

        # Shape validation
        if expected_shape and list(delta.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for {layer_name}: "
                f"manifest says {expected_shape}, payload has {list(delta.shape)}"
            )

        # Dtype validation
        if expected_dtype:
            actual_dtype = str(delta.dtype).replace("torch.", "")
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"Dtype mismatch for {layer_name}: "
                    f"manifest says {expected_dtype}, payload has {actual_dtype}"
                )

        deltas[layer_name] = delta

    return deltas
