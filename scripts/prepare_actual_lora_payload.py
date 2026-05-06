"""
Prepare Actual Trained LoRA Payload for Neural-Scalpel

This script downloads a PEFT LoRA adapter from HuggingFace, 
projects the low-rank matrices (lora_B @ lora_A * scaling) into 
full-rank weight deltas, and packages them as a Neural-Scalpel 
`.scalpel_route` with a `.safetensors` payload.
"""

import os
import sys
import json
import torch
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

def fuse_qwen_layers(projected_delta: dict) -> dict:
    """
    Fuses Qwen-style separate projections into vLLM's fused layers.

    vLLM tested representation:
    - gate_proj + up_proj -> gate_up_proj
    - q_proj + k_proj + v_proj -> qkv_proj

    This is implemented as a two-pass transform so the result does not
    depend on safetensors/dict key iteration order.
    """
    fused = {}
    consumed = set()

    # Pass 1: create fused tensors.
    for key, tensor in projected_delta.items():
        # MLP Fusion: gate_proj + up_proj -> gate_up_proj
        if ".mlp.gate_proj.weight" in key:
            up_key = key.replace(".mlp.gate_proj.weight", ".mlp.up_proj.weight")
            fused_key = key.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight")

            if up_key in projected_delta:
                gate = tensor
                up = projected_delta[up_key]

                if list(gate.shape) != list(up.shape):
                    raise ValueError(
                        f"MLP fusion shape mismatch: {key}={list(gate.shape)}, "
                        f"{up_key}={list(up.shape)}"
                    )

                fused[fused_key] = torch.cat([gate, up], dim=0)
                consumed.add(key)
                consumed.add(up_key)
                print(f"  [FUSE] MLP: {key} + {up_key} -> {fused_key}")
            continue

        # Attention Fusion: q_proj + k_proj + v_proj -> qkv_proj
        if ".self_attn.q_proj.weight" in key:
            k_key = key.replace(".self_attn.q_proj.weight", ".self_attn.k_proj.weight")
            v_key = key.replace(".self_attn.q_proj.weight", ".self_attn.v_proj.weight")
            fused_key = key.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight")

            if k_key in projected_delta and v_key in projected_delta:
                q = tensor
                k = projected_delta[k_key]
                v = projected_delta[v_key]

                if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
                    raise ValueError(
                        f"QKV fusion input-dim mismatch: {key}={list(q.shape)}, "
                        f"{k_key}={list(k.shape)}, {v_key}={list(v.shape)}"
                    )

                fused[fused_key] = torch.cat([q, k, v], dim=0)
                consumed.add(key)
                consumed.add(k_key)
                consumed.add(v_key)
                print(f"  [FUSE] ATTN: {key} + {k_key} + {v_key} -> {fused_key}")
            continue

    # Pass 2: keep only non-consumed tensors.
    for key, tensor in projected_delta.items():
        if key in consumed:
            continue

        # Safety: do not keep standalone fused-source projections if their group was incomplete.
        # They do not exist in the tested vLLM Qwen representation.
        if (
            ".mlp.gate_proj.weight" in key
            or ".mlp.up_proj.weight" in key
            or ".self_attn.q_proj.weight" in key
            or ".self_attn.k_proj.weight" in key
            or ".self_attn.v_proj.weight" in key
        ):
            print(f"  [SKIP] Unfused source projection not kept: {key}")
            continue

        fused[key] = tensor

    return fused

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024 * 64) -> str:
    """Calculates SHA256 in chunks to avoid MemoryError on large payloads."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sha256_tensor(tensor: torch.Tensor, chunk_size: int = 64 * 1024 * 1024) -> str:
    """Calculates SHA256 of a tensor in chunks to avoid memory issues with .tobytes()."""
    h = hashlib.sha256()
    # Ensure it's on CPU and contiguous before hashing
    arr = tensor.detach().cpu().contiguous().numpy()
    view = memoryview(arr).cast("B")
    for i in range(0, len(view), chunk_size):
        h.update(view[i : i + chunk_size])
    return h.hexdigest()

def project_peft_lora(hf_repo_id: str, output_dir: str, target_model: str = None):
    """
    Downloads adapter_config.json and adapter_model.safetensors.
    Projects to full rank and saves as scalpel payload.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading LoRA config from {hf_repo_id}...")
    config_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    r = config.get("r", 8)
    alpha = config.get("lora_alpha", 16)
    scaling = alpha / r
    print(f"LoRA Rank: {r}, Alpha: {alpha}, Scaling: {scaling}")
    
    print(f"Downloading LoRA weights from {hf_repo_id}...")
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_model.safetensors")
    lora_sd = load_file(weights_path)
    
    # Calculate source adapter hash for manifest
    source_adapter_sha256 = sha256_file(weights_path)

    base_model = config.get("base_model_name_or_path", "unknown")
    
    # Map key back to base model parameter name
    projected_delta = {}
    lora_A_keys = [k for k in lora_sd.keys() if "lora_A" in k]
    
    print("\n[INFO] Starting Key Mapping...")
    for key_A in lora_A_keys:
        key_B = key_A.replace("lora_A", "lora_B")
        if key_B not in lora_sd:
            print(f"  [WARN] Missing {key_B} for {key_A}. Skipping.")
            continue
            
        A = lora_sd[key_A]
        B = lora_sd[key_B]
        
        # Project: Delta_W = B @ A * scaling
        delta_W = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
        
        clean_key = key_A
        if clean_key.startswith("base_model.model.model."):
            clean_key = clean_key.replace("base_model.model.model.", "model.", 1)
        elif clean_key.startswith("base_model.model."):
            clean_key = clean_key.replace("base_model.model.", "", 1)
        
        clean_key = clean_key.replace(".lora_A", "")
        if not clean_key.endswith(".weight") and key_A.endswith(".weight"):
            clean_key += ".weight"
            
        projected_delta[clean_key] = delta_W.to(torch.float16)
        
    if not projected_delta:
        raise ValueError("[ERROR] No weight tensors were successfully projected.")

    # Apply vLLM-specific fusion
    print("\n[INFO] Applying vLLM layer fusion (gate_up, qkv)...")
    projected_delta = fuse_qwen_layers(projected_delta)

    # Save payload
    payload_name = f"{hf_repo_id.split('/')[-1]}_payload.safetensors"
    payload_path = out_path / payload_name
    print(f"\n[INFO] Saving payload to {payload_path}...")
    save_file(projected_delta, str(payload_path))
    
    # Hash and Metadata
    payload_size_bytes = payload_path.stat().st_size
    sha256 = sha256_file(payload_path)
    chunk_size = 1024 * 1024 * 64
        
    # Create route manifest
    route_id = hf_repo_id.split("/")[-1].lower()
    layers_manifest = []
    for name, tensor in projected_delta.items():
        # Use chunked tensor hashing to avoid MemoryError
        layer_hash = sha256_tensor(tensor)
        layers_manifest.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "delta_sha256": layer_hash
        })

    manifest = {
        "route_schema_version": "1.0.0",
        "route_id": route_id,
        "tenant_id": "eval-tenant",
        "description": f"Evaluation-only projected LoRA from {hf_repo_id}.",
        "evaluation_only": True,
        "license": "UNVERIFIED",
        "projection_method": "peft_lora_to_full_rank_delta",
        "target_shape_validation": {
            "status": "PENDING",
            "note": (
                "Generated payload tensor shapes have not yet been validated "
                "against the target model runtime state_dict. For cross-scale "
                "source adapters, additional shape projection may be required."
            )
        },
        "source_model": base_model,
        "target_model": target_model or base_model,
        "source_adapter_sha256": source_adapter_sha256,
        "target_model_sha256": "0" * 64,
        "payload": {
            "format": "safetensors",
            "uri": str(payload_path.resolve()),
            "sha256": sha256,
            "size_bytes": payload_size_bytes,
            "hash_method": "sha256_streaming_chunked",
            "hash_chunk_size_bytes": chunk_size
        },
        "diagnostics": {
            "verdict": "NOT_EVALUATED",
            "ppl_degradation": None,
            "kl_divergence": None,
            "note": "Payload generation completed; downstream diagnostics not evaluated."
        },
        "layers": layers_manifest,
        "signature": {"algorithm": "none", "key_id": "eval-only", "value": "unsigned"}
    }
    
    manifest_path = out_path / f"{route_id}.scalpel_route"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\n[SUCCESS] Payload saved: {payload_path}")
    print(f"  Size: {payload_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Hash: {sha256}")
    print(f"  Method: sha256_streaming_chunked (64MB chunks)")
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Actual LoRA Payload")
    parser.add_argument("--lora_id", type=str, default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    parser.add_argument("--output_dir", type=str, default="routes/actual_loras")
    parser.add_argument("--target-model", type=str, default=None, help="Target model for the manifest (defaults to base_model)")
    args = parser.parse_args()
    
    project_peft_lora(args.lora_id, args.output_dir, args.target_model)
