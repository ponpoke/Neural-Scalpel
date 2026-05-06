import os
import sys
import json
import torch
import torch.nn.functional as F
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from safetensors.torch import load_file, save_file

def fuse_qwen_layers(projected_delta: dict) -> dict:
    """Fuses Qwen-style separate projections into vLLM's fused layers."""
    fused = {}
    consumed = set()
    for key, tensor in projected_delta.items():
        if ".mlp.gate_proj.weight" in key:
            up_key = key.replace(".mlp.gate_proj.weight", ".mlp.up_proj.weight")
            fused_key = key.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight")
            if up_key in projected_delta:
                fused[fused_key] = torch.cat([tensor, projected_delta[up_key]], dim=0)
                consumed.add(key); consumed.add(up_key)
                print(f"  [FUSE] MLP: {fused_key}")
        if ".self_attn.q_proj.weight" in key:
            k_key = key.replace(".self_attn.q_proj.weight", ".self_attn.k_proj.weight")
            v_key = key.replace(".self_attn.q_proj.weight", ".self_attn.v_proj.weight")
            fused_key = key.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight")
            if k_key in projected_delta and v_key in projected_delta:
                fused[fused_key] = torch.cat([tensor, projected_delta[k_key], projected_delta[v_key]], dim=0)
                consumed.add(key); consumed.add(k_key); consumed.add(v_key)
                print(f"  [FUSE] ATTN: {fused_key}")
    for key, tensor in projected_delta.items():
        if key not in consumed:
            if any(x in key for x in [".gate_proj", ".up_proj", ".q_proj", ".k_proj", ".v_proj"]): continue
            fused[key] = tensor
    return fused

def resize_and_recompress(tensor: torch.Tensor, target_shape: list, rank: int = 16) -> torch.Tensor:
    """
    Resizes a weight matrix and optionally recompresses it using SVD 
    to maintain low-rank characteristics in target shape.
    """
    if list(tensor.shape) == target_shape:
        return tensor
    
    # Structural Resizing (Bilinear)
    # (Out, In) -> (1, 1, Out, In)
    t = tensor.unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
    t = t.squeeze(0).squeeze(0)

    # Low-rank Recompression via SVD
    # W approx = U @ S @ V^T
    try:
        U, S, V = torch.svd(t)
        # Keep top rank singular values
        U_r = U[:, :rank]
        S_r = torch.diag(S[:rank])
        V_r = V[:, :rank]
        t_recompressed = U_r @ S_r @ V_r.t()
        return t_recompressed.to(torch.float16)
    except Exception as e:
        print(f"    [WARN] SVD failed for {target_shape}, falling back to raw resize: {e}")
        return t.to(torch.float16)

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024 * 64) -> str:
    path = Path(path); h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def sha256_tensor(tensor: torch.Tensor, chunk_size: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256(); arr = tensor.detach().cpu().contiguous().numpy()
    view = memoryview(arr).cast("B")
    for i in range(0, len(view), chunk_size): h.update(view[i : i + chunk_size])
    return h.hexdigest()

def project_peft_lora(hf_repo_id: str, output_dir: str, target_model_id: str = None):
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PHASE 1] Initializing Structural Projection Baseline...")
    config_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_config.json")
    with open(config_path, "r") as f: adapter_config = json.load(f)
    
    source_model_id = adapter_config.get("base_model_name_or_path", "unknown")
    if target_model_id is None: target_model_id = source_model_id
    
    print(f"  Source Model: {source_model_id}")
    print(f"  Target Model: {target_model_id}")
    
    # Load target architecture details
    print(f"  Fetching target configuration (AutoConfig)...")
    target_config = AutoConfig.from_pretrained(target_model_id)
    t_hidden = target_config.hidden_size
    t_layers = target_config.num_hidden_layers
    t_intermediate = getattr(target_config, "intermediate_size", t_hidden * 4)
    
    # GQA Awareness
    t_num_heads = target_config.num_attention_heads
    t_num_kv_heads = getattr(target_config, "num_key_value_heads", t_num_heads)
    t_head_dim = t_hidden // t_num_heads
    t_kv_dim = t_num_kv_heads * t_head_dim
    
    print(f"  Target Blueprint: {t_layers}L, {t_hidden}H (KV_Dim: {t_kv_dim}), {t_intermediate}MLP")

    r, alpha = adapter_config.get("r", 8), adapter_config.get("lora_alpha", 16)
    scaling = alpha / r

    print(f"\n[PHASE 2] Harvesting Knowledge from Source LoRA...")
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_model.safetensors")
    lora_sd = load_file(weights_path)
    source_adapter_sha256 = sha256_file(weights_path)

    # Layer mapping (Uniform sampling baseline)
    src_layers = sorted(list(set([int(k.split(".layers.")[1].split(".")[0]) for k in lora_sd.keys() if ".layers." in k])))
    num_src = len(src_layers)
    layer_map_log = {}
    
    projected_delta = {}
    print(f"  Executing Structural Resizing + SVD Recompression...")
    for t_idx in range(t_layers):
        # Uniform sampling
        s_idx = src_layers[int(t_idx * (num_src - 1) / (t_layers - 1))] if num_src > 0 else 0
        layer_map_log[str(t_idx)] = s_idx
        
        layer_keys = [k for k in lora_sd.keys() if f".layers.{s_idx}." in k and "lora_A" in k]
        for k_A in layer_keys:
            k_B = k_A.replace("lora_A", "lora_B")
            if k_B not in lora_sd: continue
            
            A, B = lora_sd[k_A], lora_sd[k_B]
            delta_W = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
            
            # Module-specific shape identification (GQA-aware)
            module_name = k_A.split(".layers.")[1].split(".", 1)[1].replace(".lora_A", "")
            target_key = f"model.layers.{t_idx}.{module_name}"
            
            if "mlp.down_proj" in module_name: 
                t_shape = [t_hidden, t_intermediate]
            elif "mlp.gate_proj" in module_name or "mlp.up_proj" in module_name: 
                t_shape = [t_intermediate, t_hidden]
            elif "self_attn.k_proj" in module_name or "self_attn.v_proj" in module_name:
                t_shape = [t_kv_dim, t_hidden]
            elif "self_attn.q_proj" in module_name or "self_attn.o_proj" in module_name:
                t_shape = [t_hidden, t_hidden]
            else:
                t_shape = [t_hidden, t_hidden] # Fallback
            
            projected_delta[target_key] = resize_and_recompress(delta_W, t_shape, rank=r)

    print(f"\n[PHASE 3] Finalizing Dense Payload (vLLM Fusion)...")
    projected_delta = fuse_qwen_layers(projected_delta)

    # Save
    payload_name = f"{hf_repo_id.split('/')[-1]}_projected.safetensors"
    payload_path = out_path / payload_name
    print(f"\n[PHASE 4] Saving Rank-Limited Dense Payload...")
    save_file(projected_delta, str(payload_path))
    
    payload_size = payload_path.stat().st_size
    sha256 = sha256_file(payload_path)
    
    # Manifest creation with strict honesty
    route_id = hf_repo_id.split("/")[-1].lower()
    layers_manifest = []
    for name, tensor in projected_delta.items():
        layers_manifest.append({
            "name": name, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "delta_sha256": sha256_tensor(tensor)
        })

    manifest = {
        "route_schema_version": "1.0.0",
        "route_id": route_id,
        "tenant_id": "eval-tenant",
        "description": f"Experimental structural projection from {hf_repo_id} to {target_model_id} architecture.",
        "evaluation_only": True,
        "license": "UNVERIFIED",
        "projection_method": "structural_bilinear_svd_recompression_baseline",
        "payload_type": "rank_limited_full_matrix_delta_dense",
        "target_shape_validation": {
            "status": "CONFIG_DERIVED",
            "note": "Payload tensors were resized using target AutoConfig dimensions (GQA-aware). Runtime state_dict shape validation is still required."
        },
        "layer_mapping": {
            "strategy": "uniform_source_layer_sampling",
            "source_layers": num_src,
            "target_layers": t_layers,
            "mapping": layer_map_log
        },
        "compression": {
            "method": "svd_reconstruct_full_matrix",
            "rank": r,
            "note": "SVD rank constraint was applied before saving, but tensors are stored as dense full matrices."
        },
        "source_model": source_model_id,
        "target_model": target_model_id,
        "source_adapter_sha256": source_adapter_sha256,
        "payload": {
            "format": "safetensors",
            "uri": str(payload_path.resolve()),
            "sha256": sha256,
            "size_bytes": payload_size,
            "hash_method": "sha256_streaming_chunked"
        },
        "diagnostics": {"verdict": "NOT_EVALUATED", "note": "Structural projection baseline; behavioral performance unknown."},
        "layers": layers_manifest,
        "signature": {"algorithm": "none", "key_id": "eval-only", "value": "unsigned"}
    }
    
    with open(out_path / f"{route_id}.scalpel_route", "w") as f: json.dump(manifest, f, indent=2)
    print(f"\n[SUCCESS] Baseline Projection Complete.")
    print(f"  Payload: {payload_path}")
    print(f"  Final Size: {payload_size / 1024 / 1024:.2f} MB")
    print(f"  Method: structural_bilinear_svd_recompression_baseline")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_id", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output_dir", default="routes/projected")
    args = parser.parse_args()
    project_peft_lora(args.lora_id, args.output_dir, args.target_model)
