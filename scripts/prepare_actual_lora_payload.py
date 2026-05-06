import os
import sys
import json
import torch
import torch.nn.functional as F
import hashlib
import math
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

def resize_and_analyze(tensor: torch.Tensor, target_shape: list, rank: int = 16):
    """
    Resizes a weight matrix and analyzes the information loss via SVD.
    Returns (reconstructed_tensor, stats, (U_r, S_r, V_r))
    """
    # Structural Resizing (Bilinear)
    t = tensor.unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
    t = t.squeeze(0).squeeze(0)

    try:
        U, S, V = torch.svd(t)
        energy_total = torch.sum(S ** 2)
        
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:, :rank]
        
        energy_kept = torch.sum(S_r ** 2)
        retention = (energy_kept / energy_total).item() if energy_total > 0 else 1.0
        
        # Reconstruction
        t_re = U_r @ torch.diag(S_r) @ V_r.t()
        
        stats = {
            "energy_retention": retention,
            "max_abs": t_re.abs().max().item(),
            "mean_abs": t_re.abs().mean().item(),
            "std": t_re.std().item()
        }
        return t_re.to(torch.float16), stats, (U_r, S_r, V_r)
    except Exception as e:
        print(f"    [WARN] SVD analysis failed: {e}")
        return t.to(torch.float16), {"energy_retention": 1.0, "max_abs": t.abs().max().item()}, None

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

def project_peft_lora(hf_repo_id: str, output_dir: str, target_model_id: str, export_peft: bool = False, scale_gamma: float = 1.0):
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PHASE 1] Initializing Structural Projection Baseline v2...")
    config_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_config.json")
    with open(config_path, "r") as f: adapter_config = json.load(f)
    
    source_model_id = adapter_config.get("base_model_name_or_path", "unknown")
    print(f"  Source Model: {source_model_id}")
    print(f"  Target Model: {target_model_id}")
    print(f"  Scale Gamma: {scale_gamma}")
    
    target_config = AutoConfig.from_pretrained(target_model_id)
    t_hidden = target_config.hidden_size
    t_layers = target_config.num_hidden_layers
    t_intermediate = getattr(target_config, "intermediate_size", t_hidden * 4)
    t_num_heads = target_config.num_attention_heads
    t_num_kv_heads = getattr(target_config, "num_key_value_heads", t_num_heads)
    t_head_dim = t_hidden // t_num_heads
    t_kv_dim = t_num_kv_heads * t_head_dim
    print(f"  Target Blueprint: {t_layers}L, {t_hidden}H (KV_Dim: {t_kv_dim}), {t_intermediate}MLP")

    r, alpha = adapter_config.get("r", 8), adapter_config.get("lora_alpha", 16)
    scaling = (alpha / r) * scale_gamma

    print(f"\n[PHASE 2] Harvesting & Interpolating Knowledge...")
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_model.safetensors")
    lora_sd = load_file(weights_path)
    source_adapter_sha256 = sha256_file(weights_path)

    src_layers = sorted(list(set([int(k.split(".layers.")[1].split(".")[0]) for k in lora_sd.keys() if ".layers." in k])))
    num_src = len(src_layers)
    
    projected_delta = {}
    peft_lora_sd = {}
    layer_map_log = {}
    all_stats = []

    for t_idx in range(t_layers):
        # Linear Interpolated Layer Folding
        source_pos = t_idx * (num_src - 1) / (t_layers - 1)
        lower = math.floor(source_pos)
        upper = math.ceil(source_pos)
        alpha_w = source_pos - lower
        
        s_low = src_layers[lower]
        s_up = src_layers[upper]
        
        layer_map_log[str(t_idx)] = {"lower": s_low, "upper": s_up, "alpha": round(alpha_w, 4)}
        
        low_keys = [k for k in lora_sd.keys() if f".layers.{s_low}." in k and "lora_A" in k]
        for k_A_low in low_keys:
            module_suffix = k_A_low.split(f".layers.{s_low}.")[1]
            k_B_low = k_A_low.replace("lora_A", "lora_B")
            k_A_up = k_A_low.replace(f".layers.{s_low}.", f".layers.{s_up}.")
            k_B_up = k_A_up.replace("lora_A", "lora_B")
            
            def get_delta(ka, kb):
                if ka in lora_sd and kb in lora_sd:
                    return (lora_sd[kb].to(torch.float32) @ lora_sd[ka].to(torch.float32)) * scaling
                return None

            dW_low = get_delta(k_A_low, k_B_low)
            dW_up = get_delta(k_A_up, k_B_up) if k_A_up in lora_sd else dW_low
            
            if dW_low is None: continue
            delta_W = (1.0 - alpha_w) * dW_low + alpha_w * dW_up
            
            clean_module = module_suffix.replace(".lora_A", "")
            target_key = f"model.layers.{t_idx}.{clean_module}"
            
            if "mlp.down_proj" in clean_module: t_shape = [t_hidden, t_intermediate]
            elif "mlp.gate_proj" in clean_module or "mlp.up_proj" in clean_module: t_shape = [t_intermediate, t_hidden]
            elif "self_attn.k_proj" in clean_module or "self_attn.v_proj" in clean_module: t_shape = [t_kv_dim, t_hidden]
            else: t_shape = [t_hidden, t_hidden]
            
            re_t, stats, svd_data = resize_and_analyze(delta_W, t_shape, rank=r)
            projected_delta[target_key] = re_t
            all_stats.append(stats)
            
            if export_peft and svd_data:
                U_r, S_r, V_r = svd_data
                peft_lora_sd[f"base_model.model.{target_key}.lora_B.weight"] = (U_r @ torch.diag(torch.sqrt(S_r))).to(torch.float16)
                peft_lora_sd[f"base_model.model.{target_key}.lora_A.weight"] = (torch.diag(torch.sqrt(S_r)) @ V_r.t()).to(torch.float16)

    print(f"\n[PHASE 3] Finalizing Payloads...")
    fused_delta = fuse_qwen_layers(projected_delta)
    
    energy_ret = [s["energy_retention"] for s in all_stats]
    summary_stats = {
        "mean_energy_retention": sum(energy_ret) / len(energy_ret),
        "min_energy_retention": min(energy_ret),
        "global_max_abs": max([s["max_abs"] for s in all_stats]),
        "nan_detected": any([math.isnan(s["max_abs"]) for s in all_stats])
    }

    payload_name = f"{hf_repo_id.split('/')[-1]}_projected.safetensors"
    payload_path = out_path / payload_name
    save_file(fused_delta, str(payload_path))
    
    if export_peft:
        peft_path = out_path / "peft_adapter"
        peft_path.mkdir(exist_ok=True)
        save_file(peft_lora_sd, str(peft_path / "adapter_model.safetensors"))
        with open(peft_path / "adapter_config.json", "w") as f:
            target_adapter_config = adapter_config.copy()
            target_adapter_config["base_model_name_or_path"] = target_model_id
            json.dump(target_adapter_config, f, indent=2)
        print(f"  [EXPORT] PEFT-style adapter export generated at {peft_path}")

    route_id = hf_repo_id.split("/")[-1].lower()
    manifest = {
        "route_schema_version": "1.0.0",
        "route_id": route_id,
        "description": "Baseline v2: GQA-aware structural interpolation + SVD recompression.",
        "projection_method": "structural_bilinear_svd_recompression_v2",
        "payload_type": "rank_limited_full_matrix_delta_dense",
        "scale_gamma": scale_gamma,
        "target_shape_validation": {
            "status": "CONFIG_DERIVED",
            "note": "Validated against AutoConfig. Use verify_target_runtime_shapes.py for RUNTIME_SHAPE_VERIFIED status."
        },
        "layer_mapping": {
            "strategy": "linear_interpolated_layer_folding",
            "source_layers": num_src, "target_layers": t_layers, "mapping": layer_map_log
        },
        "compression_stats": summary_stats,
        "source_model": source_model_id, "target_model": target_model_id,
        "payload": {
            "format": "safetensors", "uri": str(payload_path.resolve()),
            "sha256": sha256_file(payload_path), "size_bytes": payload_path.stat().st_size
        },
        "diagnostics": {"verdict": "NOT_EVALUATED", "note": "Structural projection baseline; behavioral performance pending."}
    }
    
    with open(out_path / f"{route_id}.scalpel_route", "w") as f: json.dump(manifest, f, indent=2)
    print(f"\n[SUCCESS] Baseline v2 Complete. Size: {payload_path.stat().st_size/1024/1024:.2f} MB")
    print(f"  Mean SVD Energy Retention: {summary_stats['mean_energy_retention']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_id", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output_dir", default="routes/projected")
    parser.add_argument("--export-peft", action="store_true")
    parser.add_argument("--scale-gamma", type=float, default=1.0)
    args = parser.parse_args()
    project_peft_lora(args.lora_id, args.output_dir, args.target_model, args.export_peft, args.scale_gamma)
