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

def infer_qwen_target_shape(module_name: str, config) -> list[int]:
    """Infers the target shape for a given module name and model config (GQA-aware)."""
    t_hidden = config.hidden_size
    t_intermediate = getattr(config, "intermediate_size", t_hidden * 4)
    t_num_heads = config.num_attention_heads
    t_num_kv_heads = getattr(config, "num_key_value_heads", t_num_heads)
    t_head_dim = t_hidden // t_num_heads
    t_kv_dim = t_num_kv_heads * t_head_dim

    if "mlp.down_proj" in module_name: 
        return [t_hidden, t_intermediate]
    elif "mlp.gate_proj" in module_name or "mlp.up_proj" in module_name: 
        return [t_intermediate, t_hidden]
    elif "self_attn.k_proj" in module_name or "self_attn.v_proj" in module_name:
        return [t_kv_dim, t_hidden]
    return [t_hidden, t_hidden]

def build_interpolated_layer_mapping(source_layers: list[int], target_layers: int) -> dict:
    """Builds a weighted linear interpolation map for layer folding."""
    num_src = len(source_layers)
    mapping = {}
    for t_idx in range(target_layers):
        source_pos = t_idx * (num_src - 1) / (target_layers - 1)
        lower = math.floor(source_pos)
        upper = math.ceil(source_pos)
        alpha = source_pos - lower
        mapping[str(t_idx)] = {
            "lower": source_layers[lower],
            "upper": source_layers[upper],
            "alpha": round(alpha, 4)
        }
    return mapping

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
        if ".self_attn.q_proj.weight" in key:
            k_key = key.replace(".self_attn.q_proj.weight", ".self_attn.k_proj.weight")
            v_key = key.replace(".self_attn.q_proj.weight", ".self_attn.v_proj.weight")
            fused_key = key.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight")
            if k_key in projected_delta and v_key in projected_delta:
                fused[fused_key] = torch.cat([tensor, projected_delta[k_key], projected_delta[v_key]], dim=0)
                consumed.add(key); consumed.add(k_key); consumed.add(v_key)
    for key, tensor in projected_delta.items():
        if key not in consumed:
            if any(x in key for x in [".gate_proj", ".up_proj", ".q_proj", ".k_proj", ".v_proj"]): continue
            fused[key] = tensor
    return fused

def resize_and_analyze(tensor: torch.Tensor, target_shape: list, rank: int = 16):
    """Resizes a matrix and analyze loss via SVD. Returns (reconstructed, stats, svd_data)."""
    t = tensor.unsqueeze(0).unsqueeze(0).to(torch.float32)
    t = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
    t = t.squeeze(0).squeeze(0)

    try:
        U, S, V = torch.svd(t)
        energy_total = torch.sum(S ** 2)
        U_r, S_r, V_r = U[:, :rank], S[:rank], V[:, :rank]
        energy_kept = torch.sum(S_r ** 2)
        retention = (energy_kept / energy_total).item() if energy_total > 0 else 1.0
        t_re = U_r @ torch.diag(S_r) @ V_r.t()
        stats = {"energy_retention": retention, "max_abs": t_re.abs().max().item()}
        return t_re.to(torch.float16), stats, (U_r, S_r, V_r)
    except Exception:
        return t.to(torch.float16), {"energy_retention": 1.0, "max_abs": t.abs().max().item()}, None

def sha256_file(path: str | Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def project_peft_lora(hf_repo_id: str, output_dir: str, target_model_id: str, export_peft: bool = False, scale_gamma: float = 1.0):
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[PHASE 1] Initializing Structural Projection Baseline v2...")
    if os.path.exists(hf_repo_id):
        config_path = os.path.join(hf_repo_id, "adapter_config.json")
        weights_path = os.path.join(hf_repo_id, "adapter_model.safetensors")
    else:
        config_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_config.json")
        weights_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_model.safetensors")
    
    with open(config_path, "r") as f: adapter_config = json.load(f)
    
    target_config = AutoConfig.from_pretrained(target_model_id)
    r, alpha = adapter_config.get("r", 8), adapter_config.get("lora_alpha", 16)
    scaling = (alpha / r) * scale_gamma

    print(f"\n[PHASE 2] Interpolating Layers...")
    lora_sd = load_file(weights_path)
    src_layers = sorted(list(set([int(k.split(".layers.")[1].split(".")[0]) for k in lora_sd.keys() if ".layers." in k])))
    layer_mapping = build_interpolated_layer_mapping(src_layers, target_config.num_hidden_layers)
    
    projected_delta = {}; peft_lora_sd = {}; all_stats = []

    for t_idx, map_info in layer_mapping.items():
        s_low, s_up, alpha_w = map_info["lower"], map_info["upper"], map_info["alpha"]
        low_keys = [k for k in lora_sd.keys() if f".layers.{s_low}." in k and "lora_A" in k]
        for k_A_low in low_keys:
            suffix = k_A_low.split(f".layers.{s_low}.")[1]
            k_B_low = k_A_low.replace("lora_A", "lora_B")
            k_A_up = k_A_low.replace(f".layers.{s_low}.", f".layers.{s_up}.")
            k_B_up = k_A_up.replace("lora_A", "lora_B")
            
            def get_dW(ka, kb):
                if ka in lora_sd and kb in lora_sd:
                    return (lora_sd[kb].to(torch.float32) @ lora_sd[ka].to(torch.float32)) * scaling
                return None

            dW_low = get_dW(k_A_low, k_B_low)
            if dW_low is None: continue
            
            dW_up = get_dW(k_A_up, k_B_up)
            if dW_up is None: dW_up = dW_low # Fallback to low if up missing

            dW = (1.0 - alpha_w) * dW_low + alpha_w * dW_up
            clean_module = suffix.replace(".lora_A", "").replace(".weight", "")
            target_shape = infer_qwen_target_shape(clean_module, target_config)
            re_t, stats, svd = resize_and_analyze(dW, target_shape, rank=r)
            
            target_key = f"model.layers.{t_idx}.{clean_module}"
            projected_delta[target_key] = re_t
            all_stats.append(stats)
            
            if export_peft and svd:
                U_r, S_r, V_r = svd
                peft_lora_sd[f"base_model.model.{target_key}.lora_B.weight"] = (U_r @ torch.diag(torch.sqrt(S_r))).to(torch.float16)
                peft_lora_sd[f"base_model.model.{target_key}.lora_A.weight"] = (torch.diag(torch.sqrt(S_r)) @ V_r.t()).to(torch.float16)

    print(f"\n[PHASE 3] Finalizing Payloads...")
    fused_delta = fuse_qwen_layers(projected_delta)
    payload_path = out_path / f"{hf_repo_id.split('/')[-1]}_projected.safetensors"
    save_file(fused_delta, str(payload_path))
    
    if export_peft:
        peft_path = out_path / "peft_adapter"
        peft_path.mkdir(exist_ok=True)
        save_file(peft_lora_sd, str(peft_path / "adapter_model.safetensors"))
        with open(peft_path / "adapter_config.json", "w") as f:
            json.dump({**adapter_config, "base_model_name_or_path": target_model_id}, f, indent=2)
        with open(peft_path / "projection_metadata.json", "w") as f:
            json.dump({
                "projection_method": "structural_bilinear_svd_recompression_v2",
                "scale_gamma": scale_gamma,
                "source_adapter": hf_repo_id,
                "target_model": target_model_id,
                "behavioral_validation": "PENDING"
            }, f, indent=2)

    manifest = {
        "route_schema_version": "1.0.0", "route_id": hf_repo_id.split("/")[-1].lower(),
        "projection_method": "structural_bilinear_svd_recompression_v2",
        "scale_gamma": scale_gamma,
        "target_shape_validation": {"status": "CONFIG_DERIVED", "note": "Use verify_target_runtime_shapes.py for RUNTIME_SHAPE_VERIFIED."},
        "layer_mapping": {"strategy": "linear_interpolated_layer_folding", "mapping": layer_mapping},
        "compression_stats": {"mean_energy_retention": sum(s["energy_retention"] for s in all_stats)/len(all_stats)},
        "payload": {"uri": str(payload_path.resolve()), "sha256": sha256_file(payload_path)},
        "diagnostics": {"verdict": "NOT_EVALUATED", "note": "Behavioral transfer pending."}
    }
    with open(out_path / f"{manifest['route_id']}.scalpel_route", "w") as f: json.dump(manifest, f, indent=2)
    print(f"[SUCCESS] Baseline v2 Complete. Retention: {manifest['compression_stats']['mean_energy_retention']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_id", required=True); parser.add_argument("--target-model", required=True)
    parser.add_argument("--output_dir", default="routes/projected"); parser.add_argument("--export-peft", action="store_true")
    parser.add_argument("--scale-gamma", type=float, default=1.0)
    args = parser.parse_args()
    project_peft_lora(args.lora_id, args.output_dir, args.target_model, args.export_peft, args.scale_gamma)
