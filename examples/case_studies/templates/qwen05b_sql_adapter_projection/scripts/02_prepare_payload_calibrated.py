import os
import sys
import json
import torch
import torch.nn.functional as F
import math
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from safetensors.torch import load_file, save_file
from neural_scalpel.core.math import head_wise_orthogonal_procrustes, jacobian_tangent_space_alignment

# Helper from baseline
def infer_qwen_target_shape(module_name: str, config) -> list[int]:
    t_hidden = config.hidden_size
    t_intermediate = getattr(config, "intermediate_size", t_hidden * 4)
    t_num_heads = config.num_attention_heads
    t_num_kv_heads = getattr(config, "num_key_value_heads", t_num_heads)
    t_head_dim = t_hidden // t_num_heads
    t_kv_dim = t_num_kv_heads * t_head_dim
    if "mlp.down_proj" in module_name: return [t_hidden, t_intermediate]
    elif "mlp.gate_proj" in module_name or "mlp.up_proj" in module_name: return [t_intermediate, t_hidden]
    elif "self_attn.k_proj" in module_name or "self_attn.v_proj" in module_name: return [t_kv_dim, t_hidden]
    return [t_hidden, t_hidden]

def build_interpolated_layer_mapping(source_layers, target_layers):
    num_src = len(source_layers)
    mapping = {}
    for t_idx in range(target_layers):
        source_pos = t_idx * (num_src - 1) / (target_layers - 1)
        lower = math.floor(source_pos); upper = math.ceil(source_pos); alpha = source_pos - lower
        mapping[str(t_idx)] = {"lower": source_layers[lower], "upper": source_layers[upper], "alpha": round(alpha, 4)}
    return mapping

def fuse_qwen_layers(projected_delta):
    fused = {}; consumed = set()
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

def project_calibrated(hf_repo_id, target_model_id, calibration_path, output_dir, scale_gamma=1.0):
    print(f"\n[PHASE 4] Initializing Activation-Calibrated Projection...")
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    config_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_config.json")
    with open(config_path, "r") as f: adapter_config = json.load(f)
    target_config = AutoConfig.from_pretrained(target_model_id)
    
    # Load calibration data
    print(f"Loading calibration from {calibration_path}...")
    calibration_data = torch.load(calibration_path)
    
    # Load source weights
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="adapter_model.safetensors")
    lora_sd = load_file(weights_path)
    
    r = adapter_config.get("r", 8)
    alpha = adapter_config.get("lora_alpha", 16)
    scaling = (alpha / r) * scale_gamma
    
    src_layers = sorted(list(set([int(k.split(".layers.")[1].split(".")[0]) for k in lora_sd.keys() if ".layers." in k])))
    layer_mapping = build_interpolated_layer_mapping(src_layers, target_config.num_hidden_layers)
    
    projected_delta = {}; peft_lora_sd = {}
    
    print("Performing Layer-wise Calibrated Projection (JTSA)...")
    for t_idx, map_info in layer_mapping.items():
        s_low, s_up, alpha_w = map_info["lower"], map_info["upper"], map_info["alpha"]
        layer_key = f"layers.{t_idx}"
        
        # Calibration state for JTSA
        calib_entry = calibration_data.get(layer_key)
        calib_state = None
        if calib_entry is not None:
            # For JTSA, we use the manifold mean as a representative state
            calib_state = calib_entry["mean"].unsqueeze(0) # (1, target_hidden)
        
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
            if dW_up is None: dW_up = dW_low
            
            dW = (1.0 - alpha_w) * dW_low + alpha_w * dW_up
            clean_module = suffix.replace(".lora_A", "")
            target_shape = infer_qwen_target_shape(clean_module, target_config)
            
            # --- Activation-Calibrated Alignment (JTSA) ---
            # dW is (out_dim, in_dim). For simplicity, we first resize then align.
            # But the 'scalpel' way is to align the subspace.
            
            # 1. Resize to target shape (Naive Bilinear as starting point)
            t = dW.unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=target_shape, mode='bilinear', align_corners=False)
            t = t.squeeze(0).squeeze(0)
            
            # 2. Apply JTSA if calibration is available and it's a hidden-dimension projection
            if calib_state is not None and target_shape[1] == calib_state.shape[1]:
                # JTSA requires (N, d). We treat weight rows as samples.
                # This aligns the weight's input features to the calibration manifold.
                t_trans, _, _ = jacobian_tangent_space_alignment(
                    t, t, # Simplified self-alignment with Jacobian compensation
                    num_heads=target_config.num_attention_heads,
                    activation_states=calib_state
                )
                t = t_trans

            target_key = f"model.layers.{t_idx}.{clean_module}"
            projected_delta[target_key] = t.to(torch.float16)
            
            # PEFT export (via SVD on the calibrated weight)
            U, S, V = torch.svd(t.to(torch.float32))
            U_r, S_r, V_r = U[:, :r], S[:r], V[:, :r]
            peft_lora_sd[f"base_model.model.{target_key}.lora_B.weight"] = (U_r @ torch.diag(torch.sqrt(S_r))).to(torch.float16)
            peft_lora_sd[f"base_model.model.{target_key}.lora_A.weight"] = (torch.diag(torch.sqrt(S_r)) @ V_r.t()).to(torch.float16)

    # Finalize
    fused_delta = fuse_qwen_layers(projected_delta)
    save_file(fused_delta, str(out_path / "qwen05b_sql_projected_calibrated.safetensors"))
    
    peft_path = out_path / "peft_adapter_calibrated"
    peft_path.mkdir(exist_ok=True)
    save_file(peft_lora_sd, str(peft_path / "adapter_model.safetensors"))
    with open(peft_path / "adapter_config.json", "w") as f:
        json.dump({**adapter_config, "base_model_name_or_path": target_model_id}, f, indent=2)
    
    print(f"[SUCCESS] Phase 4 Calibrated Projection Complete: {peft_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_id", default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    parser.add_argument("--target_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--calibration", default="routes/qwen05b_sql_projection/calibration.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection")
    parser.add_argument("--scale-gamma", type=float, default=1.0)
    args = parser.parse_args()
    
    project_calibrated(args.lora_id, args.target_model, args.calibration, args.output_dir, args.scale_gamma)
