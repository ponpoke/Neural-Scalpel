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

def project_peft_lora(hf_repo_id: str, output_dir: str):
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
    
    # PEFT names are like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # We want Neural-Scalpel names: model.layers.0.self_attn.q_proj.weight
    
    projected_delta = {}
    lora_A_keys = [k for k in lora_sd.keys() if "lora_A" in k]
    
    for key_A in lora_A_keys:
        key_B = key_A.replace("lora_A", "lora_B")
        if key_B not in lora_sd:
            print(f"Warning: Missing {key_B} for {key_A}")
            continue
            
        A = lora_sd[key_A]  # shape: (r, in_features)
        B = lora_sd[key_B]  # shape: (out_features, r)
        
        # Project: Delta_W = B @ A * scaling
        delta_W = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
        
        # Map key back to base model parameter name
        # e.g., base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # -> model.layers.0.self_attn.q_proj.weight
        clean_key = key_A.replace("base_model.model.", "").replace(".lora_A", "")
        projected_delta[clean_key] = delta_W.to(torch.float16)  # save back as fp16
        
    print(f"Projected {len(projected_delta)} weight tensors.")
    
    # Save payload
    payload_name = f"{hf_repo_id.split('/')[-1]}_payload.safetensors"
    payload_path = out_path / payload_name
    save_file(projected_delta, str(payload_path))
    
    # Hash it
    with open(payload_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    # Create route manifest
    route_id = hf_repo_id.split("/")[-1].lower()
    manifest = {
        "route_id": route_id,
        "tenant_id": "eval-tenant",
        "description": f"Actual projected LoRA from {hf_repo_id}",
        "license_mode": "OPEN",
        "payload": {
            "type": "safetensors",
            "uri": f"file://{payload_path.resolve().as_posix()}",
            "sha256": sha256
        },
        "payload_key": "custom"
    }
    
    manifest_path = out_path / f"{route_id}.scalpel_route"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved payload to {payload_path}")
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo = sys.argv[1]
    else:
        repo = "onurerkan/qwen2.5-0.5b-alpaca-lora-demo" # default example
    project_peft_lora(repo, "routes/actual_loras")
