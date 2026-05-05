"""
Generates a dummy safetensors payload for OPT-125m Phase 7G validation.
"""
import torch
from safetensors.torch import save_file
import os

def generate_test_payload():
    # Target: model.decoder.layers.0.self_attn.qkv_proj.weight [2304, 768]
    # We generate a tiny delta (e.g. all 1e-5) to verify it loads correctly
    delta = torch.full((2304, 768), 1e-5, dtype=torch.float16)
    
    tensors = {
        "model.decoder.layers.0.self_attn.qkv_proj.weight": delta
    }
    
    output_dir = "vllm_registry_storage/payloads"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "opt125m_sql_delta.safetensors")
    save_file(tensors, output_path)
    print(f"[Phase 7G] Generated test payload: {output_path}")

if __name__ == "__main__":
    generate_test_payload()
