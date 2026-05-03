"""
verify_real_safetensors.py

A validation script that downloads a tiny, real .safetensors model from Hugging Face
and runs it through the Neural-Scalpel Layer 2 Adapter pipeline to prove
it works on actual production formatted files.

Usage:
    python examples/verify_real_safetensors.py
"""

import os
import sys
import torch
import tempfile
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_scalpel.core.adapters import get_adapter

def main():
    print("===============================================================")
    print(" 🚀 Neural-Scalpel: Real .safetensors Verification")
    print("===============================================================\n")

    # We will use a very tiny model's safetensors file just to prove the I/O pipeline
    repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    filename = "model.safetensors"
    
    print(f"[1/3] Downloading real safetensors from HF Hub ({repo_id})...")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"      -> Downloaded successfully to: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download real safetensors: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        target_dir = os.path.join(tmpdir, "target")
        os.makedirs(target_dir, exist_ok=True)
        
        # Load the real state dict
        print("\n[2/3] Loading and Porting using Layer 2 Adapters...")
        state_dict = load_file(model_path)
        
        # We will pretend to port this tiny Llama (hidden_size 16, heads 4) 
        # to a tiny Qwen (hidden_size 12, heads 2)
        adapter = get_adapter("llama", "qwen", (16, 4), (12, 2))
        
        new_state_dict = {}
        for key, tensor in state_dict.items():
            # For demonstration, only process a few layers to keep output clean
            if "layers.0" in key and "weight" in key:
                print(f"  Mapping {key} | original shape: {list(tensor.shape)}")
                new_key = adapter.map_key(key)
                new_tensor = adapter.project_tensor(key, tensor)
                new_state_dict[new_key] = new_tensor.contiguous()
                print(f"    -> Result: {list(new_tensor.shape)}")
            else:
                # Keep other tensors as is for this test
                new_state_dict[key] = tensor

        # Save to prove safetensors serialization works
        print("\n[3/3] Saving Ported Safetensors...")
        target_file = os.path.join(target_dir, "ported_model.safetensors")
        save_file(new_state_dict, target_file)
        
        # Verify the saved file is loadable
        test_load = load_file(target_file)
        
        print("\n===============================================================")
        print(" [SUCCESS] Real .safetensors file loaded, projected, and saved.")
        print(f" Output File: {target_file} (Loadable: {len(test_load.keys())} keys)")
        print("===============================================================")

if __name__ == "__main__":
    main()
