"""
llama3_to_qwen2_port.py

A standalone example demonstrating the Neural-Scalpel Layer 2 I/O Adapter pipeline.
This script creates a mock Llama-3 LoRA in .safetensors format and ports it 
to Qwen-2 architecture using Soft-Routing Head Pooling (SRHP).
"""

import os
import sys
import torch
import tempfile
from safetensors.torch import save_file, load_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_scalpel.core.adapters import get_adapter

LLAMA_HIDDEN = 4096
LLAMA_HEADS = 32
QWEN_HIDDEN = 3584
QWEN_HEADS = 28

def create_mock_llama_lora(output_dir):
    print("[1/3] Generating Mock Llama-3 LoRA (safetensors)...")
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, LLAMA_HIDDEN),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(LLAMA_HIDDEN, 16),
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.randn(16, LLAMA_HIDDEN),
        "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.randn(LLAMA_HIDDEN, 16),
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.randn(16, LLAMA_HIDDEN),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.randn(LLAMA_HIDDEN, 16),
    }
    file_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, file_path)
    return file_path

def main():
    print("===============================================================")
    print(" 🚀 Neural-Scalpel Layer 2: LLaMA-3 to Qwen-2 Pipeline")
    print("===============================================================\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        target_dir = os.path.join(tmpdir, "target")
        
        # 1. Create source
        source_file = create_mock_llama_lora(source_dir)
        
        # 2. Porting using the Adapter
        print("\n[2/3] Porting through Layer 2 Adapters (Applying SRHP)...")
        adapter = get_adapter(
            "llama", "qwen", 
            (LLAMA_HIDDEN, LLAMA_HEADS), 
            (QWEN_HIDDEN, QWEN_HEADS)
        )
        
        state_dict = load_file(source_file)
        new_state_dict = {}
        
        for key, tensor in state_dict.items():
            print(f"  Mapping {key} | shape: {list(tensor.shape)}")
            new_key = adapter.map_key(key)
            new_tensor = adapter.project_tensor(key, tensor)
            new_state_dict[new_key] = new_tensor.contiguous()
            print(f"    -> Result: {list(new_tensor.shape)}")
            
        # 3. Save
        print("\n[3/3] Saving Ported Qwen-2 LoRA (safetensors)...")
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, "adapter_model.safetensors")
        save_file(new_state_dict, target_file)
        
        print("\n===============================================================")
        print(" [SUCCESS] LoRA successfully ported and serialized.")
        print(f" Output File: {target_file}")
        print("===============================================================")

if __name__ == '__main__':
    main()
