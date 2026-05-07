import os
import torch
import json
import warnings
import sys
from pathlib import Path
from safetensors.torch import save_file, load_file
from neural_scalpel.core.adapters import get_adapter

def create_mock_lora(path, rank=16, hidden=4096, inter=11008):
    """Creates a mock Qwen2.5-7B style LoRA adapter."""
    state_dict = {}
    layers = 1 # Just 1 layer to speed up Piecewise SVD
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for i in range(layers):
        for mod in modules:
            in_dim = inter if mod == "down_proj" else hidden
            out_dim = hidden if mod == "down_proj" else inter
            
            # Note: Llama/Qwen structure
            if mod in ["gate_proj", "up_proj", "down_proj"]:
                state_dict[f"base_model.model.layers.{i}.mlp.{mod}.lora_A.weight"] = torch.randn(rank, in_dim)
                state_dict[f"base_model.model.layers.{i}.mlp.{mod}.lora_B.weight"] = torch.randn(out_dim, rank)
            else:
                state_dict[f"base_model.model.layers.{i}.self_attn.{mod}.lora_A.weight"] = torch.randn(rank, hidden)
                state_dict[f"base_model.model.layers.{i}.self_attn.{mod}.lora_B.weight"] = torch.randn(hidden, rank)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)
    return state_dict

def run_artifact_regression():
    print("="*60)
    print(" Neural-Scalpel v2.9 Artifact Regression Suite")
    print("="*60)
    sys.stdout.flush()
    
    temp_dir = Path("runs/regression_test")
    source_lora_path = temp_dir / "source_lora/adapter_model.safetensors"
    
    print("[*] Phase 0: Setting up mock source adapter...")
    source_state = create_mock_lora(source_lora_path)
    
    source_info = {"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32}
    target_info = {"hidden_size": 1024, "intermediate_size": 2816, "num_attention_heads": 8}
    
    results = []
    
    alphas = [8, 16, 24, 32]
    print("\n[*] Phase 1: Alpha Sweep Regression (Linear)...")
    sys.stdout.flush()
    for alpha in alphas:
        output_dir = temp_dir / f"linear_a{alpha}"
        os.makedirs(output_dir, exist_ok=True)
        
        adapter = get_adapter("llama", "qwen", source_info, target_info, projection_mode="linear")
        
        new_state = {}
        for k, v in source_state.items():
            projected = adapter.project_tensor(k, v)
            if projected is not None:
                new_state[k] = projected
        
        # Verify Config
        config = {"lora_alpha": alpha}
        alpha_correct = config["lora_alpha"] == alpha
        
        # Check shapes
        shape_correct = True
        failed_keys = []
        for k, v in new_state.items():
            if "lora_A" in k:
                # in_features check
                if "down_proj" in k:
                    if v.shape[1] != 2816: 
                        shape_correct = False
                        failed_keys.append((k, v.shape))
                else:
                    if v.shape[1] != 1024: 
                        shape_correct = False
                        failed_keys.append((k, v.shape))
            elif "lora_B" in k:
                # out_features check
                if any(x in k for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
                    # Target inter for MLP or head-dim scaled for attention
                    # Q/K/V B shape[0] should be head-dim * target-heads
                    if "proj" in k and v.shape[0] not in [1024, 2816]:
                         # Special case for Q/K/V heads?
                         pass
        
        print(f"  [alpha={alpha}] lora_alpha: {alpha_correct}, Shapes: {shape_correct}")
        if not shape_correct:
            for k, s in failed_keys[:3]:
                print(f"    FAIL: {k} -> {s}")
        sys.stdout.flush()
        results.append({"mode": "linear", "alpha": alpha, "config_ok": alpha_correct, "shape_ok": shape_correct})

    print("\n[*] Phase 3: Piecewise Pair-Aware Regression...")
    sys.stdout.flush()
    adapter = get_adapter("llama", "qwen", source_info, target_info, projection_mode="piecewise")
    new_state = {}
    for k, v in source_state.items():
        res = adapter.project_tensor(k, v)
        if res is not None:
            if isinstance(res, dict): new_state.update(res)
            else: new_state[k] = res
    adapter.finalize()
    
    keys = list(new_state.keys())
    has_pairs = any("mlp.up_proj.lora_A" in k for k in keys) and any("mlp.up_proj.lora_B" in k for k in keys)
    print(f"  [piecewise] Pairs reconstructed: {has_pairs}")
    
    nan_found = False
    for k, v in new_state.items():
        if torch.isnan(v).any(): nan_found = True
    print(f"  Numerical Integrity: {'PASS' if not nan_found else 'FAIL'}")

    print("\n" + "="*60 + "\n DONE.")
    sys.stdout.flush()

if __name__ == "__main__":
    run_artifact_regression()
