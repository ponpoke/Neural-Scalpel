import os
import json
import argparse
from pathlib import Path
import torch

def inspect_adapter(adapter_path, target_model_id, real_check=False):
    mode = "REAL" if real_check else "SIMULATED"
    print(f"[Phase 1] Inspecting adapter ({mode}): {adapter_path}")
    
    if real_check:
        if not os.path.exists(adapter_path):
            print(f"Error: Adapter file not found at {adapter_path}")
            return None
            
        try:
            from safetensors.torch import load_file
            tensors = load_file(adapter_path, device="cpu")
            keys = list(tensors.keys())
            
            def infer_target_module(key: str) -> str | None:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part.startswith("lora_A") or part.startswith("lora_B"):
                        if i > 0:
                            return parts[i - 1]
                return None

            modules = []
            ranks = []
            for k in keys:
                m = infer_target_module(k)
                if m:
                    modules.append(m)
                
                # Check if any part starts with lora_A for rank detection
                if any(part.startswith("lora_A") for part in k.split(".")):
                    ranks.append(tensors[k].shape[0])
            
            detected_rank = max(ranks) if ranks else "unknown"
            
            inspection_results = {
                "adapter_path": adapter_path,
                "target_model_id": target_model_id,
                "mode": "REAL",
                "lora_rank": detected_rank,
                "num_tensors": len(keys),
                "target_modules": sorted(list(set(modules))),
                "status": "COMPLETED"
            }
        except Exception as e:
            print(f"Error during real inspection: {e}")
            return None
    else:
        inspection_results = {
            "adapter_path": adapter_path,
            "target_model_id": target_model_id,
            "mode": "SIMULATED",
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "status": "SCAFFOLD",
            "note": "This is a SIMULATED inspection. No real files were read."
        }
    
    report_path = Path("reports/source_adapter_inspection.json")
    os.makedirs(report_path.parent, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(inspection_results, f, indent=2)
        
    print(f"Inspection results saved to {report_path}")
    return inspection_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Perform real safetensors inspection")
    parser.add_argument("--adapter", default="path/to/source/adapter.safetensors")
    parser.add_argument("--target", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()

    inspect_adapter(args.adapter, args.target, real_check=args.real)
