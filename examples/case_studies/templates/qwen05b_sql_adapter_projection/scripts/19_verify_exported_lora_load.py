import torch
import os
import json
import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

def verify_lora_load(model_id, lora_path):
    print(f"Base Model: {model_id}")
    print(f"Loading from: {lora_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load Base
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Attempt PEFT Load
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        
        print("\n[SUCCESS] PEFT Model loaded successfully!")
        print(f"Active Adapter: {model.active_adapter}")
        
        # Physical Weight Check: Ensure weights are not zero or uninitialized
        print("\nVerifying weight magnitudes...")
        lora_params = []
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                mean_val = p.abs().mean().item()
                max_val = p.abs().max().item()
                lora_params.append({"name": name, "mean": mean_val, "max": max_val})
                if max_val == 0:
                    print(f"  Warning: Parameter {name} is all ZEROS!")
        
        if not lora_params:
            print("  Warning: No LoRA parameters found in the model!")
        else:
            avg_magnitude = sum(p["mean"] for p in lora_params) / len(lora_params)
            print(f"  Detected {len(lora_params)} LoRA parameters. Average magnitude: {avg_magnitude:.6f}")
        
        # Verify specific layer keys
        state_dict = torch.load(Path(lora_path) / "adapter_model.bin", map_location="cpu")
        
        report = {
            "status": "LOAD_SUCCESS",
            "model_id": model_id,
            "lora_path": str(lora_path),
            "num_weights_in_file": len(state_dict),
            "num_params_in_model": len(lora_params),
            "avg_magnitude": avg_magnitude if lora_params else 0,
            "active_adapter": model.active_adapter,
            "weight_check": "SUCCESS" if lora_params and avg_magnitude > 0 else "WARNING_ZERO_WEIGHTS"
        }
        
    except Exception as e:
        print(f"\n[FAILURE] PEFT Load Failed: {str(e)}")
        report = {
            "status": "LOAD_FAILURE",
            "error": str(e)
        }

    output_report = Path(lora_path) / "load_verification_report.json"
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", default="routes/qwen05b_sql_projection/peft_lora")
    args = parser.parse_args()
    
    verify_lora_load(args.model_id, args.lora_path)
