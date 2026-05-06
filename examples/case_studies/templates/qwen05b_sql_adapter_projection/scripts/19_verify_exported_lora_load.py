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
        # This will fail if the config, key names, or shapes are incorrect
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        
        print("\n[SUCCESS] PEFT Model loaded successfully!")
        print(f"Active Adapter: {model.active_adapter}")
        
        # Verify specific layer keys
        state_dict = torch.load(Path(lora_path) / "adapter_model.bin", map_location="cpu")
        print(f"Number of weights in state_dict: {len(state_dict)}")
        
        report = {
            "status": "LOAD_SUCCESS",
            "model_id": model_id,
            "lora_path": str(lora_path),
            "num_weights": len(state_dict),
            "active_adapter": model.active_adapter
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
