import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def smoke_test_peft(peft_dir: str, target_model_id: str):
    print(f"\n[PHASE 1] Initializing PEFT Smoke Test...")
    print(f"  Target Model: {target_model_id}")
    print(f"  Adapter Dir: {peft_dir}")

    peft_dir = Path(peft_dir)
    if not (peft_dir / "adapter_model.safetensors").exists():
        print(f"[ERROR] Adapter file not found in {peft_dir}")
        return False

    report = {
        "status": "FAIL",
        "validation_type": "peft_load_one_token_generation_smoke",
        "target_model": target_model_id,
        "adapter_dir": str(peft_dir),
        "generated_token": None,
        "does_not_validate": [
            "task quality",
            "SQL correctness",
            "long-form generation stability"
        ]
    }

    try:
        print(f"\n[PHASE 2] Loading Base Model (Meta/CPU)...")
        tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )

        print(f"\n[PHASE 3] Attaching Projected Adapter...")
        model = PeftModel.from_pretrained(base_model, str(peft_dir))
        
        print(f"\n[PHASE 4] Executing Single-Token Generation...")
        inputs = tokenizer("SELECT count(*) FROM users", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        
        token_out = tokenizer.decode(outputs[0, -1])
        print(f"  [SUCCESS] PEFT Adapter loaded and generated 1 token: '{token_out}'")
        
        report["status"] = "PASS"
        report["generated_token"] = token_out
        
    except Exception as e:
        print(f"\n[ERROR] PEFT Load/Inference failed: {e}")
        report["error"] = str(e)

    report_path = peft_dir / "smoke_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")
    return report["status"] == "PASS"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-dir", required=True)
    parser.add_argument("--target-model", required=True)
    args = parser.parse_args()
    
    success = smoke_test_peft(args.peft_dir, args.target_model)
    sys.exit(0 if success else 1)
