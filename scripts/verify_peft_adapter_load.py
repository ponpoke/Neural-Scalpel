import os
import sys
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

    try:
        print(f"\n[PHASE 2] Loading Base Model (Meta/CPU)...")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        
        # Load base model (using CPU for smoke test to avoid GPU OOM or missing drivers)
        base_model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16,
            device_map="cpu", # Explicitly CPU for portability in CI/scripts
            trust_remote_code=True
        )

        print(f"\n[PHASE 3] Attaching Projected Adapter...")
        model = PeftModel.from_pretrained(base_model, str(peft_dir))
        
        print(f"\n[PHASE 4] Executing Single-Token Generation...")
        inputs = tokenizer("SELECT count(*) FROM users", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        
        token_out = tokenizer.decode(outputs[0, -1])
        print(f"  [SUCCESS] PEFT Adapter loaded and generated 1 token: '{token_out}'")
        
        return True
    except Exception as e:
        print(f"\n[ERROR] PEFT Load/Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-dir", required=True)
    parser.add_argument("--target-model", required=True)
    args = parser.parse_args()
    
    success = smoke_test_peft(args.peft_dir, args.target_model)
    sys.exit(0 if success else 1)
