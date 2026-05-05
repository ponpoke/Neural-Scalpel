import os
import sys
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.experimental.audit import AuditLogger

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = "Qwen/Qwen2.5-0.5B"
PAYLOAD_PATH = "routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors"
MANIFEST_PATH = "routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo.scalpel_route"
DATASET_PATH = "tests/alpaca_evaluation_dataset.json"
OUTPUT_PATH = "reports/phase_4b_quantitative_eval.json"

def calculate_score(output, reference):
    """Refined keyword-based correctness check."""
    out_lower = output.lower()
    ref_words = reference.lower().replace(",", "").replace(".", "").split()
    if not ref_words: return 0.0
    
    matches = 0
    for word in ref_words:
        # Check for presence of the reference word in output
        if word in out_lower:
            matches += 1
    
    return matches / len(ref_words) if ref_words else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Phase 4-B Quantitative Evaluation on {device}...")
    
    # 1. Load Model and Tokenizer
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    
    # 2. Setup Runtime
    signer = RouteSigner(secret_keys={"eval-key": "eval-secret"})
    registry = RouteRegistry(storage_dir="vllm_registry_storage", signer=signer)
    audit_logger = AuditLogger(log_file_path="reports/phase_4b_audit.jsonl")
    
    runtime = HotSwapRuntime(
        target_model=model,
        registry=registry,
        runtime_model_hash="qwen2.5-0.5b-hash",
        audit_logger=audit_logger
    )
    
    # 3. Load Dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    # 4. Load Actual Payload and Un-fuse
    from safetensors.torch import load_file
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    
    print("Un-fusing vLLM payload for Transformers evaluation...")
    vllm_deltas = load_file(PAYLOAD_PATH)
    transformers_deltas = {}
    for k, v in vllm_deltas.items():
        if "gate_up_proj" in k:
            gate, up = v.chunk(2, dim=0)
            transformers_deltas[k.replace("gate_up_proj", "gate_proj")] = gate
            transformers_deltas[k.replace("gate_up_proj", "up_proj")] = up
        elif "qkv_proj" in k:
            q_part, k_part, v_part = v.split([896, 128, 128], dim=0)
            transformers_deltas[k.replace("qkv_proj", "q_proj")] = q_part
            transformers_deltas[k.replace("qkv_proj", "k_proj")] = k_part
            transformers_deltas[k.replace("qkv_proj", "v_proj")] = v_part
        else:
            transformers_deltas[k] = v
            
    transformers_manifest = manifest.copy()
    new_layers = []
    for layer in manifest["layers"]:
        name = layer["name"]
        if "gate_up_proj" in name:
            new_layers.append({"name": name.replace("gate_up_proj", "gate_proj"), "shape": [4864, 896], "dtype": "float16"})
            new_layers.append({"name": name.replace("gate_up_proj", "up_proj"), "shape": [4864, 896], "dtype": "float16"})
        elif "qkv_proj" in name:
            new_layers.append({"name": name.replace("qkv_proj", "q_proj"), "shape": [896, 896], "dtype": "float16"})
            new_layers.append({"name": name.replace("qkv_proj", "k_proj"), "shape": [128, 896], "dtype": "float16"})
            new_layers.append({"name": name.replace("qkv_proj", "v_proj"), "shape": [128, 896], "dtype": "float16"})
        else:
            new_layers.append(layer)
    transformers_manifest["layers"] = new_layers
    
    # 5. Evaluation Loop
    modes = ["Base", "Neural-Scalpel (Alpaca)", "Rollback"]
    results = {mode: [] for mode in modes}
    
    for mode in modes:
        print(f"\nEvaluating mode: {mode}...")
        
        if mode == "Neural-Scalpel (Alpaca)":
            runtime.capture_snapshot(transformers_manifest)
            runtime.apply_swap(transformers_manifest, transformers_deltas)
        elif mode == "Rollback":
            runtime.rollback()
            
        for item in dataset:
            prompt = item["instruction"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    do_sample=False, # GREEDY FOR DETERMINISM
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            score = calculate_score(generated_text, item["reference"])
            
            results[mode].append({
                "instruction": prompt,
                "output": generated_text,
                "score": score
            })
            
    # 6. Aggregate Results
    summary = {}
    for mode, data in results.items():
        avg_score = sum(d["score"] for d in data) / len(data)
        summary[mode] = {
            "avg_score": avg_score,
            "data": data
        }
        
    # 7. Final Report
    print("\n" + "="*60)
    print(" PHASE 4-B PRELIMINARY QUANTITATIVE SMOKE EVALUATION")
    print(" NOTE: This keyword-overlap score is not a dataset-level task-improvement proof.")
    print("="*60)
    for mode in modes:
        print(f"{mode:25} Score: {summary[mode]['avg_score']:.4f}")
    
    improvement = summary["Neural-Scalpel (Alpaca)"]["avg_score"] - summary["Base"]["avg_score"]
    print("-" * 60)
    print(f"Score Delta vs Base:   {improvement:+.4f}")
    
    if improvement <= 0:
        print("Task Improvement:      NOT PROVEN under this metric")
    else:
        print("Task Improvement:      preliminary positive signal under this metric")

    rb_delta = abs(summary["Rollback"]["avg_score"] - summary["Base"]["avg_score"])
    print(f"Rollback Consistency:  {'PASS' if rb_delta < 1e-6 else 'FAIL'} (Delta: {rb_delta:.6f})")
    print("="*60)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
