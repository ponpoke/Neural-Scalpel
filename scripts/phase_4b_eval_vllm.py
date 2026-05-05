import argparse
import time
import os
import json
import logging

# MUST APPLY PATCHES BEFORE VLLM IMPORTS
from integrations.vllm_route_plugin.patch import apply_all_patches
apply_all_patches()

from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase4B")

def calculate_similarity(output, reference):
    """Simple token-level overlap similarity."""
    out_tokens = set(output.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens: return 0.0
    overlap = out_tokens.intersection(ref_tokens)
    return len(overlap) / len(ref_tokens)

def main():
    parser = argparse.ArgumentParser(description="Phase 4-B preliminary route-behavior smoke evaluation (single mode)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--dataset", type=str, default="tests/alpaca_evaluation_dataset.json")
    parser.add_argument("--output_dir", type=str, default="reports/phase_4b")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--mode", type=str, choices=["base", "scalpel"], required=True)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    prompts = [item["instruction"] for item in dataset]
    references = [item["reference"] for item in dataset]

    # Initialize vLLM
    if args.mode == "scalpel":
        from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
        from neural_scalpel.route.policy import RouteStatus
        
        # Load manifest
        manifest_path = args.payload.replace("_payload.safetensors", ".scalpel_route")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        # PERSIST TO REGISTRY STORAGE (so EngineCore can see it)
        registry = get_vllm_registry()
        registry_path = os.path.join(registry.storage_dir, "alpaca.scalpel_route")
        os.makedirs(registry.storage_dir, exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(manifest, f)
        
        # Mark status as PRODUCTION so it's allowed
        # Note: statuses might still be process-local, but the registry check often allows PRODUCTION by default if no status file is found?
        # Actually, let's check how statuses are persisted.
        # For this prototype, the registry.routes dictionary is checked.
    
    # Force V1 engine just in case
    os.environ["VLLM_USE_V1"] = "1"
    
    llm = LLM(model=args.model, enforce_eager=True, dtype=args.dtype)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    if args.mode == "scalpel":
        sampling_params.extra_args = {"route_id": "alpaca"}

    logger.info(f"Running [{args.mode}] Evaluation...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Calculate scores
    results_list = []
    for i, (out, ref) in enumerate(zip(outputs, references)):
        text = out.outputs[0].text
        score = calculate_similarity(text, ref)
        results_list.append({
            "instruction": prompts[i],
            "output": text,
            "reference": ref,
            "score": score
        })

    avg_score = sum(r["score"] for r in results_list) / len(results_list)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"results_{args.mode}.json")
    with open(output_file, "w") as f:
        json.dump({
            "mode": args.mode,
            "avg_score": avg_score,
            "results": results_list
        }, f, indent=2)

    print(f"\n[{args.mode}] Avg Score: {avg_score:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
