import argparse
import time
import os
import json
import logging
from vllm import LLM, SamplingParams
from integrations.vllm_route_plugin.patch import apply_all_patches
from integrations.vllm_route_plugin.runtime_context import get_vllm_runtime

def main():
    parser = argparse.ArgumentParser(description="Verify Alpaca LoRA Improvement")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--output", type=str, default="reports/qualitative_alpaca_smoke_check.json")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # 1. Apply patches BEFORE LLM init
    apply_all_patches()

    # Load manifest
    manifest_path = args.payload.replace("_payload.safetensors", ".scalpel_route")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Ensure output directory exists
    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-populate registry
    from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
    from neural_scalpel.route.policy import RouteStatus
    
    registry = get_vllm_registry()
    registry.routes["alpaca"] = manifest
    registry.statuses["alpaca"] = RouteStatus.PRODUCTION
    print(f"Registered route 'alpaca' in runtime registry.")

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a time-traveling toaster.",
        "Give me a step-by-step recipe for a simple chocolate cake.",
        "Write a python function to calculate the nth Fibonacci number.",
        "What are the three laws of thermodynamics?"
    ]

    # 2. Run Base Generation
    print("\n--- Running Base Generation ---")
    sampling_params_base = SamplingParams(temperature=0.7, max_tokens=256)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        dtype=args.dtype
    )
    
    results_base = llm.generate(prompts, sampling_params_base)
    
    # 3. Run Scalpel Generation (with Alpaca Route)
    print("\n--- Running Scalpel Generation (Alpaca Route) ---")
    sampling_params_scalpel = SamplingParams(temperature=0.7, max_tokens=256)
    # Ensure extra_args exists and set the route_id
    sampling_params_scalpel.extra_args = {"route_id": "alpaca"}
    
    results_scalpel = llm.generate(prompts, sampling_params_scalpel)

    print("\n" + "="*80)
    print(" PHASE 4: QUALITATIVE REAL-LORA ROUTE SMOKE CHECK")
    print(" NOTE: This shows route-specific output changes, not dataset-level task improvement.")
    print("="*80)

    comparison_results = []

    for i in range(len(prompts)):
        base_out = results_base[i].outputs[0].text.strip()
        scalpel_out = results_scalpel[i].outputs[0].text.strip()
        
        print(f"\nPROMPT {i+1}: {prompts[i]}")
        print("-" * 40)
        print("ROUTE: [__base__]")
        print(f"OUTPUT:\n{base_out}")
        print("-" * 40)
        print("ROUTE: [alpaca]")
        print(f"OUTPUT:\n{scalpel_out}")
        print("="*80)

        comparison_results.append({
            "prompt": prompts[i],
            "base_output": base_out,
            "alpaca_output": scalpel_out,
            "route_id": "alpaca"
        })

    # Save results to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Qualitative results saved to: {output_path}")

if __name__ == "__main__":
    main()
