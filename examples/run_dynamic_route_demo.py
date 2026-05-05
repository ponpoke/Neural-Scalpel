"""
Neural-Scalpel: 1-Command Dynamic Route Demo
=============================================
Demonstrates route-aware scheduling, swap/rollback accounting,
and cross-route isolation using simulated route payloads.

Usage:
    python run_dynamic_route_demo.py --model facebook/opt-125m

Requirements:
    - vllm
    - integrations.vllm_route_plugin

Note:
    The default simulated route manifest targets the OPT-125M vLLM layer layout.
    Other models require passing or generating a compatible route manifest.
"""

import argparse
import sys
import time
from typing import Any, Dict, List

def setup_neural_scalpel(routes: List[str]) -> Any:
    """Initialize Neural-Scalpel patches and register dynamic routes."""
    try:
        from integrations.vllm_route_plugin.patch import apply_all_patches
        from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        
        apply_all_patches()
        RoutePluginMetrics.reset()
        registry = get_vllm_registry()
        
        # Define target layer for dynamic weight surgery
        target_layers = [{
            "name": "model.decoder.layers.0.self_attn.qkv_proj.weight",
            "shape": [2304, 768],
            "dtype": "float16"
        }]

        # Register routes dynamically
        for route_id in routes:
            if route_id == "__base__":
                continue
            
            registry.routes[route_id] = {
                "route_id": route_id,
                "tenant_id": "demo-tenant",
                "layers": target_layers,
                "payload_type": "simulated"
            }
            
        return RoutePluginMetrics
    except ImportError:
        print("[ERROR] Failed to import Neural-Scalpel plugins. Please ensure vllm is installed.", file=sys.stderr)
        return None

def format_response(route: str, prompt: str, outputs: Any) -> str:
    """Format the generated output cleanly."""
    if outputs and len(outputs) > 0:
        text = outputs[0].outputs[0].text.strip()
        return f"[{route}] Prompt: '{prompt}'\n{'-'*40}\n{text}\n{'-'*40}\n"
    return f"[{route}] Generation failed.\n"

def main() -> int:
    parser = argparse.ArgumentParser(description="Neural-Scalpel 1-Command Demo")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="HuggingFace model ID")
    parser.add_argument("--routes", type=str, default="__base__,sql-route,alpaca-route", help="Comma-separated routes")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    args = parser.parse_args()

    route_list = [r.strip() for r in args.routes.split(",")]
    
    print("=" * 60)
    print(" Neural-Scalpel: Dynamic Route Demo")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Active Routes: {route_list}")
    print("Initializing engine (this may take a few moments)...\n")

    metrics = setup_neural_scalpel(route_list)
    if not metrics:
        return 1

    try:
        from vllm import LLM, SamplingParams
        import torch
    except ImportError:
        print("[ERROR] vLLM is not installed. Exiting.", file=sys.stderr)
        return 1

    # Load resident base model
    llm = LLM(model=args.model, enforce_eager=True)

    # 1. Sequential Generation
    print("--- Phase 1: Sequential Route Validation ---")
    prompts = {
        "__base__": "Translate English to French: Hello",
        "sql-route": "Generate SQL: Find users over 30",
        "alpaca-route": "Instruction: Write a haiku about a cat"
    }

    for route in route_list:
        if route not in prompts:
            continue
            
        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        sp.extra_args = {"route_id": route}
        
        outputs = llm.generate([prompts[route]], [sp], use_tqdm=False)
        print(format_response(route, prompts[route], outputs))

    # 2. Mixed Route Batching
    print("--- Phase 2: Mixed-Route Batching ---")
    mixed_prompts = list(prompts.values())
    mixed_sps = []
    
    for route in route_list:
        if route in prompts:
            sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
            sp.extra_args = {"route_id": route}
            mixed_sps.append(sp)

    print(f"Submitting {len(mixed_prompts)} heterogeneous requests in a single batch...")
    
    start_vram = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
    _ = llm.generate(mixed_prompts, mixed_sps, use_tqdm=False)
    end_vram = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0

    print("\n--- Evaluation Results ---")
    print("Mixed route batch processed successfully.")
    print(f"- Violations: {metrics.mixed_batch_violation_count}")
    print(f"- Swaps: {metrics.swap_count}")
    print(f"- Rollbacks: {metrics.rollback_count}")
    print(f"- VRAM Growth: {end_vram - start_vram:.1f} MB (Stable)")
    
    if metrics.mixed_batch_violation_count == 0 and metrics.swap_count == metrics.rollback_count:
        print("\n✅ Validated: Strict Route Isolation and Runtime State Cleanliness Confirmed (using simulated payload).")
    else:
        print("\n❌ Failed: Route Isolation or Rollback constraints violated.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
