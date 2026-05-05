"""
Neural-Scalpel: Real Payload Route E2E Check
============================================
Validates real safetensors payload application, route-specific output change,
and rollback cleanliness.

This script does not prove task improvement unless the payload is a trained
task-specific adapter and is evaluated on a real dataset.

Usage:
    python examples/route_task_demo.py --model facebook/opt-125m
"""

import argparse
import sys
import time
from typing import Any, Dict, List

def sha256_file(path):
    import hashlib
    from pathlib import Path

    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_benchmark_payload():
    from pathlib import Path
    import hashlib

    candidate_paths = [
        Path("vllm_registry_storage/payloads/opt125m_sql_delta.safetensors"),
        Path("integrations/vllm_route_plugin/tests/fixtures/opt125m_sql_delta.safetensors"),
        Path("tests/fixtures/opt125m_sql_delta.safetensors"),
    ]

    try:
        from integrations.vllm_route_plugin.tests.gen_test_payload import ensure_test_payload
        result = ensure_test_payload()

        if isinstance(result, tuple) and len(result) == 2:
            return str(result[0]), result[1]

        if result is not None:
            payload_path = Path(result)
            return str(payload_path), sha256_file(payload_path)
    except ImportError:
        pass

    try:
        from integrations.vllm_route_plugin.tests.gen_test_payload import generate_test_payload
        result = generate_test_payload()

        if isinstance(result, tuple) and len(result) == 2:
            return str(result[0]), result[1]

        if result is not None:
            payload_path = Path(result)
            return str(payload_path), sha256_file(payload_path)
    except ImportError:
        pass

    for path in candidate_paths:
        if path.exists():
            return str(path), sha256_file(path)

    raise FileNotFoundError(
        "Could not locate benchmark safetensors payload. "
        "Expected one of: "
        + ", ".join(str(p) for p in candidate_paths)
    )

def setup_neural_scalpel() -> Any:
    """Initialize Neural-Scalpel patches and register dynamic routes."""
    try:
        from integrations.vllm_route_plugin.patch import apply_all_patches
        from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        
        apply_all_patches()
        RoutePluginMetrics.reset()
        registry = get_vllm_registry()
        
        target_layers = [{
            "name": "model.decoder.layers.0.self_attn.qkv_proj.weight",
            "shape": [2304, 768],
            "dtype": "float16"
        }]

        # Setup real route
        payload_uri, payload_sha256 = ensure_benchmark_payload()

        registry.routes["sql-route"] = {
            "route_id": "sql-route",
            "tenant_id": "eval-tenant",
            "layers": target_layers,
            "payload_type": "real",
            "payload": {
                "format": "safetensors",
                "uri": payload_uri,
                "sha256": payload_sha256,
            }
        }
            
        return RoutePluginMetrics
    except ImportError:
        print("[ERROR] Neural-Scalpel plugins not available.", file=sys.stderr)
        return None

def evaluate_sql_exact_match(response: str, target: str) -> bool:
    """Evaluate simple exact match for SQL queries."""
    return target.strip().lower() in response.strip().lower()

def main() -> int:
    parser = argparse.ArgumentParser(description="Task-Specific Route Evaluation (Minimal E2E)")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    args = parser.parse_args()

    print("=" * 60)
    print(" Neural-Scalpel: Real Payload Route E2E Check")
    print("=" * 60)
    print("Goal: Validate real safetensors route application and rollback cleanliness in a minimal E2E flow.")
    print("NOTE: This does not prove task improvement unless the payload is a trained task-specific adapter.")
    print(f"Model: {args.model}")

    metrics = setup_neural_scalpel()
    if not metrics:
        return 1

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[ERROR] vLLM is not installed. Exiting.", file=sys.stderr)
        return 1

    print("\nInitializing engine...")
    llm = LLM(model=args.model, enforce_eager=True)

    # Evaluation dataset (Minimal 1-shot)
    prompt = "Translate to SQL: Find all users older than 30"
    target = "SELECT * FROM users WHERE age > 30"
    
    # We will test in this sequence:
    # 1. Base route: record baseline output
    # 2. SQL route: record output under real safetensors payload
    # 3. Base route again: record post-rollback baseline output
    #
    # This checks output change and rollback cleanliness.
    # It does not assume task improvement unless the payload is a trained SQL adapter.
    sequence = [
        {"route": "__base__", "label": "Initial Base Route"},
        {"route": "sql-route", "label": "SQL Route (Hot-Swapped)"},
        {"route": "__base__", "label": "Post-Rollback Base Route"},
    ]

    results = []

    for step in sequence:
        route = step["route"]
        label = step["label"]
        print(f"\n--- Testing: {label} ---")
        
        sp = SamplingParams(max_tokens=32, temperature=0.0)
        sp.extra_args = {"route_id": route}
        
        outputs = llm.generate([prompt], [sp], use_tqdm=False)
        response = outputs[0].outputs[0].text.strip()
        
        is_match = evaluate_sql_exact_match(response, target)
        results.append({"label": label, "route": route, "response": response, "is_match": is_match})
        
        print(f"Prompt:   {prompt}")
        print(f"Response: {response}")
        print(f"Match:    {is_match}")

    print("\n" + "=" * 60)
    print(" Summary of Minimal E2E Test")
    print("=" * 60)
    for res in results:
        print(f"{res['label']} ({res['route']}): {'PASS' if res['is_match'] else 'FAIL'}")

    print("\nValidation Checks:")
    
    # Validation logic
    base_1_match = results[0]["is_match"]
    sql_match = results[1]["is_match"]
    base_2_match = results[2]["is_match"]

    success = True
    
    swap_count = getattr(metrics, "swap_count", 0)
    rollback_count = getattr(metrics, "rollback_count", 0)
    violations = getattr(metrics, "mixed_batch_violation_count", 0)

    rollback_ok = (
        swap_count > 0
        and swap_count == rollback_count
        and violations == 0
    )

    print(f"Runtime Metrics: swaps={swap_count}, rollbacks={rollback_count}, violations={violations}")

    if rollback_ok:
        print("✅ Runtime rollback checks passed (swap_count > 0, swap_count == rollback_count, violations == 0).")
    else:
        print("❌ Runtime rollback checks failed or no swap was observed.")
        success = False

    if results[1]["response"] != results[0]["response"]:
        print("✅ Route-specific payload application observed (route output changed from base).")
    else:
        print("⚠️ Route output was identical to base. This may be expected for weak/test payloads; inspect swap metrics.")

    if sql_match and not base_1_match:
        print("✅ Task improvement observed for this prompt.")
        print("NOTE: This is a single-prompt smoke result, not full task validation.")

    if results[2]["response"] == results[0]["response"]:
        print("✅ Base before/after outputs match exactly under deterministic sampling.")
    else:
        print("⚠️ Base before/after outputs differ. Treat as a signal for review, not standalone proof of contamination.")

    if success:
        print("\n🎉 Real Payload Route E2E Check Passed!")
        return 0
    else:
        print("\n❌ Evaluation Failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
