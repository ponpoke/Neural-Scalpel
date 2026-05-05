import argparse
import json
import time
import os
import shutil
import random
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Phase 5-E: Multi-Route Transition & Mixed-Batch Safety Validation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--alpaca-payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=str, default="reports/phase_5e_mixed_batch.json")
    args = parser.parse_args()

    # Prepare environment
    base_dir = os.environ.get("SCALPEL_HOME", os.getcwd())
    storage_dir = os.path.join(base_dir, "vllm_registry_storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    # 1. Setup routes (Alpaca Real and Base)
    # Avoiding simulated route to prevent runtime schema issues for now.
    manifest_path = args.alpaca_payload.replace("_payload.safetensors", ".scalpel_route")
    with open(manifest_path, "r", encoding="utf-8") as f:
        alpaca_manifest = json.load(f)
    alpaca_route_id = alpaca_manifest["route_id"]
    shutil.copy2(manifest_path, os.path.join(storage_dir, f"{alpaca_route_id}.scalpel_route"))

    audit_log = os.path.join(base_dir, f"reports/audit_5e_{os.getpid()}.jsonl")
    if os.path.exists(audit_log):
        os.remove(audit_log)
        
    os.environ["SCALPEL_HOME"] = base_dir
    os.environ["SCALPEL_VLLM_REGISTRY_DIR"] = storage_dir
    os.environ["SCALPEL_AUDIT_LOG"] = audit_log
    os.environ["SCALPEL_RUNTIME_MODEL_HASH"] = "0" * 64

    # Apply patches and start vLLM
    from integrations.vllm_route_plugin.patch import apply_all_patches
    apply_all_patches()
    
    from vllm import LLM, SamplingParams
    print(f"[INFO] Initializing vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        disable_log_stats=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    
    # Generate prompts and routes
    available_routes = ["__base__", alpaca_route_id]
    prompts = []
    sampling_params_list = []
    route_assignments = []
    
    print(f"[INFO] Generating {args.requests} mixed-route requests...")
    random.seed(42)
    
    for i in range(args.requests):
        route = random.choice(available_routes)
        route_assignments.append(route)
        prompts.append(f"Mixed-batch safety test prompt {i} routed to {route}")
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        sp.extra_args = {"route_id": route}
        sampling_params_list.append(sp)
        
    route_distribution = dict(Counter(route_assignments))
    print(f"[INFO] Route distribution: {route_distribution}")

    # Run Generation
    print("[INFO] Submitting mixed batch to vLLM (testing route-homogeneous scheduling)...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params_list)
    t1 = time.time()
    
    # Force final rollback
    from integrations.vllm_route_plugin.runtime_context import get_current_vllm_runtime
    runtime = get_current_vllm_runtime()
    
    cleanup_rollback_executed = False
    active_route_after_cleanup = "UNKNOWN"
    
    if runtime and hasattr(runtime, "clear_active_route"):
        cleanup_rollback_executed = runtime.clear_active_route()
        active_route_after_cleanup = getattr(runtime, "active_route_id", "UNKNOWN")
        print(f"[INFO] Post-benchmark cleanup executed: {cleanup_rollback_executed}. Active route is now: {active_route_after_cleanup}")
        
    # Analyze audit log
    swap_events = 0
    rollback_events = 0
    verified_rollbacks = 0
    quarantine_events = 0
    route_violations = 0
    
    try:
        with open(audit_log, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                event = data.get("event")
                phase = data.get("phase")
                
                if event == "swap_completed" or phase == "SWAPPED":
                    swap_events += 1
                elif event == "rollback_completed" or phase == "BASE_RESTORED":
                    rollback_events += 1
                    if data.get("rollback_verified") is True:
                        verified_rollbacks += 1
                elif event == "quarantine" or data.get("quarantine") is True:
                    quarantine_events += 1
                elif event == "mixed_batch_violation" or phase == "VIOLATION":
                    route_violations += 1
    except Exception as e:
        print(f"[WARN] Audit log parsing error: {e}")
        
    # Read metrics directly if possible
    try:
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        route_violations = max(route_violations, RoutePluginMetrics.mixed_batch_violation_count)
    except:
        pass

    is_healthy = runtime.is_healthy if (runtime and hasattr(runtime, "is_healthy")) else True

    result = {
        "benchmark_type": "phase_5e_mixed_batch_safety",
        "config": {
            "requests": args.requests,
            "routes": available_routes,
            "model": args.model
        },
        "metrics": {
            "total_time_sec": round(t1 - t0, 2),
            "swap_count": swap_events,
            "rollback_count": rollback_events,
            "verified_rollbacks": verified_rollbacks,
            "route_violations": route_violations,
            "quarantine_events": quarantine_events,
            "worker_is_healthy": is_healthy,
            "route_distribution": route_distribution,
            "cleanup_rollback_executed": cleanup_rollback_executed,
            "active_route_after_cleanup": active_route_after_cleanup
        },
        "pass_criteria": {
            "violations_zero": route_violations == 0,
            "quarantine_zero": quarantine_events == 0 and is_healthy,
            "all_routes_requested": len([r for r, c in route_distribution.items() if c > 0]) == len(available_routes),
            "route_transitions_observed": swap_events >= 2 and rollback_events >= 2,
            "verified_rollbacks_recorded": verified_rollbacks >= 1,
            "all_requests_completed": len(outputs) == args.requests
        }
    }
    
    passed = all(result["pass_criteria"].values())
    result["status"] = "PASSED" if passed else "FAILED"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f" PHASE 5-E: MIXED-BATCH SAFETY RESULTS - {result['status']}")
    print("=" * 60)
    print(f" Requests:             {args.requests}")
    print(f" Route Distribution:   {route_distribution}")
    print(f" Total Time:           {result['metrics']['total_time_sec']} sec")
    print(f" Swap Count:           {swap_events}")
    print(f" Rollback Count:       {rollback_events}")
    print(f" Verified Rollbacks:   {verified_rollbacks}")
    print(f" Route Violations:     {route_violations}")
    print(f" Quarantine Events:    {quarantine_events}")
    print(f" Worker Healthy:       {is_healthy}")
    print("=" * 60)
    
    if not passed:
        print("[ERROR] Phase 5-E criteria not met:")
        for k, v in result["pass_criteria"].items():
            if not v:
                print(f"  - {k} failed")
    
if __name__ == "__main__":
    main()
