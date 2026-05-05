import argparse
import time
import json
import traceback
import sys
from pathlib import Path
import torch

def ensure_benchmark_payload():
    from pathlib import Path
    import hashlib

    try:
        from integrations.vllm_route_plugin.tests.gen_test_payload import ensure_test_payload
        return ensure_test_payload()
    except ImportError:
        from integrations.vllm_route_plugin.tests.gen_test_payload import generate_test_payload

        result = generate_test_payload()

        if isinstance(result, tuple):
            return result

        payload_path = Path(result) if result else Path(
            "vllm_registry_storage/payloads/opt125m_sql_delta.safetensors"
        )

        if not payload_path.exists():
            raise FileNotFoundError(f"Generated payload not found: {payload_path}")

        sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        return str(payload_path), sha256

def run_soak_test(
    duration_hours: float,
    prompts_per_batch: int,
    routes: list[str],
    output_path: str,
    interval_seconds: float = 0.0,
    require_worker_health: bool = False
):
    print(f"[SOAK TEST] Starting Soak Test for {duration_hours}h.")
    print(f"Routes: {routes}")
    print(f"Output: {output_path}")

    try:
        from vllm import LLM, SamplingParams
        from integrations.vllm_route_plugin.patch import apply_all_patches
        from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
    except ImportError:
        print("[ERROR] vLLM or integrations not installed. Run in correct environment.")
        sys.exit(1)

    # Setup
    apply_all_patches()
    RoutePluginMetrics.reset()
    registry = get_vllm_registry()
    
    target_layers = [{"name": "model.decoder.layers.0.self_attn.qkv_proj.weight", "shape": [2304, 768], "dtype": "float16"}]
    
    # Load real payload if sql-route is used
    payload_uri, payload_sha256 = None, None
    if "sql-route" in routes:
        payload_uri, payload_sha256 = ensure_benchmark_payload()

    for r in routes:
        if r == "__base__": continue
        is_real = (r == "sql-route")
        route_dict = {
            "route_id": r,
            "tenant_id": "soak-tenant",
            "layers": target_layers,
            "payload_type": "real" if is_real else "simulated"
        }
        if is_real and payload_uri:
            route_dict["payload"] = {
                "format": "safetensors",
                "uri": payload_uri,
                "sha256": payload_sha256,
            }
        registry.routes[r] = route_dict

    print("[SOAK TEST] Initializing Engine...")
    llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    
    start_time = time.time()
    duration_secs = duration_hours * 3600
    
    stats = {
        "start_time": start_time,
        "batches_completed": 0,
        "total_requests": 0,
        "errors": 0,
        "history": []
    }
    
    prompts = ["Tell me a long story."] * prompts_per_batch
    
    print("[SOAK TEST] Engine initialized. Entering main loop...")
    while time.time() - start_time < duration_secs:
        batch_start = time.time()
        
        sampling_params_list = []
        for i in range(prompts_per_batch):
            route = routes[i % len(routes)]
            sp = SamplingParams(max_tokens=32, temperature=0.0)
            sp.extra_args = {"route_id": route}
            sampling_params_list.append(sp)

        try:
            _ = llm.generate(prompts, sampling_params_list)
            stats["batches_completed"] += 1
            stats["total_requests"] += prompts_per_batch
        except Exception as e:
            print(f"[SOAK TEST ERROR] Exception during generate: {e}")
            traceback.print_exc()
            stats["errors"] += 1
            if stats["errors"] > 10:
                print("[SOAK TEST] Too many errors. Aborting.")
                break
                
        # Metrics Collection
        vram_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        vram_res = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
        
        current_metrics = {
            "timestamp": time.time(),
            "elapsed_secs": time.time() - start_time,
            "requests_so_far": stats["total_requests"],
            "swap_count": RoutePluginMetrics.swap_count,
            "rollback_count": RoutePluginMetrics.rollback_count,
            "violations": RoutePluginMetrics.mixed_batch_violation_count,
            "vram_allocated_mb": vram_alloc,
            "vram_reserved_mb": vram_res
        }
        
        stats["history"].append(current_metrics)
        
        # Dump state periodically
        if stats["batches_completed"] % 10 == 0:
            print(f"Elapsed: {current_metrics['elapsed_secs']/3600:.2f}h | Reqs: {stats['total_requests']} | Swaps: {RoutePluginMetrics.swap_count} | Violations: {RoutePluginMetrics.mixed_batch_violation_count} | VRAM: {vram_res:.1f}MB")
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)

        if interval_seconds > 0:
            time.sleep(interval_seconds)

    print("[SOAK TEST] Completed.")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    # Validation Checks
    failed = False
    
    if RoutePluginMetrics.swap_count != RoutePluginMetrics.rollback_count:
        print("[SOAK TEST FAILED] swap_count != rollback_count mismatch detected.")
        failed = True
        
    if RoutePluginMetrics.mixed_batch_violation_count > 0:
        print("[SOAK TEST FAILED] mixed_batch_violation_count > 0 detected.")
        failed = True
        
    if stats["errors"] > 0:
        print("[SOAK TEST FAILED] unhandled generation errors occurred.")
        failed = True
        
    # Worker Health Check
    runtime = None
    try:
        from integrations.vllm_route_plugin.runtime_context import get_current_vllm_runtime
        runtime = get_current_vllm_runtime()
    except Exception:
        runtime = getattr(registry, "runtime", None)

    if runtime is not None and hasattr(runtime, "is_healthy"):
        if not runtime.is_healthy:
            print("[SOAK TEST FAILED] Worker quarantine detected (is_healthy=False).")
            failed = True
    else:
        msg = "[SOAK TEST WARNING] Worker health check unavailable."
        if require_worker_health:
            msg = "[SOAK TEST FAILED] Worker health check unavailable and --require-worker-health is set."
            failed = True
        print(msg)
            
    # VRAM Growth Check
    if len(stats["history"]) > 1:
        warmup_index = min(5, len(stats["history"]) - 1)
        baseline_vram = stats["history"][warmup_index]["vram_reserved_mb"]
        final_vram = stats["history"][-1]["vram_reserved_mb"]
        if final_vram - baseline_vram > 100:
            print(f"[SOAK TEST FAILED] VRAM growth exceeded 100MB (Baseline: {baseline_vram:.1f}MB -> Final: {final_vram:.1f}MB).")
            failed = True

    if failed:
        print("[SOAK TEST] Verdict: FAILED.")
        sys.exit(1)
    else:
        print("[SOAK TEST] Verdict: PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--prompts-per-batch", type=int, default=100)
    parser.add_argument("--routes", type=str, default="__base__,sql-route,alpaca-route")
    parser.add_argument("--output", type=str, default="reports/soak_24h.json")
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--require-worker-health", action="store_true", help="Fail if worker health check is unavailable")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    routes_list = [r.strip() for r in args.routes.split(",")]
    
    run_soak_test(
        duration_hours=args.duration_hours,
        prompts_per_batch=args.prompts_per_batch,
        routes=routes_list,
        output_path=args.output,
        interval_seconds=args.interval,
        require_worker_health=args.require_worker_health
    )
