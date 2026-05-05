import argparse
import json
import subprocess
import time
import os
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

# 1. APPLY PATCHES BEFORE VLLM IMPORTS
from integrations.vllm_route_plugin.patch import apply_all_patches
apply_all_patches()

from vllm import LLM, SamplingParams

# ── Shared canonical prompt (must match Native LoRA benchmark exactly) ──
CANONICAL_PROMPT = "Write a short poem about a neural scalpel that swaps intelligence."

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])

def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "avg": 0.0}
    return {
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "p99": round(percentile(values, 99), 2),
        "avg": round(sum(values) / len(values), 2),
    }

def get_gpu_memory_mb() -> float:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        return float(output.strip())
    except Exception:
        return 0.0

def parse_audit_log(log_path: str) -> Dict[str, Any]:
    swap_latencies = []
    rollback_latencies = []
    verified_rollbacks = 0
    try:
        if not os.path.exists(log_path):
            print(f"[WARN] Audit log not found at {log_path}")
            return {"swaps": 0, "rollbacks": 0, "swap_stats": stats([]), "rollback_stats": stats([]), "verified_rollbacks": 0}
            
        with open(log_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    event = data.get("event")
                    phase = data.get("phase")
                    latency = data.get("latency_ms", 0.0)
                    
                    if event == "swap_completed" or phase == "SWAPPED":
                        swap_latencies.append(latency)
                    elif event == "rollback_completed" or phase == "BASE_RESTORED":
                        rollback_latencies.append(latency)
                        if data.get("rollback_verified") is True:
                            verified_rollbacks += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to parse audit log: {e}")
        
    return {
        "swaps": len(swap_latencies),
        "rollbacks": len(rollback_latencies),
        "swap_stats": stats(swap_latencies),
        "rollback_stats": stats(rollback_latencies),
        "verified_rollbacks": verified_rollbacks,
    }

def run_bench(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    audit_log: str,
    name: str = "base",
) -> Dict[str, Any]:
    print(f"[INFO] Running benchmark for: {name}")
    
    # Ensure audit log is fresh
    if os.path.exists(audit_log):
        os.remove(audit_log)
    Path(audit_log).parent.mkdir(parents=True, exist_ok=True)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    latencies = []
    
    # 1. Throughput & Batch Latency
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    # Output sample for qualitative comparison
    if outputs:
        print(f"[DEBUG] {name} Output Sample: {outputs[0].outputs[0].text[:120]}...")
    
    batch_time = t1 - t0
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    
    # 2. Individual Latencies distribution (sampled from 10 single-prompt runs)
    print(f"[INFO] Sampling individual latencies for {name} (10 requests)...")
    for p in prompts[:10]:
        st = time.perf_counter()
        _ = llm.generate([p], sampling_params)
        latencies.append((time.perf_counter() - st) * 1000)
    
    vram_used = get_gpu_memory_mb()
    audit_metrics = parse_audit_log(audit_log)

    return {
        "name": name,
        "batch_time_sec": round(batch_time, 2),
        "total_tokens": total_tokens,
        "throughput_req_per_s": round(len(prompts) / batch_time, 2),
        "throughput_tok_per_s": round(total_tokens / batch_time, 2),
        "latency_ms": stats(latencies),
        "vram_used_mb_nvidia_smi": vram_used,
        "swap_count": audit_metrics["swaps"],
        "rollback_count": audit_metrics["rollbacks"],
        "swap_stats": audit_metrics["swap_stats"],
        "rollback_stats": audit_metrics["rollback_stats"],
        "output_sample": outputs[0].outputs[0].text[:200] if outputs else "",
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 5-C: Neural-Scalpel route-window swap optimization benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model ID")
    parser.add_argument("--payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--prompts", type=int, default=50, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default="reports/bench_vllm_scalpel_overhead.json", help="Output JSON path")
    args = parser.parse_args()

    # ── Path Resolution ──────────────────────────────────────────────
    # Support cross-platform path resolution via environment variables.
    # Fallback: detect WSL home, then os.getcwd().
    base_dir = os.environ.get("SCALPEL_HOME")
    if not base_dir:
        if os.path.exists("/home/ponzu"):
            base_dir = "/home/ponzu/Neural-Scalpel"
        else:
            base_dir = os.getcwd()
            
    storage_dir = os.environ.get("SCALPEL_VLLM_REGISTRY_DIR", os.path.join(base_dir, "vllm_registry_storage"))
    audit_log = os.environ.get("SCALPEL_AUDIT_LOG", os.path.join(base_dir, "reports/bench_scalpel_audit.jsonl"))
    
    os.environ["SCALPEL_HOME"] = base_dir
    os.environ["SCALPEL_VLLM_REGISTRY_DIR"] = storage_dir
    os.environ["SCALPEL_AUDIT_LOG"] = audit_log
    os.environ["SCALPEL_RUNTIME_MODEL_HASH"] = "0" * 64

    print("=" * 60)
    print(" PHASE 5-C: NEURAL-SCALPEL ROUTE-WINDOW SWAP BENCHMARK")
    print("=" * 60)
    print(f"Base Model:    {args.model}")
    print(f"Payload:       {args.payload}")
    print(f"Prompts:       {args.prompts}")
    print(f"Max Tokens:    {args.max_tokens}")
    print(f"Audit Log:     {audit_log}")
    print(f"Prompt:        {CANONICAL_PROMPT}")
    print("-" * 60)
    print("PURPOSE:")
    print("  Measure Neural-Scalpel route-window persistent swap overhead")
    print("  on the same workload/prompt as the Native LoRA baseline.")
    print("=" * 60)

    # ── 1. Prepare Registry ──────────────────────────────────────────
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
        
    manifest_path = args.payload.replace("_payload.safetensors", ".scalpel_route")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    route_id = manifest["route_id"]
    
    target_manifest = os.path.join(storage_dir, f"{route_id}.scalpel_route")
    shutil.copy2(manifest_path, target_manifest)
    print(f"[INFO] Persisted route '{route_id}' to {target_manifest}")

    # ── 2. Initialize vLLM ───────────────────────────────────────────
    # IMPORTANT: Match Native LoRA benchmark params exactly for fair comparison
    print("[INFO] Initializing vLLM (enforce_eager=True, dtype=float16, gpu_memory_utilization=0.8)...")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        disable_log_stats=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
    )

    sampling_params_base = SamplingParams(
        temperature=0.0,  # Deterministic for comparison
        max_tokens=args.max_tokens,
    )
    
    sampling_params_scalpel = SamplingParams(
        temperature=0.0,  # Deterministic for comparison
        max_tokens=args.max_tokens,
    )
    sampling_params_scalpel.extra_args = {"route_id": route_id}

    # Use the EXACT same prompt as the Native LoRA benchmark
    test_prompts = [CANONICAL_PROMPT] * args.prompts

    # ── 3. Warmup ────────────────────────────────────────────────────
    print("[INFO] Warming up Base and Scalpel configurations...")
    _ = llm.generate(["Warmup"] * 5, sampling_params_base)
    _ = llm.generate(["Warmup"] * 5, sampling_params_scalpel)

    # After warmup, explicitly reset to base state
    from integrations.vllm_route_plugin.runtime_context import get_current_vllm_runtime
    runtime = get_current_vllm_runtime()
    if runtime and hasattr(runtime, "clear_active_route"):
        runtime.clear_active_route()
        print("[INFO] Warmup complete. Model explicitly reset to __base__ state.")

    # ── 4. Benchmark Base ────────────────────────────────────────────
    base_results = run_bench(llm, test_prompts, sampling_params_base, audit_log, name="base")
    
    # ── 5. Benchmark Scalpel ─────────────────────────────────────────
    scalpel_results = run_bench(llm, test_prompts, sampling_params_scalpel, audit_log, name="scalpel_alpaca")

    # ── 6. Phase 5-C Cleanup: Force rollback and re-parse audit ──────
    runtime = get_current_vllm_runtime()
    cleanup_rollback_executed = False
    if runtime and hasattr(runtime, "clear_active_route"):
        cleanup_rollback_executed = runtime.clear_active_route()
        print(f"[INFO] Post-benchmark cleanup: clear_active_route() -> {cleanup_rollback_executed}")

    # ── 7. Post-Cleanup Base Restoration Verification ────────────────
    print("[INFO] Verifying base restoration after cleanup...")
    # NOTE: We use the same test_prompts (BS=50) to ensure exact math match with base_results
    verify_outputs = llm.generate(test_prompts, sampling_params_base)
    post_cleanup_text = verify_outputs[0].outputs[0].text[:200] if verify_outputs else "FAILED"
    print(f"[DEBUG] Post-Cleanup Base Output: {post_cleanup_text[:100]}...")

    # Re-parse audit log AFTER verification to capture any rollbacks triggered by the base request
    audit_final = parse_audit_log(audit_log)
    scalpel_results["swap_count"] = audit_final["swaps"]
    scalpel_results["rollback_count"] = audit_final["rollbacks"]
    scalpel_results["verified_rollbacks"] = audit_final["verified_rollbacks"]
    scalpel_results["swap_stats"] = audit_final["swap_stats"]
    scalpel_results["rollback_stats"] = audit_final["rollback_stats"]

    # ── 8. CRITICAL: Validate that route was actually applied and restored ──
    route_proven = (scalpel_results["swap_count"] > 0)
    rollback_proven = (scalpel_results["rollback_count"] > 0 or cleanup_rollback_executed)
    
    if not route_proven:
        print("\n" + "!" * 60)
        print(" CRITICAL: swap_count = 0. Route application is NOT proven.")
        print("!" * 60)
    
    # Check exact match with pre-scalpel base output
    base_sample = base_results.get("output_sample", "")
    exact_match = (post_cleanup_text.strip() == base_sample.strip()) if base_sample else False
    print(f"[INFO] Base exact match after cleanup: {exact_match}")

    # ── 9. Build Final Result Structure ──────────────────────────────
    throughput_delta = round(
        ((scalpel_results["throughput_tok_per_s"] - base_results["throughput_tok_per_s"]) 
         / base_results["throughput_tok_per_s"]) * 100, 2
    )

    results = {
        "status": "measured",
        "benchmark_type": "phase_5c_route_window_swap_optimization",
        "route_application_proven": route_proven,
        "rollback_proven": rollback_proven,
        "canonical_prompt": CANONICAL_PROMPT,
        "model": args.model,
        "payload": route_id,
        "performance_interpretation": (
            f"Phase 5-C Route-Window optimization result. Throughput delta vs base: {throughput_delta}%. "
            f"Route proven: {route_proven}, Rollback proven: {rollback_proven}. "
            f"exact_match: {exact_match}. "
            + ("Confirmed persistent route usage with verified cleanup." if (route_proven and rollback_proven and exact_match) else "WARNING: Evidence incomplete.")
        ),
        "config": {
            "num_prompts": args.prompts,
            "max_tokens": args.max_tokens,
            "sampling_temperature": 0.0,
            "enforce_eager": True,
            "dtype": "float16",
            "gpu_memory_utilization": 0.8,
        },
        "metrics": {
            "base_tok_per_sec": base_results["throughput_tok_per_s"],
            "scalpel_tok_per_sec": scalpel_results["throughput_tok_per_s"],
            "throughput_delta_pct": throughput_delta,
            "e2e_latency_p50_ms": scalpel_results["latency_ms"]["p50"],
            "e2e_latency_p90_ms": scalpel_results["latency_ms"]["p90"],
            "e2e_latency_p99_ms": scalpel_results["latency_ms"]["p99"],
            "generated_tokens": scalpel_results["total_tokens"],
            "swap_count": scalpel_results["swap_count"],
            "rollback_count": scalpel_results["rollback_count"],
            "verified_rollbacks": scalpel_results["verified_rollbacks"],
            "swaps_per_token": round(scalpel_results["swap_count"] / scalpel_results["total_tokens"], 6) if scalpel_results["total_tokens"] > 0 else 0,
            "rollbacks_per_token": round(scalpel_results["rollback_count"] / scalpel_results["total_tokens"], 6) if scalpel_results["total_tokens"] > 0 else 0,
            "swap_latency_p50_ms": scalpel_results["swap_stats"]["p50"],
            "swap_latency_p99_ms": scalpel_results["swap_stats"]["p99"],
            "rollback_latency_p50_ms": scalpel_results["rollback_stats"]["p50"],
            "rollback_latency_p99_ms": scalpel_results["rollback_stats"]["p99"],
            "vram_peak_mb": scalpel_results["vram_used_mb_nvidia_smi"]
        },
        "qualitative_samples": {
            "base_sample": base_sample,
            "scalpel_sample": scalpel_results.get("output_sample", ""),
            "post_cleanup_base_sample": post_cleanup_text,
        },
        "post_cleanup_verification": {
            "executed": True,
            "exact_match": exact_match,
            "cleanup_rollback_executed": cleanup_rollback_executed,
            "rollback_checksum_verified": (scalpel_results["verified_rollbacks"] > 0),
            "active_route_after_cleanup": "__base__" if exact_match else "UNKNOWN",
        },
        "raw": {
            "base": base_results,
            "scalpel": scalpel_results
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(" PHASE 5-C RESULTS SUMMARY")
    print("=" * 60)
    print(f"Route Proven:            {results['route_application_proven']}")
    print(f"Rollback Proven:         {results['rollback_proven']}")
    print(f"Base Throughput:         {results['metrics']['base_tok_per_sec']:.2f} tok/s")
    print(f"Scalpel Throughput:      {results['metrics']['scalpel_tok_per_sec']:.2f} tok/s")
    print(f"Throughput Delta:        {results['metrics']['throughput_delta_pct']:.2f}%")
    print(f"Swap Count:              {results['metrics']['swap_count']}")
    print(f"Rollback Count:          {results['metrics']['rollback_count']}")
    print(f"Verified Rollbacks:      {results['metrics']['verified_rollbacks']}")
    print(f"Swaps/Token:             {results['metrics']['swaps_per_token']}")
    print(f"Swap Latency (p50/p99):  {results['metrics']['swap_latency_p50_ms']} / {results['metrics']['swap_latency_p99_ms']} ms")
    print(f"VRAM Peak:               {results['metrics']['vram_peak_mb']} MB")
    print(f"Post-Cleanup Exact Match: {results['post_cleanup_verification']['exact_match']}")
    print("=" * 60)
    print(f"Interpretation: {results['performance_interpretation']}")
    print("=" * 60)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
