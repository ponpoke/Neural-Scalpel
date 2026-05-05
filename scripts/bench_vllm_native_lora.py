import argparse
import json
import subprocess
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

# ── Shared Measurement Utilities ──────────────────────────────────────

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
    """Gets total GPU memory usage via nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        return float(output.strip())
    except Exception:
        return 0.0


def run_bench(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest] = None,
    name: str = "base",
) -> Dict[str, Any]:
    print(f"[INFO] Running benchmark for: {name}")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    latencies = []
    
    # 1. Throughput & Batch Latency
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    t1 = time.perf_counter()
    
    # Output sample for qualitative comparison
    if outputs:
        print(f"[DEBUG] {name} Output Sample: {outputs[0].outputs[0].text[:120]}...")
    
    batch_time = t1 - t0
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    
    # 2. Individual Latencies (sampled from 10 single-prompt runs)
    print(f"[INFO] Sampling individual latencies for {name} (10 requests)...")
    for p in prompts[:10]:
        st = time.perf_counter()
        _ = llm.generate([p], sampling_params, lora_request=lora_request)
        latencies.append((time.perf_counter() - st) * 1000)
    
    vram_used = get_gpu_memory_mb()

    return {
        "name": name,
        "batch_time_sec": round(batch_time, 2),
        "total_tokens": total_tokens,
        "throughput_req_per_s": round(len(prompts) / batch_time, 2),
        "throughput_tok_per_s": round(total_tokens / batch_time, 2),
        "latency_ms": stats(latencies),
        "vram_used_mb_nvidia_smi": vram_used,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Native LoRA Direct Benchmark (Same-Prompt Rerun)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model ID")
    parser.add_argument("--lora-repo", type=str, default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo", help="LoRA adapter repo")
    parser.add_argument("--lora-revision", type=str, default=None, help="LoRA adapter revision (commit hash)")
    parser.add_argument("--prompts", type=int, default=50, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default="reports/bench_vllm_native_lora.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print(" vLLM Native LoRA Direct Benchmark (Same-Prompt Rerun)")
    print("=" * 60)
    print(f"Base Model:    {args.model}")
    print(f"LoRA Repo:     {args.lora_repo}")
    print(f"LoRA Revision: {args.lora_revision or 'latest'}")
    print(f"Prompts:       {args.prompts}")
    print(f"Max Tokens:    {args.max_tokens}")
    print(f"Prompt:        {CANONICAL_PROMPT}")
    print("=" * 60)

    # Download LoRA
    print(f"[INFO] Downloading LoRA adapter from {args.lora_repo}...")
    lora_path = snapshot_download(repo_id=args.lora_repo, revision=args.lora_revision)
    print(f"[INFO] LoRA downloaded to: {lora_path}")

    # Initialize vLLM with LoRA support
    # IMPORTANT: Match Scalpel benchmark params exactly for fair comparison
    print("[INFO] Initializing vLLM with enable_lora=True, enforce_eager=True, dtype=float16, gpu_memory_utilization=0.8...")
    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=64,
        enforce_eager=True,
        disable_log_stats=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for comparison
        max_tokens=args.max_tokens,
    )
    
    # Use the EXACT same prompt as the Scalpel benchmark
    test_prompts = [CANONICAL_PROMPT] * args.prompts

    # Warmup both configurations to ensure fair measurement
    print("[INFO] Warming up Base configuration...")
    _ = llm.generate(["Warmup"] * 5, sampling_params)
    
    print("[INFO] Warming up LoRA configuration...")
    lora_request = LoRARequest("alpaca-lora", 1, lora_path)
    _ = llm.generate(["Warmup"] * 5, sampling_params, lora_request=lora_request)

    # Benchmark Base
    base_results = run_bench(llm, test_prompts, sampling_params, name="base")
    
    # Benchmark LoRA
    lora_results = run_bench(llm, test_prompts, sampling_params, lora_request=lora_request, name="native_lora")

    results = {
        "benchmark_type": "native_lora_same_prompt_rerun",
        "canonical_prompt": CANONICAL_PROMPT,
        "token_count_note": "generated_tokens counts only the main batch throughput run, not the 10 single-prompt latency samples.",
        "config": {
            "model": args.model,
            "lora_repo": args.lora_repo,
            "lora_revision": args.lora_revision,
            "lora_path": str(lora_path),
            "num_prompts": args.prompts,
            "max_tokens": args.max_tokens,
            "sampling_temperature": 0.0,
            "enforce_eager": True,
            "dtype": "float16",
            "gpu_memory_utilization": 0.8,
            "method_note": (
                "Native LoRA benchmark measures single-adapter throughput using vLLM enable_lora=True and LoRARequest. "
                "Same prompt, sampling params, and engine config as the Neural-Scalpel Phase 5-C benchmark."
            )
        },
        "base": base_results,
        "native_lora": lora_results,
        "comparison": {
            "throughput_delta_pct": round(((lora_results["throughput_tok_per_s"] - base_results["throughput_tok_per_s"]) / base_results["throughput_tok_per_s"]) * 100, 2),
            "vram_delta_mb_nvidia_smi": round(lora_results["vram_used_mb_nvidia_smi"] - base_results["vram_used_mb_nvidia_smi"], 2),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(" Native LoRA Benchmark Results Summary")
    print("=" * 60)
    print(f"Base Throughput:   {base_results['throughput_tok_per_s']} tok/s")
    print(f"LoRA Throughput:   {lora_results['throughput_tok_per_s']} tok/s")
    print(f"Throughput Delta:  {results['comparison']['throughput_delta_pct']}%")
    print(f"Base VRAM (SMI):   {base_results['vram_used_mb_nvidia_smi']} MB")
    print(f"LoRA VRAM (SMI):   {lora_results['vram_used_mb_nvidia_smi']} MB")
    print(f"VRAM Delta (SMI):  {results['comparison']['vram_delta_mb_nvidia_smi']} MB")
    print(f"E2E Latency p50:   {lora_results['latency_ms']['p50']} ms")
    print(f"E2E Latency p99:   {lora_results['latency_ms']['p99']} ms")
    print("=" * 60)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
