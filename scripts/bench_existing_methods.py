import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "avg": None}
    return {
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "avg": sum(values) / len(values),
    }


def run_worker_subprocess(
    *,
    model: str,
    prompts: int,
    max_tokens: int,
    worker_output: Path,
    worker_kind: str,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--worker-kind",
        worker_kind,
        "--model",
        model,
        "--prompts",
        str(prompts),
        "--max-tokens",
        str(max_tokens),
        "--worker-output",
        str(worker_output),
    ]

    print(f"[MASTER] Running worker: {worker_kind} -> {worker_output}")
    subprocess.run(cmd, check=True)

    with worker_output.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_master(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_path.parent / "existing_methods_workers"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    reload_results: List[Dict[str, Any]] = []

    print("=" * 60)
    print(" Existing Serving Methods Benchmark")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Reloads:     {args.reloads}")
    print(f"Prompts/run: {args.prompts}")
    print(f"Max tokens:  {args.max_tokens}")

    for i in range(args.reloads):
        worker_output = tmp_dir / f"reload_worker_{i}.json"
        result = run_worker_subprocess(
            model=args.model,
            prompts=args.prompts,
            max_tokens=args.max_tokens,
            worker_output=worker_output,
            worker_kind="reload",
        )
        reload_results.append(result)

    vram_output = tmp_dir / "vram_worker.json"
    vram_result = run_worker_subprocess(
        model=args.model,
        prompts=1,
        max_tokens=1,
        worker_output=vram_output,
        worker_kind="vram",
    )

    reload_latencies = [r["load_latency_sec"] for r in reload_results]
    generation_latencies = [r["generation_latency_sec"] for r in reload_results]
    reload_vram_reserved = [r["vram_reserved_mb"] for r in reload_results]
    reload_vram_allocated = [r["vram_allocated_mb"] for r in reload_results]

    single_reserved = vram_result["vram_reserved_mb"]
    single_allocated = vram_result["vram_allocated_mb"]

    results = {
        "model_id": args.model,
        "method_note": (
            "Model reload is measured as process-level vLLM reinitialization. "
            "Each reload iteration runs in a fresh Python subprocess to avoid "
            "in-process vLLM/NCCL/CUDA state retention."
        ),
        "multi_instance_note": (
            "Multi-instance VRAM is estimated assuming linear scaling from "
            "single-instance VRAM. Actual multi-process throughput measurements remain pending."
        ),
        "config": {
            "num_reloads": args.reloads,
            "prompts_per_run": args.prompts,
            "max_tokens": args.max_tokens,
        },
        "metrics": {
            "reload_latency_sec": stats(reload_latencies),
            "generation_latency_sec": stats(generation_latencies),
            "reload_worker_vram_reserved_mb": stats(reload_vram_reserved),
            "reload_worker_vram_allocated_mb": stats(reload_vram_allocated),
            "single_instance_vram_reserved_mb": single_reserved,
            "single_instance_vram_allocated_mb": single_allocated,
            "estimated_3_instance_vram_reserved_mb": single_reserved * 3,
            "estimated_3_instance_vram_allocated_mb": single_allocated * 3,
        },
        "raw_reload_results": reload_results,
        "raw_vram_result": vram_result,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[SUCCESS] Benchmark complete.")
    print(f"Results saved to: {output_path}")
    print(f"Reload p50:       {results['metrics']['reload_latency_sec']['p50']:.2f}s")
    print(f"Reload avg:       {results['metrics']['reload_latency_sec']['avg']:.2f}s")
    print(f"Single VRAM:      {single_reserved:.1f} MB reserved")
    print(f"Est. 3 instances: {single_reserved * 3:.1f} MB reserved")

    return 0


def run_worker(args: argparse.Namespace) -> int:
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[WORKER] kind={args.worker_kind}, model={args.model}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        disable_log_stats=True,
    )
    t1 = time.perf_counter()

    load_latency_sec = t1 - t0

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    prompts = ["List the prime numbers up to 100."] * args.prompts
    if args.worker_kind == "vram":
        prompts = ["warmup"]

    gen_t0 = time.perf_counter()
    _ = llm.generate(prompts, sp, use_tqdm=False)
    gen_t1 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vram_reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
    vram_allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    result = {
        "worker_kind": args.worker_kind,
        "model_id": args.model,
        "load_latency_sec": load_latency_sec,
        "generation_latency_sec": gen_t1 - gen_t0,
        "prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "vram_reserved_mb": vram_reserved,
        "vram_allocated_mb": vram_allocated,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[WORKER] wrote {output_path}")
    print(f"[WORKER] load={load_latency_sec:.2f}s, gen={gen_t1 - gen_t0:.2f}s, vram={vram_reserved:.1f}MB")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Existing Serving Methods: Model Reload / Multi-Instance Estimate"
    )
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--reloads", type=int, default=3)
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=str, default="reports/bench_existing_methods.json")

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-kind", type=str, choices=["reload", "vram"], default="reload")
    parser.add_argument("--worker-output", type=str, default="reports/bench_existing_methods_worker.json")

    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())