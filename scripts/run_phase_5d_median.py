import argparse
import json
import subprocess
import os
import statistics
from pathlib import Path
from typing import List

def run_worker(method: str, run_id: int, args: argparse.Namespace) -> dict:
    output_file = f"reports/tmp_5d_{method}_run{run_id}.json"
    cmd = [
        "python", "scripts/run_phase_5d_worker.py",
        "--method", method,
        "--output", output_file,
        "--num-prompts", str(args.prompts),
        "--max-tokens", str(args.max_tokens)
    ]
    
    print(f"\n>>> Running {method} (Run {run_id}/{args.runs})")
    subprocess.run(cmd, check=True)
    
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    os.remove(output_file)
    
    # ── CRITICAL: Enforce Scalpel Validation ──
    if method == "scalpel":
        if data.get("swap_count", 0) <= 0:
            raise RuntimeError(f"Scalpel route application not proven in run {run_id}: swap_count={data.get('swap_count')}")
        if data.get("verified_rollbacks", 0) <= 0:
            raise RuntimeError(f"Scalpel rollback verification not proven in run {run_id}: verified_rollbacks={data.get('verified_rollbacks')}")

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of times to run each configuration")
    parser.add_argument("--prompts", type=int, default=25, help="Number of unique prompts to use")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per generation")
    parser.add_argument("--output", type=str, default="reports/phase_5d_repeated_median.json")
    args = parser.parse_args()

    dataset_path = Path("tests/alpaca_evaluation_dataset.json")
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        if args.prompts > len(dataset):
            raise ValueError(f"--prompts={args.prompts} exceeds dataset size={len(dataset)}")

    methods = ["base", "native_lora", "scalpel"]
    all_results = {m: [] for m in methods}

    for run_id in range(1, args.runs + 1):
        for method in methods:
            result = run_worker(method, run_id, args)
            all_results[method].append(result)

    # Calculate Medians and Reproducibility Stats
    summary = {}
    for method in methods:
        tok_s_list = [r["throughput_tok_per_s"] for r in all_results[method]]
        req_s_list = [r["throughput_req_per_s"] for r in all_results[method]]
        vram_list = [r["vram_used_mb"] for r in all_results[method]]
        
        summary[method] = {
            "throughput_tok_per_s_median": statistics.median(tok_s_list),
            "throughput_tok_per_s_min": min(tok_s_list),
            "throughput_tok_per_s_max": max(tok_s_list),
            "throughput_tok_per_s_mean": statistics.mean(tok_s_list),
            "throughput_tok_per_s_stdev": statistics.stdev(tok_s_list) if len(tok_s_list) > 1 else 0.0,
            
            "throughput_req_per_s_median": statistics.median(req_s_list),
            "vram_used_mb_median": statistics.median(vram_list),
            "runs_data": tok_s_list
        }
        
        if method == "scalpel":
            swaps = [r["swap_count"] for r in all_results[method]]
            verified = [r["verified_rollbacks"] for r in all_results[method]]
            summary[method]["swap_count_median"] = statistics.median(swaps)
            summary[method]["verified_rollbacks_median"] = statistics.median(verified)

    final_report = {
        "benchmark_type": "phase_5d_repeated_median",
        "config": {
            "runs": args.runs,
            "prompts_requested": args.prompts,
            "max_tokens": args.max_tokens,
            "prompt_source": "tests/alpaca_evaluation_dataset.json",
            "prompt_selection": "deterministic_first_N"
        },
        "summary": summary,
        "all_results": all_results
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print(" PHASE 5-D: REPEATED MEDIAN BENCHMARK RESULTS")
    print("=" * 60)
    for method in methods:
        med_tok = summary[method]["throughput_tok_per_s_median"]
        stdev_tok = summary[method]["throughput_tok_per_s_stdev"]
        print(f" {method.upper().ljust(15)} : {med_tok:.2f} tok/s ± {stdev_tok:.2f} (Median of {args.runs} runs)")
    
    print("-" * 60)
    base_tok = summary["base"]["throughput_tok_per_s_median"]
    scalpel_tok = summary["scalpel"]["throughput_tok_per_s_median"]
    native_tok = summary["native_lora"]["throughput_tok_per_s_median"]
    
    if base_tok > 0:
        scalpel_delta = ((scalpel_tok - base_tok) / base_tok) * 100
        native_delta = ((native_tok - base_tok) / base_tok) * 100
        print(f" Scalpel vs Base:  {scalpel_delta:+.2f}%")
        print(f" Native vs Base:   {native_delta:+.2f}%")
        
    if native_tok > 0:
        scalpel_vs_native = ((scalpel_tok - native_tok) / native_tok) * 100
        print(f" Scalpel vs Native:{scalpel_vs_native:+.2f}%")

    print("=" * 60)
    print(f"Full report saved to {args.output}")

if __name__ == "__main__":
    main()
