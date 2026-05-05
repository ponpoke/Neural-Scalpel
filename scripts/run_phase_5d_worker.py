import argparse
import json
import time
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

# Try to load prompt dataset
def load_prompts(dataset_path: str, count: int) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["instruction"] for item in data]
    return prompts[:count]

def get_gpu_memory_mb() -> float:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        return float(output.strip())
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["base", "native_lora", "scalpel"], required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lora-repo", type=str, default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    parser.add_argument("--payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--prompts-file", type=str, default="tests/alpaca_evaluation_dataset.json")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    
    # Pre-setup dependencies
    from vllm import LLM, SamplingParams
    
    if args.method == "scalpel":
        from integrations.vllm_route_plugin.patch import apply_all_patches
        apply_all_patches()
        
        # Setup registry
        import shutil
        base_dir = os.environ.get("SCALPEL_HOME", os.getcwd())
        storage_dir = os.path.join(base_dir, "vllm_registry_storage")
        os.makedirs(storage_dir, exist_ok=True)
        manifest_path = args.payload.replace("_payload.safetensors", ".scalpel_route")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        route_id = manifest["route_id"]
        target_manifest = os.path.join(storage_dir, f"{route_id}.scalpel_route")
        shutil.copy2(manifest_path, target_manifest)
        
        os.environ["SCALPEL_HOME"] = base_dir
        os.environ["SCALPEL_VLLM_REGISTRY_DIR"] = storage_dir
        audit_log = os.path.join(base_dir, f"reports/audit_worker_{os.getpid()}.jsonl")
        os.environ["SCALPEL_AUDIT_LOG"] = audit_log
        os.environ["SCALPEL_RUNTIME_MODEL_HASH"] = "0" * 64
        
        if os.path.exists(audit_log):
            os.remove(audit_log)
    elif args.method == "native_lora":
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest
        lora_path = snapshot_download(repo_id=args.lora_repo)

    # Init vLLM
    print(f"[{args.method}] Initializing vLLM...")
    if args.method == "native_lora":
        llm = LLM(
            model=args.model,
            enable_lora=True,
            max_lora_rank=64,
            enforce_eager=True,
            disable_log_stats=True,
            dtype="float16",
            gpu_memory_utilization=0.8,
        )
    else:
        llm = LLM(
            model=args.model,
            enforce_eager=True,
            disable_log_stats=True,
            dtype="float16",
            gpu_memory_utilization=0.8,
        )
        
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    if args.method == "scalpel":
        sampling_params.extra_args = {"route_id": route_id}

    # Warmup
    print(f"[{args.method}] Warming up...")
    if args.method == "native_lora":
        lora_request = LoRARequest("alpaca-lora", 1, lora_path)
        _ = llm.generate(["Warmup"] * 5, sampling_params, lora_request=lora_request)
    else:
        _ = llm.generate(["Warmup"] * 5, sampling_params)

    # Base state cleanup for scalpel
    if args.method == "scalpel":
        from integrations.vllm_route_plugin.runtime_context import get_current_vllm_runtime
        runtime = get_current_vllm_runtime()
        if runtime and hasattr(runtime, "clear_active_route"):
            runtime.clear_active_route()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Benchmark Run
    print(f"[{args.method}] Running batch generation...")
    t0 = time.perf_counter()
    if args.method == "native_lora":
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()

    batch_time = t1 - t0
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    throughput_tok_s = total_tokens / batch_time if batch_time > 0 else 0
    throughput_req_s = len(prompts) / batch_time if batch_time > 0 else 0
    vram_mb = get_gpu_memory_mb()
    
    # Audit log parsing
    swap_count = 0
    rollback_count = 0
    verified_rollbacks = 0
    
    if args.method == "scalpel":
        # Force rollback
        if runtime and hasattr(runtime, "clear_active_route"):
            runtime.clear_active_route()
            
        try:
            with open(audit_log, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("event") == "swap_completed" or data.get("phase") == "SWAPPED":
                            swap_count += 1
                        elif data.get("event") == "rollback_completed" or data.get("phase") == "BASE_RESTORED":
                            rollback_count += 1
                            if data.get("rollback_verified") is True:
                                verified_rollbacks += 1
                    except:
                        pass
            if os.path.exists(audit_log):
                os.remove(audit_log)
        except:
            pass

    results = {
        "method": args.method,
        "batch_time_sec": batch_time,
        "total_tokens": total_tokens,
        "throughput_tok_per_s": throughput_tok_s,
        "throughput_req_per_s": throughput_req_s,
        "vram_used_mb": vram_mb,
        "swap_count": swap_count,
        "rollback_count": rollback_count,
        "verified_rollbacks": verified_rollbacks
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"[{args.method}] Completed. Tok/s: {throughput_tok_s:.2f}")

if __name__ == "__main__":
    main()
