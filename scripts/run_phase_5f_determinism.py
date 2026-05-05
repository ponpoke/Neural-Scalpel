import argparse
import json
import time
import os
import shutil
from pathlib import Path

def get_top_logprobs(outputs):
    """Extracts a simple list of top token IDs and their logprobs for comparison."""
    if not outputs or not outputs[0].outputs:
        return []
    
    logprobs_list = outputs[0].outputs[0].logprobs
    if not logprobs_list:
        return []
        
    extracted = []
    for step in logprobs_list:
        # Get the token with the highest logprob at this step
        if not step:
            continue
        top_token_id = max(step.keys(), key=lambda k: step[k].logprob)
        top_logprob = step[top_token_id].logprob
        extracted.append({"token_id": top_token_id, "logprob": top_logprob})
    return extracted

def main():
    parser = argparse.ArgumentParser(description="Phase 5-F: Determinism & Top-Token Logprob Trace Rollback Verification")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--alpaca-payload", type=str, default="routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    parser.add_argument("--prompt", type=str, default="Write a short poem about a neural scalpel.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=str, default="reports/phase_5f_determinism.json")
    args = parser.parse_args()

    # Prepare environment
    base_dir = os.environ.get("SCALPEL_HOME", os.getcwd())
    storage_dir = os.path.join(base_dir, "vllm_registry_storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    manifest_path = args.alpaca_payload.replace("_payload.safetensors", ".scalpel_route")
    with open(manifest_path, "r", encoding="utf-8") as f:
        alpaca_manifest = json.load(f)
    alpaca_route_id = alpaca_manifest["route_id"]
    shutil.copy2(manifest_path, os.path.join(storage_dir, f"{alpaca_route_id}.scalpel_route"))

    audit_log = os.path.join(base_dir, f"reports/audit_5f_{os.getpid()}.jsonl")
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
    
    sampling_params_base = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    
    sampling_params_scalpel = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    sampling_params_scalpel.extra_args = {"route_id": alpaca_route_id}

    # 1. Run Base (Before)
    print("[INFO] Running Base (Before)...")
    out_base_before = llm.generate([args.prompt], sampling_params_base)
    text_before = out_base_before[0].outputs[0].text
    logprobs_before = get_top_logprobs(out_base_before)

    # 2. Run Scalpel (Alpaca)
    print(f"[INFO] Running Scalpel Route ({alpaca_route_id})...")
    out_scalpel = llm.generate([args.prompt], sampling_params_scalpel)
    text_scalpel = out_scalpel[0].outputs[0].text
    logprobs_scalpel = get_top_logprobs(out_scalpel)

    # 3. Explicit Cleanup / Rollback
    print("[INFO] Calling clear_active_route() to force rollback...")
    from integrations.vllm_route_plugin.runtime_context import get_current_vllm_runtime
    runtime = get_current_vllm_runtime()
    
    cleanup_executed = False
    if runtime and hasattr(runtime, "clear_active_route"):
        cleanup_executed = runtime.clear_active_route()
        print(f"[INFO] clear_active_route() returned: {cleanup_executed}")

    # Clear vLLM Caches if possible (To isolate purely weight rollback differences from KV cache differences)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    engine = getattr(llm, "llm_engine", None)
    if engine:
        for fn_name in ("reset_prefix_cache", "reset_encoder_cache"):
            try:
                fn = getattr(engine, fn_name, None)
                if callable(fn):
                    fn()
                    print(f"[INFO] Called llm_engine.{fn_name}()")
            except Exception as e:
                print(f"[WARN] {fn_name} failed: {e}")

    # 4. Run Base (After)
    print("[INFO] Running Base (After Rollback)...")
    out_base_after = llm.generate([args.prompt], sampling_params_base)
    text_after = out_base_after[0].outputs[0].text
    logprobs_after = get_top_logprobs(out_base_after)
    
    # Analyze similarities
    text_exact_match = (text_before == text_after)
    
    # Calculate Top-Token Logprob Trace Similarity
    matched_tokens = 0
    total_tokens = min(len(logprobs_before), len(logprobs_after))
    first_mismatch_index = None
    
    for i in range(total_tokens):
        same_token = logprobs_before[i]["token_id"] == logprobs_after[i]["token_id"]
        diff = abs(logprobs_before[i]["logprob"] - logprobs_after[i]["logprob"]) if same_token else None
        
        if same_token and diff is not None and diff < 1e-3:  # Tolerance for floating point math non-determinism
            matched_tokens += 1
        elif first_mismatch_index is None:
            first_mismatch_index = i

    top_token_logprob_similarity_pct = (matched_tokens / total_tokens * 100) if total_tokens > 0 else 0.0

    # Parse Audit Log for Checksum verification
    verified_rollbacks = 0
    try:
        with open(audit_log, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if (data.get("event") == "rollback_completed" or data.get("phase") == "BASE_RESTORED"):
                    if data.get("rollback_verified") is True:
                        verified_rollbacks += 1
    except Exception as e:
        print(f"[WARN] Audit log parsing error: {e}")

    result = {
        "benchmark_type": "phase_5f_determinism",
        "note": "Uses vLLM RequestOutput logprobs API as a top-token logprob trace proxy, not a full-vocabulary logits distribution.",
        "config": {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens
        },
        "metrics": {
            "cleanup_executed": cleanup_executed,
            "checksum_verified_rollbacks": verified_rollbacks,
            "text_exact_match": text_exact_match,
            "top_token_logprob_similarity_pct": round(top_token_logprob_similarity_pct, 2),
            "matched_tokens": matched_tokens,
            "total_tokens_compared": total_tokens,
            "first_mismatch_index": first_mismatch_index
        },
        "samples": {
            "text_before": text_before,
            "text_scalpel": text_scalpel,
            "text_after": text_after
        },
        "pass_criteria": {
            "cleanup_executed": cleanup_executed,
            "checksum_rollback_verified": verified_rollbacks > 0,
            "logprobs_available": total_tokens > 0,
            "top_token_logprob_similarity_high": top_token_logprob_similarity_pct > 95.0, # Accept minor vLLM math non-determinism
            "route_behavior_changed": text_before != text_scalpel
        }
    }
    
    passed = all(result["pass_criteria"].values())
    result["status"] = "PASSED" if passed else "FAILED"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f" PHASE 5-F: DETERMINISM & LOGPROB TRACE VERIFICATION - {result['status']}")
    print("=" * 60)
    print(f" Route Behavior Changed: {result['pass_criteria']['route_behavior_changed']} (Scalpel output differs from Base)")
    print(f" Checksum Verified:      {verified_rollbacks > 0} (Count: {verified_rollbacks})")
    print(f" Text Exact Match:       {text_exact_match}")
    print(f" Trace Similarity:       {top_token_logprob_similarity_pct:.2f}% ({matched_tokens}/{total_tokens} tokens match closely)")
    if first_mismatch_index is not None:
        print(f" First Mismatch Index:   {first_mismatch_index}")
    print("=" * 60)
    print("[DEBUG] Base Before: ", text_before[:50].replace('\n', ' ') + "...")
    print("[DEBUG] Scalpel:     ", text_scalpel[:50].replace('\n', ' ') + "...")
    print("[DEBUG] Base After:  ", text_after[:50].replace('\n', ' ') + "...")
    print("=" * 60)
    
if __name__ == "__main__":
    main()