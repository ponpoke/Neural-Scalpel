"""
Step 3: Route Injection Quality Evaluation

Compares task-level performance across injection modes:
  1. Target Base         - Qwen2.5-0.5B with no route (baseline)
  2. Target + Naive      - Uniform random delta (non-structured noise)
  3. Target + Random     - Random low-rank delta (structured but untrained)
  4. Target + Projected  - LoRA-style projected payload (JTSA simulation)
  5. After Rollback      - Must exactly match Target Base

Metrics:
  - PPL (perplexity on held-out text)
  - KL divergence from base distribution
  - Coding task pass rate (HumanEval-style subset)
  - Text quality (entropy, repetition, coherence)
"""

import os, sys, json, time, math, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.payload import compute_file_sha256, compute_tensor_sha256
from neural_scalpel.experimental.runtime import HotSwapRuntime, RuntimeState
from neural_scalpel.experimental.audit import AuditLogger

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SECRET_KEYS = {"eval-key": "eval-secret-value"}

# ── HumanEval-Style Coding Tasks (Small Subset) ───────────────

CODING_TASKS = [
    {
        "id": "fizzbuzz",
        "prompt": "def fizzbuzz(n: int) -> str:\n    \"\"\"Return 'FizzBuzz' if n is divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, else str(n).\"\"\"\n",
        "tests": [
            ("fizzbuzz(15)", "'FizzBuzz'"),
            ("fizzbuzz(3)", "'Fizz'"),
            ("fizzbuzz(5)", "'Buzz'"),
            ("fizzbuzz(7)", "'7'"),
        ],
    },
    {
        "id": "is_palindrome",
        "prompt": "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is a palindrome, ignoring case.\"\"\"\n",
        "tests": [
            ("is_palindrome('racecar')", "True"),
            ("is_palindrome('hello')", "False"),
            ("is_palindrome('Aba')", "True"),
        ],
    },
    {
        "id": "factorial",
        "prompt": "def factorial(n: int) -> int:\n    \"\"\"Return the factorial of n. n >= 0.\"\"\"\n",
        "tests": [
            ("factorial(0)", "1"),
            ("factorial(5)", "120"),
            ("factorial(1)", "1"),
        ],
    },
    {
        "id": "max_element",
        "prompt": "def max_element(lst: list) -> int:\n    \"\"\"Return the maximum element in the list.\"\"\"\n",
        "tests": [
            ("max_element([1, 3, 2])", "3"),
            ("max_element([-1, -5, -2])", "-1"),
            ("max_element([42])", "42"),
        ],
    },
    {
        "id": "reverse_string",
        "prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reverse of the string s.\"\"\"\n",
        "tests": [
            ("reverse_string('hello')", "'olleh'"),
            ("reverse_string('')", "''"),
            ("reverse_string('a')", "'a'"),
        ],
    },
    {
        "id": "count_vowels",
        "prompt": "def count_vowels(s: str) -> int:\n    \"\"\"Return the number of vowels (a,e,i,o,u) in s, case insensitive.\"\"\"\n",
        "tests": [
            ("count_vowels('hello')", "2"),
            ("count_vowels('AEIOU')", "5"),
            ("count_vowels('xyz')", "0"),
        ],
    },
    {
        "id": "sum_list",
        "prompt": "def sum_list(lst: list) -> int:\n    \"\"\"Return the sum of all elements in the list.\"\"\"\n",
        "tests": [
            ("sum_list([1, 2, 3])", "6"),
            ("sum_list([])", "0"),
            ("sum_list([-1, 1])", "0"),
        ],
    },
    {
        "id": "is_even",
        "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even.\"\"\"\n",
        "tests": [
            ("is_even(4)", "True"),
            ("is_even(3)", "False"),
            ("is_even(0)", "True"),
        ],
    },
]


# ── Text Quality Prompts ──────────────────────────────────────

TEXT_PROMPTS = [
    "Explain in one paragraph why neural networks are useful for natural language processing.",
    "Write a short recipe for chocolate chip cookies.",
    "Describe the water cycle in simple terms.",
    "What are three benefits of regular exercise?",
    "Explain what an API is to someone who is not a programmer.",
]

PPL_EVAL_TEXT = (
    "The transformer architecture has revolutionized natural language processing "
    "by enabling models to attend to all positions in the input sequence simultaneously. "
    "This parallel processing capability, combined with the self-attention mechanism, "
    "allows transformers to capture long-range dependencies more effectively than "
    "recurrent neural networks. The key innovation is the scaled dot-product attention "
    "function, which computes compatibility scores between query and key vectors."
)


# ── Core Evaluation Functions ─────────────────────────────────

def compute_ppl(model, tokenizer, text: str, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())


def compute_kl_divergence(model, base_logits_cache, tokenizer, text: str, device: str) -> float:
    """KL(base || current) on token-level distributions."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc)
    current_logprobs = torch.log_softmax(out.logits, dim=-1)
    base_logprobs = base_logits_cache
    kl = torch.nn.functional.kl_div(
        current_logprobs, base_logprobs.exp(), reduction="batchmean", log_target=False
    )
    return float(kl.item())


def generate_text(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def eval_coding_task(model, tokenizer, task: dict, device: str) -> dict:
    """Generate code completion and test it."""
    completion = generate_text(model, tokenizer, task["prompt"], device, max_new_tokens=200)

    # Extract just the function body (stop at next def or class)
    lines = completion.split("\n")
    body_lines = []
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            break
        body_lines.append(line)
    func_code = task["prompt"] + "\n".join(body_lines)

    # Run tests
    passed = 0
    total = len(task["tests"])
    for call, expected in task["tests"]:
        try:
            local_ns = {}
            exec(func_code, {}, local_ns)
            result = eval(call, {}, local_ns)
            if str(result) == expected:
                passed += 1
        except Exception:
            pass

    return {"task_id": task["id"], "passed": passed, "total": total, "pass_rate": passed / total}


def eval_text_quality(model, tokenizer, prompt: str, device: str) -> dict:
    """Evaluate text generation quality metrics."""
    output = generate_text(model, tokenizer, prompt, device, max_new_tokens=128)

    tokens = tokenizer.encode(output)
    length = len(tokens)

    # Repetition rate: fraction of 4-grams that are repeated
    if length >= 4:
        ngrams = [tuple(tokens[i:i+4]) for i in range(length - 3)]
        unique = len(set(ngrams))
        rep_rate = 1.0 - (unique / len(ngrams)) if ngrams else 0
    else:
        rep_rate = 0

    # Token entropy
    if length > 0:
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        probs = np.array(list(token_counts.values())) / length
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        entropy = 0

    return {
        "output_length": length,
        "repetition_rate": round(rep_rate, 4),
        "token_entropy": round(entropy, 4),
        "output_preview": output[:200],
    }


# ── Delta Generators ──────────────────────────────────────────

def create_naive_delta(param: torch.Tensor, scale: float = 0.02) -> torch.Tensor:
    """Uniform random noise (non-structured, worst case)."""
    return torch.randn_like(param, device="cpu") * scale


def create_random_lowrank_delta(param: torch.Tensor, rank: int = 8, scale: float = 0.02) -> torch.Tensor:
    """Random low-rank delta (structured but untrained)."""
    out_dim, in_dim = param.shape
    A = torch.randn(out_dim, rank, dtype=param.dtype, device="cpu") * scale
    B = torch.randn(rank, in_dim, dtype=param.dtype, device="cpu") * scale
    return A @ B


def create_projected_delta(param: torch.Tensor, rank: int = 8, scale: float = 0.015) -> torch.Tensor:
    """
    Simulated JTSA-projected delta: low-rank with SVD-cleaned structure.
    More realistic than random low-rank because it preserves variance structure.
    """
    out_dim, in_dim = param.shape
    A = torch.randn(out_dim, rank, dtype=param.dtype, device="cpu") * scale
    B = torch.randn(rank, in_dim, dtype=param.dtype, device="cpu") * scale
    raw = A @ B
    # SVD cleanup: preserve top singular values, zero out noise
    U, S, Vh = torch.linalg.svd(raw.float(), full_matrices=False)
    # Keep only significant components (energy-preserving truncation)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    keep = max(1, int((energy < 0.99).sum().item()) + 1)
    S_clean = S.clone()
    S_clean[keep:] = 0
    cleaned = (U @ torch.diag(S_clean) @ Vh).to(param.dtype)
    return cleaned


# ── Build Routes with Payloads ────────────────────────────────

TARGET_LAYERS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
]


def build_route_with_payload(model, route_id, tenant_id, delta_fn, model_hash, out_dir, signer):
    """Build a .scalpel_route manifest with safetensors payload."""
    sd = model.state_dict()
    payload_dir = os.path.join(out_dir, "payloads")
    os.makedirs(payload_dir, exist_ok=True)

    deltas = {}
    layer_specs = []
    for layer_name in TARGET_LAYERS:
        if layer_name not in sd:
            continue
        param = sd[layer_name]
        delta = delta_fn(param)
        payload_key = f"{layer_name}.delta"
        deltas[payload_key] = delta
        layer_specs.append({
            "name": layer_name, "shape": list(param.shape),
            "dtype": str(param.dtype).replace("torch.", ""),
            "delta_sha256": compute_tensor_sha256(delta),
            "payload_key": payload_key,
        })

    payload_filename = f"{route_id}.safetensors"
    payload_path = os.path.join(payload_dir, payload_filename)
    save_file(deltas, payload_path)

    route_data = {
        "route_schema_version": "0.1.0", "route_id": route_id,
        "source_model": "eval-source", "target_model": MODEL_ID,
        "source_adapter_sha256": hashlib.sha256(route_id.encode()).hexdigest(),
        "target_model_sha256": model_hash,
        "tenant_id": tenant_id, "license": "MIT",
        "projection_method": "EVAL", "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 80},
        "payload": {
            "format": "safetensors", "uri": os.path.join("payloads", payload_filename),
            "sha256": compute_file_sha256(payload_path),
            "size_bytes": os.path.getsize(payload_path),
        },
        "layers": layer_specs,
    }

    signed = signer.sign(route_data, "eval-key")
    manifest_path = os.path.join(out_dir, f"{route_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(signed, f)
    return manifest_path


# ── Main Evaluation ───────────────────────────────────────────

def evaluate_mode(model, tokenizer, mode_name, base_logits_cache, device):
    """Run full evaluation suite for current model state."""
    print(f"\n  [{mode_name}]")
    results = {"mode": mode_name}

    # PPL
    ppl = compute_ppl(model, tokenizer, PPL_EVAL_TEXT, device)
    results["ppl"] = round(ppl, 4)
    print(f"    PPL: {ppl:.4f}")

    # KL divergence
    kl = compute_kl_divergence(model, base_logits_cache, tokenizer, PPL_EVAL_TEXT, device)
    results["kl_divergence"] = round(kl, 6)
    print(f"    KL:  {kl:.6f}")

    # Coding tasks
    coding_results = []
    total_passed = 0
    total_tests = 0
    for task in CODING_TASKS:
        r = eval_coding_task(model, tokenizer, task, device)
        coding_results.append(r)
        total_passed += r["passed"]
        total_tests += r["total"]

    pass_rate = total_passed / total_tests if total_tests > 0 else 0
    results["coding_pass_rate"] = round(pass_rate, 4)
    results["coding_passed"] = total_passed
    results["coding_total"] = total_tests
    results["coding_details"] = coding_results
    print(f"    Coding: {total_passed}/{total_tests} ({pass_rate*100:.1f}%)")

    # Text quality
    text_results = []
    for prompt in TEXT_PROMPTS:
        r = eval_text_quality(model, tokenizer, prompt, device)
        text_results.append(r)

    avg_length = np.mean([r["output_length"] for r in text_results])
    avg_rep = np.mean([r["repetition_rate"] for r in text_results])
    avg_entropy = np.mean([r["token_entropy"] for r in text_results])
    results["avg_output_length"] = round(float(avg_length), 1)
    results["avg_repetition_rate"] = round(float(avg_rep), 4)
    results["avg_token_entropy"] = round(float(avg_entropy), 4)
    results["text_details"] = text_results
    print(f"    Text: len={avg_length:.0f} rep={avg_rep:.3f} entropy={avg_entropy:.2f}")

    return results


def apply_route_and_eval(runtime, model, tokenizer, route_id, tenant_id, mode_name, base_logits, device):
    """Apply a route, evaluate, then rollback."""
    from neural_scalpel.route.tenant import TenantContext
    tenant = TenantContext(tenant_id)
    route_data = runtime.registry.get_route(route_id)

    runtime.lock.acquire()
    try:
        runtime.capture_and_verify(route_data)
        runtime.swap(route_data)
        runtime.transition(RuntimeState.INFERENCE_ACTIVE)

        results = evaluate_mode(model, tokenizer, mode_name, base_logits, device)

        runtime.rollback()
        runtime.verify_rollback()
        runtime.transition(RuntimeState.IDLE)
    finally:
        runtime.lock.release()

    return results


def main():
    print("=" * 70)
    print("  Step 3: Route Injection Quality Evaluation")
    print(f"  Model: {MODEL_ID} | Device: {DEVICE}")
    print("=" * 70)

    out_dir = os.path.join(os.path.dirname(__file__), "quality_eval_results")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Compute model hash
    sd = model.state_dict()
    keys = sorted(sd.keys())
    h = hashlib.sha256()
    for k in [keys[0], keys[-1]]:
        h.update(sd[k].cpu().contiguous().numpy().tobytes()[:1024])
    model_hash = h.hexdigest()

    # Cache base logits for KL computation
    print("  Caching base logits...")
    enc = tokenizer(PPL_EVAL_TEXT, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        base_out = model(**enc)
    base_logits = torch.log_softmax(base_out.logits, dim=-1).detach()

    # Setup runtime
    signer = RouteSigner(SECRET_KEYS)
    registry = RouteRegistry(storage_dir=out_dir, signer=signer)
    audit = AuditLogger(os.path.join(out_dir, "audit.jsonl"))
    runtime = HotSwapRuntime(model, registry, model_hash,
                             audit_logger=audit, payload_base_dir=out_dir)

    # Build routes for each mode
    print("\n  Building evaluation routes...")

    build_route_with_payload(model, "naive-delta", "eval-tenant", 
        lambda p: create_naive_delta(p, scale=0.02), model_hash, out_dir, signer)
    registry.register_route(os.path.join(out_dir, "naive-delta.json"))

    build_route_with_payload(model, "random-lowrank", "eval-tenant",
        lambda p: create_random_lowrank_delta(p, rank=8, scale=0.02), model_hash, out_dir, signer)
    registry.register_route(os.path.join(out_dir, "random-lowrank.json"))

    build_route_with_payload(model, "projected-jtsa", "eval-tenant",
        lambda p: create_projected_delta(p, rank=8, scale=0.015), model_hash, out_dir, signer)
    registry.register_route(os.path.join(out_dir, "projected-jtsa.json"))

    # Load Actual LoRA Payload if it exists
    actual_lora_path = os.path.join(os.path.dirname(__file__), "..", "routes", "actual_loras", "qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    if os.path.exists(actual_lora_path):
        print(f"  Loading actual LoRA payload from {actual_lora_path}...")
        actual_sd = load_file(actual_lora_path)
        
        # We will build a delta function that returns the actual weight from the loaded sd
        def get_actual_delta(param, name):
            if name in actual_sd:
                return actual_sd[name].to(param.device, dtype=param.dtype)
            return torch.zeros_like(param)
            
        def build_actual_route(model, route_id, tenant_id, model_hash, out_dir, signer):
            sd = model.state_dict()
            payload_dir = os.path.join(out_dir, "payloads")
            os.makedirs(payload_dir, exist_ok=True)
            deltas = {}
            layer_specs = []
            
            # Use all layers present in the actual LoRA that are also in the model
            for layer_name in actual_sd.keys():
                if layer_name not in sd: continue
                param = sd[layer_name]
                delta = actual_sd[layer_name].to("cpu", dtype=torch.float16)
                payload_key = f"{layer_name}.delta"
                deltas[payload_key] = delta
                layer_specs.append({
                    "name": layer_name, "shape": list(param.shape),
                    "dtype": "float16",
                    "delta_sha256": compute_tensor_sha256(delta),
                    "payload_key": payload_key,
                })
                
            payload_filename = f"{route_id}.safetensors"
            payload_path = os.path.join(payload_dir, payload_filename)
            save_file(deltas, payload_path)
            
            route_data = {
                "route_schema_version": "0.1.0", "route_id": route_id,
                "source_model": "eval-source", "target_model": MODEL_ID,
                "source_adapter_sha256": hashlib.sha256(route_id.encode()).hexdigest(),
                "target_model_sha256": model_hash,
                "tenant_id": tenant_id, "license": "MIT",
                "projection_method": "EVAL", "calibration": {"forward_passes": 64},
                "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 80},
                "payload": {
                    "format": "safetensors", "uri": os.path.join("payloads", payload_filename),
                    "sha256": compute_file_sha256(payload_path),
                    "size_bytes": os.path.getsize(payload_path),
                },
                "layers": layer_specs,
            }
            signed = signer.sign(route_data, "eval-key")
            manifest_path = os.path.join(out_dir, f"{route_id}.json")
            with open(manifest_path, "w") as f:
                json.dump(signed, f)
            return manifest_path

        build_actual_route(model, "actual-lora", "eval-tenant", model_hash, out_dir, signer)
        registry.register_route(os.path.join(out_dir, "actual-lora.json"))
        has_actual_lora = True
    else:
        print("  Actual LoRA payload not found. Skipping actual LoRA benchmark.")
        has_actual_lora = False

    print("  Routes registered.")

    # ── Run Evaluations ───────────────────────────────────────

    all_results = []

    # 1. Target Base
    r = evaluate_mode(model, tokenizer, "Target Base", base_logits, DEVICE)
    all_results.append(r)

    # 2. Target + Naive
    r = apply_route_and_eval(runtime, model, tokenizer, "naive-delta", "eval-tenant",
                             "Target + Naive", base_logits, DEVICE)
    all_results.append(r)

    # 3. Target + Random Low-Rank
    r = apply_route_and_eval(runtime, model, tokenizer, "random-lowrank", "eval-tenant",
                             "Target + Random LR", base_logits, DEVICE)
    all_results.append(r)

    # 4. Target + Projected (JTSA)
    r = apply_route_and_eval(runtime, model, tokenizer, "projected-jtsa", "eval-tenant",
                             "Target + Projected", base_logits, DEVICE)
    all_results.append(r)

    # 4.5 Target + Actual LoRA
    if has_actual_lora:
        r = apply_route_and_eval(runtime, model, tokenizer, "actual-lora", "eval-tenant",
                                 "Target + Actual LoRA", base_logits, DEVICE)
        all_results.append(r)

    # 5. After Rollback (must match base)
    r = evaluate_mode(model, tokenizer, "After Rollback", base_logits, DEVICE)
    all_results.append(r)

    # ── Summary Table ─────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  QUALITY EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Mode':<25} {'PPL':>8} {'KL':>10} {'Code':>8} {'Rep':>8} {'Entropy':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        code_str = f"{r['coding_passed']}/{r['coding_total']}"
        print(f"  {r['mode']:<25} {r['ppl']:>8.4f} {r['kl_divergence']:>10.6f} {code_str:>8} {r['avg_repetition_rate']:>8.4f} {r['avg_token_entropy']:>8.2f}")

    # Verify rollback integrity
    base = all_results[0]
    rollback = all_results[-1]
    ppl_match = abs(base["ppl"] - rollback["ppl"]) < 0.001
    kl_match = rollback["kl_divergence"] < 0.0001
    code_match = base["coding_passed"] == rollback["coding_passed"]

    print(f"\n  Rollback Integrity:")
    print(f"    PPL match:  {'PASS' if ppl_match else 'FAIL'} (base={base['ppl']:.4f} rollback={rollback['ppl']:.4f})")
    print(f"    KL match:   {'PASS' if kl_match else 'FAIL'} (KL={rollback['kl_divergence']:.6f})")
    print(f"    Code match: {'PASS' if code_match else 'FAIL'} (base={base['coding_passed']} rollback={rollback['coding_passed']})")

    # Save full report
    report_path = os.path.join(out_dir, "quality_eval_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save markdown
    md_path = os.path.join(out_dir, "QUALITY_EVAL_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Route Injection Quality Evaluation Report\n\n")
        f.write(f"**Model:** {MODEL_ID} | **Device:** {DEVICE}\n\n")
        f.write("| Mode | PPL | KL Div | Code Pass | Rep Rate | Entropy |\n")
        f.write("|------|-----|--------|-----------|----------|--------|\n")
        for r in all_results:
            code_str = f"{r['coding_passed']}/{r['coding_total']}"
            f.write(f"| {r['mode']} | {r['ppl']:.4f} | {r['kl_divergence']:.6f} | {code_str} | {r['avg_repetition_rate']:.4f} | {r['avg_token_entropy']:.2f} |\n")
        f.write(f"\n**Rollback Integrity:** PPL={'PASS' if ppl_match else 'FAIL'}, KL={'PASS' if kl_match else 'FAIL'}, Code={'PASS' if code_match else 'FAIL'}\n")
        f.write(f"\n*Generated: {time.strftime('%Y-%m-%d %H:%M')}*\n")

    print(f"\n  Results saved to: {out_dir}")

    assert ppl_match, "PPL rollback mismatch!"
    assert kl_match, "KL rollback mismatch!"
    print("  ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
