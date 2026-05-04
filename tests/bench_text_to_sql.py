"""
Step 3.5: Text-to-SQL Quality Evaluation

Evaluates actual capability enhancement using a real Text-to-SQL LoRA.
Base Model: Qwen/Qwen2.5-Coder-0.5B-Instruct
Modes:
  1. Target Base
  2. Target + Naive Noise
  3. Target + Random Low-Rank
  4. Target + Alpaca LoRA (Style shift)
  5. Target + Text-to-SQL LoRA (Capability enhancement)
  6. After Rollback
"""

import os, sys, json, time, math, hashlib, re
from pathlib import Path
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

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SECRET_KEYS = {"eval-key": "eval-secret-value"}

# ── Text-to-SQL Benchmark Dataset (Subset) ───────────────

SQL_TASKS = [
    {
        "id": "sql_1",
        "schema": "CREATE TABLE employees (id INT, name VARCHAR, salary INT, department_id INT); CREATE TABLE departments (id INT, name VARCHAR);",
        "question": "Find the names of all employees in the 'Sales' department.",
        "expected": "SELECT employees.name FROM employees JOIN departments ON employees.department_id = departments.id WHERE departments.name = 'Sales'"
    },
    {
        "id": "sql_2",
        "schema": "CREATE TABLE students (id INT, name VARCHAR, age INT); CREATE TABLE grades (student_id INT, grade VARCHAR);",
        "question": "Count the number of students who have an 'A' grade.",
        "expected": "SELECT count(*) FROM students JOIN grades ON students.id = grades.student_id WHERE grades.grade = 'A'"
    },
    {
        "id": "sql_3",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL);",
        "question": "What is the name of the most expensive product?",
        "expected": "SELECT name FROM products ORDER BY price DESC LIMIT 1"
    },
    {
        "id": "sql_4",
        "schema": "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, order_date DATE);",
        "question": "Show the total amount of orders placed by customer 5.",
        "expected": "SELECT sum(amount) FROM orders WHERE customer_id = 5"
    },
    {
        "id": "sql_5",
        "schema": "CREATE TABLE flights (id INT, origin VARCHAR, destination VARCHAR, price INT);",
        "question": "Find all flights from 'New York' to 'London' that cost less than 500.",
        "expected": "SELECT * FROM flights WHERE origin = 'New York' AND destination = 'London' AND price < 500"
    }
]

def normalize_sql(sql: str) -> str:
    """Basic normalization for SQL matching."""
    sql = sql.replace(';', '').replace('`', '').replace('"', "'")
    sql = re.sub(r'\s+', ' ', sql).strip().lower()
    return sql

def eval_sql_task(model, tokenizer, task: dict, device: str) -> dict:
    prompt = f"Schema:\n{task['schema']}\n\nQuestion:\n{task['question']}\n\nSQL:\n"
    # Qwen-Coder-Instruct chat template
    messages = [
        {"role": "system", "content": "You are an expert SQL generator. Output only the SQL query and nothing else."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=100, do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][enc["input_ids"].shape[1]:]
    output_str = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    norm_out = normalize_sql(output_str)
    norm_exp = normalize_sql(task['expected'])
    
    # We consider it a pass if the expected SQL is contained within the output 
    # (sometimes models add markdown formatting like ```sql ... ```)
    passed = norm_exp in norm_out
    
    return {
        "task_id": task["id"],
        "passed": 1 if passed else 0,
        "output": output_str,
        "expected": task['expected']
    }

# ── Base Metrics (PPL, KL) ───────────────────────────────

PPL_EVAL_TEXT = "SELECT e.name, d.department_name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE e.salary > 50000 ORDER BY e.salary DESC;"

def compute_ppl(model, tokenizer, text: str, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())

def compute_kl_divergence(model, base_logits_cache, tokenizer, text: str, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc)
    current_logprobs = torch.log_softmax(out.logits, dim=-1)
    base_logprobs = base_logits_cache
    kl = torch.nn.functional.kl_div(current_logprobs, base_logprobs.exp(), reduction="batchmean", log_target=False)
    return float(kl.item())

# ── Payload Builders ─────────────────────────────────────

def create_naive_delta(param: torch.Tensor, scale: float = 0.02) -> torch.Tensor:
    return torch.randn_like(param, device="cpu") * scale

def create_random_lowrank_delta(param: torch.Tensor, rank: int = 8, scale: float = 0.02) -> torch.Tensor:
    out_dim, in_dim = param.shape
    A = torch.randn(out_dim, rank, dtype=param.dtype, device="cpu") * scale
    B = torch.randn(rank, in_dim, dtype=param.dtype, device="cpu") * scale
    return A @ B

# ── Helper to build routes ───────────────────────────────

def build_simulated_route(model, route_id, tenant_id, delta_fn, model_hash, out_dir, signer):
    sd = model.state_dict()
    payload_dir = os.path.join(out_dir, "payloads")
    os.makedirs(payload_dir, exist_ok=True)

    TARGET_LAYERS = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.v_proj.weight"]
    deltas = {}
    layer_specs = []
    for layer_name in TARGET_LAYERS:
        if layer_name not in sd: continue
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
            "sha256": compute_file_sha256(payload_path)
        },
        "layers": layer_specs,
    }
    signed = signer.sign(route_data, "eval-key")
    manifest_path = os.path.join(out_dir, f"{route_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(signed, f)
    return manifest_path

def build_actual_route(model, route_id, safetensors_path, tenant_id, model_hash, out_dir, signer):
    if not os.path.exists(safetensors_path): return False
    actual_sd = load_file(safetensors_path)
    sd = model.state_dict()
    payload_dir = os.path.join(out_dir, "payloads")
    os.makedirs(payload_dir, exist_ok=True)
    
    deltas = {}
    layer_specs = []
    
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
            "sha256": compute_file_sha256(payload_path)
        },
        "layers": layer_specs,
    }
    signed = signer.sign(route_data, "eval-key")
    manifest_path = os.path.join(out_dir, f"{route_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(signed, f)
    return True

# ── Main ─────────────────────────────────────────────────

def evaluate_mode(model, tokenizer, mode_name, base_logits_cache, device):
    print(f"\n  [{mode_name}]")
    results = {"mode": mode_name}

    ppl = compute_ppl(model, tokenizer, PPL_EVAL_TEXT, device)
    kl = compute_kl_divergence(model, base_logits_cache, tokenizer, PPL_EVAL_TEXT, device)
    
    results["ppl"] = round(ppl, 4)
    results["kl_divergence"] = round(kl, 6)
    print(f"    PPL: {ppl:.4f}  KL: {kl:.6f}")

    total_passed = 0
    for task in SQL_TASKS:
        res = eval_sql_task(model, tokenizer, task, device)
        total_passed += res["passed"]
        
    pass_rate = total_passed / len(SQL_TASKS)
    results["sql_pass_rate"] = pass_rate
    results["sql_passed"] = total_passed
    results["sql_total"] = len(SQL_TASKS)
    print(f"    SQL Exact/Norm Match: {total_passed}/{len(SQL_TASKS)} ({pass_rate*100:.1f}%)")
    
    return results

def main():
    print("=" * 70)
    print("  Step 3.5: Text-to-SQL Capability Transfer Evaluation")
    print(f"  Model: {MODEL_ID} | Device: {DEVICE}")
    print("=" * 70)

    out_dir = os.path.join(os.path.dirname(__file__), "sql_eval_results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    
    sd = model.state_dict()
    h = hashlib.sha256()
    h.update(sd[list(sd.keys())[0]].cpu().contiguous().numpy().tobytes()[:1024])
    model_hash = h.hexdigest()

    enc = tokenizer(PPL_EVAL_TEXT, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        base_logits = torch.log_softmax(model(**enc).logits, dim=-1).detach()

    signer = RouteSigner(SECRET_KEYS)
    registry = RouteRegistry(storage_dir=out_dir, signer=signer)
    runtime = HotSwapRuntime(model, registry, model_hash, payload_base_dir=out_dir)

    # 1. Simulated routes
    build_simulated_route(model, "naive", "t1", lambda p: create_naive_delta(p), model_hash, out_dir, signer)
    registry.register_route(os.path.join(out_dir, "naive.json"))
    
    build_simulated_route(model, "random-lr", "t1", lambda p: create_random_lowrank_delta(p), model_hash, out_dir, signer)
    registry.register_route(os.path.join(out_dir, "random-lr.json"))

    # 2. Actual LoRAs
    alpaca_path = os.path.join(os.path.dirname(__file__), "..", "routes", "actual_loras", "qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors")
    has_alpaca = build_actual_route(model, "alpaca-lora", alpaca_path, "t1", model_hash, out_dir, signer)
    if has_alpaca: registry.register_route(os.path.join(out_dir, "alpaca-lora.json"))
    
    sql_path = os.path.join(os.path.dirname(__file__), "..", "routes", "actual_loras", "Qwen2.5-Coder-0.5B-Instruct_text_to_sql_lora_newdataset_payload.safetensors")
    has_sql = build_actual_route(model, "sql-lora", sql_path, "t1", model_hash, out_dir, signer)
    if has_sql: registry.register_route(os.path.join(out_dir, "sql-lora.json"))

    all_results = []

    def apply_and_eval(route_id, mode_name):
        route_data = runtime.registry.get_route(route_id)
        runtime.lock.acquire()
        try:
            runtime.capture_and_verify(route_data)
            runtime.swap(route_data)
            r = evaluate_mode(model, tokenizer, mode_name, base_logits, DEVICE)
            runtime.rollback()
        finally:
            runtime.lock.release()
        return r

    all_results.append(evaluate_mode(model, tokenizer, "Target Base", base_logits, DEVICE))
    all_results.append(apply_and_eval("naive", "Target + Naive"))
    all_results.append(apply_and_eval("random-lr", "Target + Random LR"))
    if has_alpaca: all_results.append(apply_and_eval("alpaca-lora", "Target + Alpaca LoRA"))
    if has_sql: all_results.append(apply_and_eval("sql-lora", "Target + SQL LoRA"))
    all_results.append(evaluate_mode(model, tokenizer, "After Rollback", base_logits, DEVICE))

    print(f"\n{'='*70}")
    print(f"  {'Mode':<25} {'PPL':>8} {'KL':>10} {'SQL Match':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*12}")
    for r in all_results:
        sql_str = f"{r['sql_passed']}/{r['sql_total']}"
        print(f"  {r['mode']:<25} {r['ppl']:>8.4f} {r['kl_divergence']:>10.6f} {sql_str:>12}")

if __name__ == "__main__":
    main()
