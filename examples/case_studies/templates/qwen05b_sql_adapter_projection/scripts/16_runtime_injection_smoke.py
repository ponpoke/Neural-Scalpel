import os
import torch
import torch.nn.functional as F
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def symmetric_kl(logits_a, logits_b):
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    kl_pq = F.kl_div(log_p, log_q, reduction='batchmean', log_target=True)
    kl_qp = F.kl_div(log_q, log_p, reduction='batchmean', log_target=True)
    return (kl_pq + kl_qp).item() / 2.0

def runtime_injection_smoke(model, tokenizer, adapter_weights, prompts, gamma=1.0):
    input_device = model.get_input_embeddings().weight.device
    hooks = []
    
    def get_injection_hook(W_cpu, scale):
        def hook(module, input, output):
            # output can be (hidden_states, ...) or hidden_states Tensor directly
            is_tuple = isinstance(output, (tuple, list))
            h = output[0] if is_tuple else output
            
            # Dynamic device adaptation (supports device_map='auto')
            W = W_cpu.to(device=h.device, dtype=h.dtype)
            
            # Robust Last-token only injection
            delta = torch.zeros_like(h)
            if h.dim() == 3:
                # (batch, seq, hidden)
                delta[:, -1, :] = torch.matmul(h[:, -1, :], W) * scale
            elif h.dim() == 2:
                # (seq, hidden)
                delta[-1, :] = torch.matmul(h[-1, :], W) * scale
            
            # Return in same format as input
            if is_tuple:
                return (h + delta,) + output[1:]
            else:
                return h + delta
        return hook

    # Register Hooks
    for layer_name, info in adapter_weights.items():
        idx = int(layer_name.split(".")[-1])
        target_layer = model.model.layers[idx]
        hooks.append(target_layer.register_forward_hook(get_injection_hook(info["weight"], gamma)))

    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {k: v.to(input_device) for k, v in tokenizer(text, return_tensors="pt").items()}
        
        # 1. Run Injected
        with torch.no_grad():
            outputs_injected = model(**inputs)
            logits_injected = outputs_injected.logits[:, -1, :]
            
        # 2. Base Run (Temporarily remove hooks)
        for h in hooks: h.remove()
        with torch.no_grad():
            outputs_base = model(**inputs)
            logits_base = outputs_base.logits[:, -1, :]
            
        # Re-register for next prompt
        hooks.clear()
        for layer_name, info in adapter_weights.items():
            idx = int(layer_name.split(".")[-1])
            target_layer = model.model.layers[idx]
            hooks.append(target_layer.register_forward_hook(get_injection_hook(info["weight"], gamma)))

        # Compare
        kl = symmetric_kl(logits_base, logits_injected)
        top1_base = torch.argmax(logits_base, dim=-1).item()
        top1_inj = torch.argmax(logits_injected, dim=-1).item()
        
        results.append({
            "prompt_preview": prompt[:80],
            "kl_divergence": kl,
            "top1_changed": top1_base != top1_inj,
            "top1_base_token": tokenizer.decode([top1_base]),
            "top1_inj_token": tokenizer.decode([top1_inj])
        })

    for h in hooks: h.remove()
    
    # Calculate Verdict for this Gamma
    mean_kl = sum(r["kl_divergence"] for r in results) / len(results) if results else 0
    changed_count = sum(1 for r in results if r["top1_changed"])
    
    status = "NO_INJECTION_SIGNAL"
    if changed_count > 0:
        status = "BEHAVIORAL_SHIFT_DETECTED"
    elif mean_kl > 1e-4:
        status = "INJECTION_SIGNAL_OBSERVED"
        
    return {
        "gamma": gamma,
        "mean_kl": mean_kl,
        "behavioral_shift_count": changed_count,
        "status": status,
        "results": results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default="routes/qwen05b_sql_projection/activation_adapter/activation_adapter_weights.pt")
    parser.add_argument("--eval_prompts", default="eval/sql_prompts_50.json")
    parser.add_argument("--gammas", default="0.1,0.5,1.0,2.0")
    args = parser.parse_args()
    
    if not os.path.exists(args.eval_prompts):
        print(f"Error: {args.eval_prompts} not found.")
        return

    with open(args.eval_prompts, "r", encoding="utf-8") as f:
        prompts_raw = json.load(f)
    prompts = [p.get("prompt", str(p)) if isinstance(p, dict) else str(p) for p in prompts_raw[:4]]
    
    if not prompts:
        raise ValueError("No prompts loaded for runtime injection smoke test.")

    print(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    print(f"Loading Activation Adapter Weights: {args.adapter_path}")
    adapter_weights = torch.load(args.adapter_path)
    
    report = {"evaluation_type": "runtime_injection_gamma_sweep", "gamma_results": []}
    
    for g_str in args.gammas.split(","):
        gamma = float(g_str)
        print(f"Running Smoke Test with Gamma={gamma}...")
        res = runtime_injection_smoke(model, tokenizer, adapter_weights, prompts, gamma=gamma)
        report["gamma_results"].append(res)
        print(f"  Status: {res['status']} | Mean KL: {res['mean_kl']:.6f} | Shifts: {res['behavioral_shift_count']}/4")

    output_path = "routes/qwen05b_sql_projection/analysis/injection_smoke_sweep.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SUCCESS] Sweep complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
