import torch
import torch.nn.functional as F
import json
import argparse
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def logit_delta_check(base_id, adapter_path, prompts, device):
    print(f"Loading base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Try to load gamma from metadata if it exists
    gamma_val = "unknown"
    meta_path = Path(adapter_path).parent / "projection_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            gamma_val = meta.get("scale_gamma", "unknown")

    print(f"Loading adapter: {adapter_path} (Detected Gamma: {gamma_val})")
    projected_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    sql_keywords = ["SELECT", "FROM", "WHERE", "ORDER", "INSERT", "UPDATE"]
    sql_token_ids = []
    for kw in sql_keywords:
        ids = tokenizer.encode(" " + kw, add_special_tokens=False)
        if ids:
            sql_token_ids.append((kw, ids[0]))

    results = []
    total_kl_sym = 0
    total_abs_delta = 0
    total_top1_same = 0
    
    print(f"Checking logits across {len(prompts)} prompts...")
    
    with torch.no_grad():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # 1. Base Logits
            with projected_model.disable_adapter():
                base_logits = projected_model(**inputs).logits[0, -1, :].to(torch.float32)
                base_logprobs = F.log_softmax(base_logits, dim=-1)
            
            # 2. Projected Logits
            proj_logits = projected_model(**inputs).logits[0, -1, :].to(torch.float32)
            proj_logprobs = F.log_softmax(proj_logits, dim=-1)
            
            # Stable Symmetric KL Divergence
            kl_base_to_proj = F.kl_div(proj_logprobs, base_logprobs.exp(), reduction='sum').item()
            kl_proj_to_base = F.kl_div(base_logprobs, proj_logprobs.exp(), reduction='sum').item()
            sym_kl = 0.5 * (kl_base_to_proj + kl_proj_to_base)
            
            abs_delta = torch.abs(proj_logits - base_logits)
            
            base_top_val, base_top_idx = torch.topk(base_logits, 2)
            proj_top_val, proj_top_idx = torch.topk(proj_logits, 2)
            
            base_margin = (base_top_val[0] - base_top_val[1]).item()
            proj_margin = (proj_top_val[0] - proj_top_val[1]).item()
            
            sql_deltas = {}
            for kw, tid in sql_token_ids:
                sql_deltas[kw] = (proj_logprobs[tid] - base_logprobs[tid]).item()

            results.append({
                "prompt": prompt[:50] + "...",
                "symmetric_kl": sym_kl,
                "kl_base_to_proj": kl_base_to_proj,
                "kl_proj_to_base": kl_proj_to_base,
                "mean_abs_delta": abs_delta.mean().item(),
                "base_top1_token": tokenizer.decode([base_top_idx[0]]),
                "proj_top1_token": tokenizer.decode([proj_top_idx[0]]),
                "top1_match": base_top_idx[0].item() == proj_top_idx[0].item(),
                "base_margin": base_margin,
                "proj_margin": proj_margin,
                "sql_logprob_deltas": sql_deltas
            })
            
            total_kl_sym += sym_kl
            total_abs_delta += abs_delta.mean().item()
            if base_top_idx[0].item() == proj_top_idx[0].item():
                total_top1_same += 1

    n = len(prompts)
    summary = {
        "evaluation_type": "logit_level_delta_check",
        "num_prompts": n,
        "gamma": gamma_val,
        "adapter_path": str(adapter_path),
        "mean_symmetric_kl": total_kl_sym / n,
        "mean_abs_logit_delta": total_abs_delta / n,
        "top1_same_rate": total_top1_same / n,
        "verdict": "PENDING"
    }
    
    if summary["top1_same_rate"] < 1.0:
        summary["verdict"] = "TOKEN_PREFERENCE_SHIFT_DETECTED"
    elif summary["mean_symmetric_kl"] > 0.01:
        summary["verdict"] = "SUB_THRESHOLD_SIGNAL_DETECTED"
    elif summary["mean_symmetric_kl"] < 0.000001:
        summary["verdict"] = "NO_MEANINGFUL_LOGIT_DELTA"
    else:
        summary["verdict"] = "MINOR_DISTRIBUTION_SHIFT"

    return summary, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="routes/qwen05b_sql_projection/gamma_32p0/peft_adapter_calibrated")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--eval_prompts", default="eval/sql_prompts_50.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.eval_prompts):
        print(f"Error: {args.eval_prompts} not found.")
        return

    with open(args.eval_prompts, "r", encoding="utf-8") as f:
        prompts_raw = json.load(f)
    
    prompts = []
    for p in prompts_raw:
        if isinstance(p, dict):
            prompts.append(p.get("prompt", str(p)))
        else:
            prompts.append(str(p))
        
    if not prompts:
        raise ValueError("No prompts provided for logit delta check.")
        
    summary, details = logit_delta_check(args.base_model, args.adapter_path, prompts, device)
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/logit_delta_check.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    
    with open("reports/logit_delta_check.md", "w", encoding="utf-8") as f:
        f.write("# Logit-level Delta Check Report\n\n")
        f.write(f"- **Verdict**: `{summary['verdict']}`\n")
        f.write(f"- **Detected Gamma**: {summary['gamma']}\n")
        f.write(f"- **Top-1 Agreement**: {summary['top1_same_rate']:.1%}\n")
        f.write(f"- **Mean Symmetric KL**: {summary['mean_symmetric_kl']:.6f}\n")
        f.write(f"- **Mean Abs Logit Delta**: {summary['mean_abs_logit_delta']:.6f}\n\n")
        
        f.write("## SQL Keyword Shifts (Avg Logprob Delta)\n")
        avg_sql = {}
        for d in details:
            for kw, val in d["sql_logprob_deltas"].items():
                avg_sql[kw] = avg_sql.get(kw, 0) + val
        
        f.write("| Keyword | Avg Logprob Delta |\n| :--- | :---: |\n")
        for kw, val in avg_sql.items():
            f.write(f"| {kw} | {val/len(details):+.6f} |\n")

    print(f"Logit check complete. Verdict: {summary['verdict']}")

if __name__ == "__main__":
    main()
