import os
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
from pathlib import Path
import argparse
import gc

def run_inference(model, tokenizer, prompt, device):
    # Use Chat Template for Instruct models
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False, # Greedy for comparison
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice to get only generated tokens
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="routes/qwen05b_sql_projection/peft_adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    prompts = [
        "Table: sales(id, product, amount, date). SQL to find total sales for 'Laptop'.",
        "Table: employees(id, name, department, salary). SQL for average salary in 'Engineering'.",
        "Write a SQL query to list all students who scored above 90 in 'Mathematics'.",
        "Write a Python function for palindrome check."
    ]

    results = []

    # --- Phase A: Base Model ---
    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto")
    
    print("\n--- Running Base Model Inference (Greedy) ---")
    for p in prompts:
        print(f"Prompt: {p}")
        out = run_inference(model, tokenizer, p, device)
        results.append({
            "prompt": p,
            "base_output": out.strip()
        })

    # Memory cleanup before adapter loading
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Phase B: Projected Adapter ---
    print(f"\nLoading base model again for adapter: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto")
    print(f"Loading PEFT adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("\n--- Running Projected Adapter Inference (Greedy) ---")
    for i, p in enumerate(prompts):
        print(f"Prompt: {p}")
        out = run_inference(model, tokenizer, p, device)
        results[i]["projected_output"] = out.strip()

    # Save to JSON
    output_json = Path("reports/real_eval_results.json")
    os.makedirs(output_json.parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReal evaluation results saved to {output_json}")

    # Save to Markdown for report synchronization
    report_md = Path("reports/before_after_real.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Qualitative Before/After Evaluation (REAL)\n\n")
        f.write(f"**Base Model:** {args.base_model}\n")
        f.write(f"**Adapter:** {args.adapter_path}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> This report is generated using greedy decoding and chat templates for objective comparison.\n\n")
        
        for i, res in enumerate(results):
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Prompt:**\n```\n{res['prompt']}\n```\n\n")
            f.write(f"**Base Output:**\n```\n{res['base_output']}\n```\n\n")
            f.write(f"**Projected Output:**\n```\n{res['projected_output']}\n```\n\n")
            f.write("---\n\n")
    
    print(f"Qualitative Markdown report saved to {report_md}")

if __name__ == "__main__":
    main()
