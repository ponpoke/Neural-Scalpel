import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
from pathlib import Path
import argparse

def run_inference(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="routes/qwen05b_sql_projection/peft_adapter")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    prompts = [
        "Table: sales(id, product, amount, date). SQL to find total sales for 'Laptop'.",
        "Table: employees(id, name, department, salary). SQL for average salary in 'Engineering'.",
        "Write a SQL query to list all students who scored above 90 in 'Mathematics'.",
        "Write a Python function for palindrome check."
    ]

    results = []

    print("\n--- Running Base Model Inference ---")
    for p in prompts:
        print(f"Prompt: {p}")
        out = run_inference(model, tokenizer, p, device)
        # Remove prompt from output
        clean_out = out.replace(p, "").strip()
        results.append({
            "prompt": p,
            "base_output": clean_out
        })

    print(f"\nLoading PEFT adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("\n--- Running Projected Adapter Inference ---")
    for i, p in enumerate(prompts):
        print(f"Prompt: {p}")
        out = run_inference(model, tokenizer, p, device)
        clean_out = out.replace(p, "").strip()
        results[i]["projected_output"] = clean_out

    # Save to JSON for report generation
    output_json = Path("reports/real_eval_results.json")
    os.makedirs(output_json.parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReal evaluation results saved to {output_json}")

    # Also generate a markdown report
    report_md = Path("reports/before_after_real.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Qualitative Before/After Evaluation (REAL)\n\n")
        f.write(f"**Base Model:** {args.base_model}\n")
        f.write(f"**Adapter:** {args.adapter_path}\n\n")
        
        for i, res in enumerate(results):
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Prompt:**\n```\n{res['prompt']}\n```\n\n")
            f.write(f"**Base Output:**\n```\n{res['base_output']}\n```\n\n")
            f.write(f"**Projected Output:**\n```\n{res['projected_output']}\n```\n\n")
            f.write("---\n\n")
    
    print(f"Markdown report saved to {report_md}")

if __name__ == "__main__":
    main()
