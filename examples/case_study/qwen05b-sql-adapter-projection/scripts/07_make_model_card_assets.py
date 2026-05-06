import os
import json
from pathlib import Path

def make_model_card():
    print("[Phase 7] Generating Hugging Face model card assets")
    
    # Check if reports are simulated
    is_simulated = True
    try:
        with open("reports/eval_summary.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
            if metrics.get("mode") == "REAL":
                is_simulated = False
    except:
        pass

    # Read data from reports
    try:
        with open("reports/eval_summary.md", "r", encoding="utf-8") as f:
            eval_summary = f.read()
    except:
        eval_summary = "Evaluation results pending."
        
    try:
        with open("reports/before_after.md", "r", encoding="utf-8") as f:
            before_after = f.read()
    except:
        before_after = "Before/After examples pending."

    try:
        with open("reports/failure_cases.md", "r", encoding="utf-8") as f:
            failure_cases = f.read()
    except:
        failure_cases = "Failure cases pending."

    # Create HF Card README
    hf_readme_path = Path("hf_card/README.md")
    os.makedirs(hf_readme_path.parent, exist_ok=True)
    
    with open(hf_readme_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-0.5B SQL Adapter Projection via Neural-Scalpel\n\n")
        
        if is_simulated:
            f.write("> [!CAUTION]\n")
            f.write("> **SIMULATED CASE-STUDY SCAFFOLD**: Real-weight evaluation has NOT been completed yet.\n")
            f.write("> The metrics and examples below are placeholders for demonstration purposes.\n\n")
            
        f.write("## What is this?\n\n")
        f.write("This is an experimental projected SQL/Coding adapter for Qwen2.5-0.5B, produced with Neural-Scalpel.\n\n")
        f.write("It tests whether task behavior from a larger Qwen-family SQL/Coding adapter can survive projection into a much smaller 0.5B target model without gradient-based retraining.\n\n")
        f.write("## Key Result\n\n")
        f.write(eval_summary)
        f.write("\n\n")
        f.write("## Before / After\n\n")
        f.write(before_after)
        f.write("\n\n")
        f.write("## Failure Cases\n\n")
        f.write(failure_cases)
        f.write("\n\n")
        f.write("## Limitations\n\n")
        f.write("- Downstream task improvement is not guaranteed.\n")
        f.write("- This is a projected adapter experiment, not model distillation.\n")
        f.write("- Use only after downstream validation.\n")
        
    # Copy other assets
    with open("hf_card/before_after.md", "w", encoding="utf-8") as f:
        f.write(before_after)
    with open("hf_card/failure_cases.md", "w", encoding="utf-8") as f:
        f.write(failure_cases)

    print(f"Model card assets generated in {hf_readme_path.parent}")

if __name__ == "__main__":
    make_model_card()
