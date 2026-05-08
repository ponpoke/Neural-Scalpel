import os
import json
import argparse
from pathlib import Path

def eval_sql_metrics(real_eval=False):
    mode = "REAL" if real_eval else "SIMULATED"
    print(f"[Phase 5] Performing quantitative SQL metric evaluation ({mode})")
    
    if real_eval:
        raise NotImplementedError(
            "Real metric calculation is not implemented in this scaffold. "
            "Generate real outputs (e.g. base_outputs.json, projected_outputs.json), then use a metrics evaluator."
        )
        
    # Simulate metrics on 50 prompts
    metrics = {
        "num_prompts": 50,
        "mode": "SIMULATED",
        "base": {
            "syntax_validity_rate": 0.82,
            "execution_accuracy": 0.64,
            "exact_match": 0.58,
            "judge_preference": 0.40
        },
        "projected": {
            "syntax_validity_rate": 0.94,
            "execution_accuracy": 0.78,
            "exact_match": 0.72,
            "judge_preference": 0.60
        },
        "delta": {
            "syntax_validity_rate": "+12%",
            "execution_accuracy": "+14%",
            "exact_match": "+14%",
            "judge_preference": "+20%"
        }
    }
    
    report_json_path = Path("reports/eval_summary.json")
    report_md_path = Path("reports/eval_summary.md")
    os.makedirs(report_json_path.parent, exist_ok=True)
    
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Quantitative Evaluation Summary ({mode})\n\n")
        if not real_eval:
            f.write(f"> [!CAUTION]\n")
            f.write(f"> **SIMULATED METRICS**: Do not use as real evaluation results.\n")
            f.write(f"> These numbers are placeholders from the case-study scaffold.\n\n")
            
        f.write(f"Evaluated on {metrics['num_prompts']} curated SQL/Coding prompts.\n\n")
        f.write("| Metric | Base Qwen2.5-0.5B | Projected SQL Route | Delta |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| Syntax Validity | {metrics['base']['syntax_validity_rate']:.0%} | {metrics['projected']['syntax_validity_rate']:.0%} | {metrics['delta']['syntax_validity_rate']} |\n")
        f.write(f"| Execution Accuracy | {metrics['base']['execution_accuracy']:.0%} | {metrics['projected']['execution_accuracy']:.0%} | {metrics['delta']['execution_accuracy']} |\n")
        f.write(f"| Exact Match | {metrics['base']['exact_match']:.0%} | {metrics['projected']['exact_match']:.0%} | {metrics['delta']['exact_match']} |\n")
        f.write(f"| Judge Preference | {metrics['base']['judge_preference']:.0%} | {metrics['projected']['judge_preference']:.0%} | {metrics['delta']['judge_preference']} |\n")
        
    # Also create a failure cases report
    failure_cases_path = Path("reports/failure_cases.md")
    with open(failure_cases_path, "w", encoding="utf-8") as f:
        f.write("# Observed Failure Cases\n\n")
        if not real_eval:
            f.write(f"> [!NOTE]\n")
            f.write(f"> These failure cases are illustrative examples for the scaffold.\n\n")
        f.write("- **Schema Hallucination:** In 2 cases, the route referenced table names not present in the prompt.\n")
        f.write("- **Redundant Aliasing:** Occasional over-aliasing in simple SELECT queries.\n")
        f.write("- **Logic Drift:** One case where a JOIN was preferred over a subquery, resulting in a correct but less efficient plan.\n")

    print(f"Metrics saved to {report_json_path} and {report_md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Perform real metric evaluation")
    args = parser.parse_args()

    eval_sql_metrics(real_eval=args.real)
