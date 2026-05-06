import json
import os
from pathlib import Path
import argparse

def normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())

def calculate_metrics(results):
    n = len(results)
    if n == 0:
        raise ValueError("No evaluation results found. Ensure 04_eval_before_after.py ran correctly.")

    metrics = {
        "num_prompts": n,
        "evaluation_type": "preliminary_heuristic_smoke",
        "behavioral_status": "PENDING",
        "base": {
            "sql_signal": 0,
            "repetition_failure": 0,
            "avg_length": 0,
            "empty_outputs": 0,
            "max_length_hits": 0
        },
        "projected": {
            "sql_signal": 0,
            "repetition_failure": 0,
            "avg_length": 0,
            "empty_outputs": 0,
            "max_length_hits": 0
        },
        "exact_same_count_raw": 0,
        "exact_same_count_normalized": 0
    }

    sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "SUM(", "AVG("]

    for res in results:
        base_out = res["base_output"]
        proj_out = res["projected_output"]
        
        if base_out == proj_out:
            metrics["exact_same_count_raw"] += 1
        
        if normalize_text(base_out) == normalize_text(proj_out):
            metrics["exact_same_count_normalized"] += 1

        # Process Base & Projected
        for key, output in [("base", base_out), ("projected", proj_out)]:
            metrics[key]["avg_length"] += len(output)
            
            # SQL signal
            if any(kw in output.upper() for kw in sql_keywords):
                metrics[key]["sql_signal"] += 1
            
            # Repetition
            lines = output.split("\n")
            if len(lines) > 5 and len(set(lines)) < len(lines) * 0.5:
                metrics[key]["repetition_failure"] += 1
            
            # Empty / Collapse
            if len(output.strip()) == 0:
                metrics[key]["empty_outputs"] += 1
                
            # Max length hit (Heuristic ONLY)
            if len(output) > 500: 
                metrics[key]["max_length_hits"] += 1

    # Normalize rates
    for key in ["base", "projected"]:
        metrics[key]["sql_signal_rate"] = metrics[key]["sql_signal"] / n
        metrics[key]["repetition_rate"] = metrics[key]["repetition_failure"] / n
        metrics[key]["empty_output_rate"] = metrics[key]["empty_outputs"] / n
        metrics[key]["max_length_hit_rate"] = metrics[key]["max_length_hits"] / n
        metrics[key]["avg_length"] = metrics[key]["avg_length"] / n

    metrics["exact_same_rate_raw"] = metrics["exact_same_count_raw"] / n
    metrics["exact_same_rate_normalized"] = metrics["exact_same_count_normalized"] / n
    
    # --- Refined Behavioral Status Logic ---
    same_rate = metrics["exact_same_rate_raw"]
    empty_rate = metrics["projected"]["empty_output_rate"]
    rep_rate = metrics["projected"]["repetition_rate"]
    sql_inc = metrics["projected"]["sql_signal_rate"] > metrics["base"]["sql_signal_rate"]
    
    status = "IDENTICAL_TO_BASE"
    if same_rate < 0.95:
        if empty_rate > 0.5:
            status = "COLLAPSE (Empty Outputs)"
        elif rep_rate > 0.5:
            status = "DEGENERATION (Repetition Loop)"
        elif sql_inc and empty_rate < 0.1 and rep_rate < 0.2:
            status = "SIGNAL_CANDIDATE (Behavioral Divergence)"
        else:
            status = "NON_IDENTICAL_OUTPUT_OBSERVED"
            
    metrics["behavioral_status"] = status
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", default="reports/real_eval_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"Error: {args.results_json} not found. Run 04_eval_before_after.py first.")
        return

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics = calculate_metrics(results)

    # Save summary
    summary_path = Path("reports/eval_summary_real.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Summary metrics saved to {summary_path}")

    # Generate Markdown table
    report_md = Path("reports/eval_summary_real.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Summary (REAL SMOKE)\n\n")
        f.write("> [!WARNING]\n")
        f.write("> This is a **preliminary heuristic smoke test**. Results are inconclusive regarding SQL capability improvement.\n\n")
        f.write(f"Evaluated on {metrics['num_prompts']} curated prompts.\n\n")
        f.write("| Metric | Base Qwen2.5-0.5B | Projected SQL Route | Delta |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        f.write(f"| Basic SQL Signal (Heuristic) | {metrics['base']['sql_signal_rate']:.0%} | {metrics['projected']['sql_signal_rate']:.0%} | {metrics['projected']['sql_signal_rate'] - metrics['base']['sql_signal_rate']:+.0%} |\n")
        f.write(f"| Observed Repetition Rate | {metrics['base']['repetition_rate']:.0%} | {metrics['projected']['repetition_rate']:.0%} | {metrics['projected']['repetition_rate'] - metrics['base']['repetition_rate']:+.0%} |\n")
        f.write(f"| Empty Output Rate | {metrics['base']['empty_output_rate']:.0%} | {metrics['projected']['empty_output_rate']:.0%} | {metrics['projected']['empty_output_rate'] - metrics['base']['empty_output_rate']:+.0%} |\n")
        f.write(f"| Avg Output Length | {metrics['base']['avg_length']:.0f} | {metrics['projected']['avg_length']:.0f} | {metrics['projected']['avg_length'] - metrics['base']['avg_length']:+.1f} |\n")
        
        f.write(f"\n**Identity Rates (Base vs Projected):**\n")
        f.write(f"- Exact Bit-Identical: {metrics['exact_same_rate_raw']:.1%}\n")
        f.write(f"- Normalized (Whitespace-Insensitive): {metrics['exact_same_rate_normalized']:.1%}\n")
        f.write(f"- **Behavioral Status**: {metrics['behavioral_status']}\n")

    print(f"Markdown report saved to {report_md}")

if __name__ == "__main__":
    main()
