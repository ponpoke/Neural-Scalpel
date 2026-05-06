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
        "behavioral_improvement": "INCONCLUSIVE",
        "base": {
            "sql_signal": 0,
            "repetition_failure": 0,
            "avg_length": 0
        },
        "projected": {
            "sql_signal": 0,
            "repetition_failure": 0,
            "avg_length": 0
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

        # Base
        metrics["base"]["avg_length"] += len(base_out)
        if any(kw in base_out.upper() for kw in sql_keywords):
            metrics["base"]["sql_signal"] += 1
        lines = base_out.split("\n")
        if len(lines) > 5 and len(set(lines)) < len(lines) * 0.5:
            metrics["base"]["repetition_failure"] += 1

        # Projected
        metrics["projected"]["avg_length"] += len(proj_out)
        if any(kw in proj_out.upper() for kw in sql_keywords):
            metrics["projected"]["sql_signal"] += 1
        lines = proj_out.split("\n")
        if len(lines) > 5 and len(set(lines)) < len(lines) * 0.5:
            metrics["projected"]["repetition_failure"] += 1

    # Normalize rates
    for key in ["base", "projected"]:
        metrics[key]["sql_signal_rate"] = metrics[key]["sql_signal"] / n
        metrics[key]["repetition_rate"] = metrics[key]["repetition_failure"] / n
        metrics[key]["avg_length"] = metrics[key]["avg_length"] / n

    metrics["exact_same_rate_raw"] = metrics["exact_same_count_raw"] / n
    metrics["exact_same_rate_normalized"] = metrics["exact_same_count_normalized"] / n
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
        f.write(f"| Avg Output Length | {metrics['base']['avg_length']:.0f} | {metrics['projected']['avg_length']:.0f} | {metrics['projected']['avg_length'] - metrics['base']['avg_length']:+.1f} |\n")
        
        f.write(f"\n**Identity Rates (Base vs Projected):**\n")
        f.write(f"- Exact Bit-Identical: {metrics['exact_same_rate_raw']:.1%}\n")
        f.write(f"- Normalized (Whitespace-Insensitive): {metrics['exact_same_rate_normalized']:.1%}\n")

    print(f"Markdown report saved to {report_md}")

if __name__ == "__main__":
    main()
