import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
from neural_scalpel.core.benchmarks.sql_50 import get_sql_50_suite
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel Phase 6: Full SQL-50 Evaluation")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to transplanted PEFT adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="sql_eval_results_full_50.json")
    parser.add_argument("--failures", type=str, default="failure_cases.json")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        device_map=args.device
    )

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    test_cases = get_sql_50_suite()
    print(f"\nRunning Full SQL-50 Evaluation on {len(test_cases)} cases...")

    # 1. Evaluate Adapter
    adapter_evaluator = SQLCapabilityEvaluator(model, tokenizer)
    adapter_res = adapter_evaluator.evaluate_suite(test_cases)
    a_stats = adapter_res["stats"]

    # 2. Evaluate Baseline
    print("Evaluating Baseline (Adapter Disabled)...")
    with model.disable_adapter():
        baseline_evaluator = SQLCapabilityEvaluator(model, tokenizer)
        baseline_res = baseline_evaluator.evaluate_suite(test_cases)
        b_stats = baseline_res["stats"]

    # 3. Consolidate and Calculate Category Deltas
    final_output = {
        "metadata": {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "num_cases": len(test_cases)
        },
        "adapter_stats": a_stats,
        "baseline_stats": b_stats,
        "comparison": {
            "overall": {
                "execution_success_delta": a_stats["execution_success_rate"] - b_stats["execution_success_rate"],
                "execution_accuracy_delta": a_stats["execution_accuracy"] - b_stats["execution_accuracy"]
            },
            "categories": {}
        }
    }

    # Detailed category deltas
    for cat in a_stats["categories"]:
        a_cat = a_stats["categories"][cat]
        b_cat = b_stats["categories"].get(cat, {"pass": 0, "correct": 0, "total": 1})
        
        final_output["comparison"]["categories"][cat] = {
            "success_delta": (a_cat["pass"] - b_cat["pass"]) / a_cat["total"],
            "accuracy_delta": (a_cat["correct"] - b_cat["correct"]) / a_cat["total"]
        }

    # Summary Printing
    print("\n--- Evaluation Summary (SQL-50) ---")
    print(f"Adapter Accuracy:  {a_stats['execution_accuracy']:.1%}")
    print(f"Baseline Accuracy: {b_stats['execution_accuracy']:.1%}")
    print(f"Net Improvement:   {final_output['comparison']['overall']['execution_accuracy_delta']:+.1%}")

    print("\n--- Category Deltas ---")
    for cat, deltas in final_output["comparison"]["categories"].items():
        print(f"  {cat:15}: {deltas['accuracy_delta']:+.1%}")

    # Save Results
    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    
    # Save Failures (Adapter Only for now, or both)
    failures = {
        "adapter_failures": adapter_res["failure_cases"],
        "baseline_failures": baseline_res["failure_cases"]
    }
    with open(args.failures, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Failure cases saved to {args.failures}")

if __name__ == "__main__":
    main()
