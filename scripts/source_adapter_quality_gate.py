import argparse
import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
from neural_scalpel.core.benchmarks.sql_50 import get_sql_50_suite
from neural_scalpel.core.quality_gate import SourceAdapterQualityReport, QualityGateConfig

def validate_adapter_metadata(adapter_path, base_model_name):
    """Checks adapter_config.json for consistency."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        # Might be a HF hub path
        return "skipped_for_hub", []
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    warnings = []
    # Check base model mismatch
    config_base = config.get("base_model_name_or_path", "")
    if config_base and base_model_name not in config_base:
        warnings.append(f"Base model mismatch: config has '{config_base}', but you provided '{base_model_name}'")
        
    return "local_validated", warnings

def run_quality_gate(args):
    print(f"\n[Gate] Initializing Source Adapter Quality Gate v1.1 (Hardened)")
    
    if args.benchmark != "sql_50":
        raise ValueError(f"Benchmark '{args.benchmark}' is not yet supported in the registry.")

    # 0. Metadata Validation
    mode, warnings = validate_adapter_metadata(args.adapter_path, args.base_model)
    if warnings:
        print(f"[Gate] Metadata Warnings: {warnings}")

    print(f"[Gate] Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    
    print(f"[Gate] Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    
    evaluator = SQLCapabilityEvaluator(model, tokenizer)
    benchmark_suite = get_sql_50_suite()
    
    # 1. Evaluate Adapter (Enabled)
    print(f"\n[Gate] Step 1: Evaluating Adapter Model...")
    adapter_results = evaluator.evaluate_suite(benchmark_suite)
    
    # 2. Evaluate Base (Disabled)
    print(f"\n[Gate] Step 2: Evaluating Base Model...")
    with model.disable_adapter():
        base_results = evaluator.evaluate_suite(benchmark_suite)
    
    # 3. Analyze results
    print(f"\n[Gate] Step 3: Diagnostic Analysis...")
    
    report = SourceAdapterQualityReport(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        benchmark=args.benchmark,
        task_type="sql"
    )
    
    report.total_cases = len(benchmark_suite)
    report.base_metrics = {
        "execution_accuracy": base_results["stats"]["execution_accuracy"],
        "execution_success": base_results["stats"]["execution_success_rate"],
        "syntax_validity": base_results["stats"]["syntax_valid"] / base_results["stats"]["total"]
    }
    report.adapter_metrics = {
        "execution_accuracy": adapter_results["stats"]["execution_accuracy"],
        "execution_success": adapter_results["stats"]["execution_success_rate"],
        "syntax_validity": adapter_results["stats"]["syntax_valid"] / adapter_results["stats"]["total"]
    }
    
    for m in report.base_metrics:
        report.delta[m] = report.adapter_metrics[m] - report.base_metrics[m]
        
    # Classification
    base_correct_ids = {res["id"] for res in base_results["results"] if res["is_correct"]}
    adapter_correct_ids = {res["id"] for res in adapter_results["results"] if res["is_correct"]}
    
    fixed = len(adapter_correct_ids - base_correct_ids)
    regressed = len(base_correct_ids - adapter_correct_ids)
    
    report.failure_classification = {
        "fixed": fixed,
        "regressed": regressed,
        "both_succeeded": len(base_correct_ids & adapter_correct_ids),
        "both_failed": report.total_cases - len(base_correct_ids | adapter_correct_ids)
    }
    report.regression_rate = regressed / report.total_cases if report.total_cases > 0 else 0
    
    # Stability Check
    empty_count = 0
    repetition_count = 0
    for res in adapter_results["results"]:
        sql = res.get("generated", "").strip()
        if not sql:
            empty_count += 1
            continue
            
        # Enhanced word-based repetition check
        tokens = sql.split()
        if len(tokens) >= 8:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.35: # High repetition
                repetition_count += 1
            
    report.stability = {
        "collapse_detected": 1.0 if (report.adapter_metrics["execution_success"] < 0.1 and report.base_metrics["execution_success"] > 0.5) else 0.0,
        "empty_output_rate": empty_count / report.total_cases,
        "repetition_rate": repetition_count / report.total_cases,
        "regression_rate": report.regression_rate
    }
    
    report.metadata = {
        "base_eval_mode": "peft_disable_adapter",
        "metadata_validation_mode": mode,
        "warnings": warnings,
        "torch_dtype": str(model_kwargs["torch_dtype"])
    }
    
    # 4. Generate Verdict
    config = QualityGateConfig(
        primary_metric=args.primary_metric,
        positive_delta_threshold=args.positive_delta_threshold,
        weak_positive_threshold=args.weak_positive_threshold,
        negative_delta_threshold=args.negative_delta_threshold,
        max_regression_rate=args.max_regression_rate,
        max_empty_output_rate=args.max_empty_output_rate,
        max_repetition_rate=args.max_repetition_rate
    )
    report.generate_verdict(config)
    
    # 5. Output
    print(f"\n--- Quality Gate Summary ---")
    print(f"Verdict: {report.verdict}")
    print(f"Status: {report.gate_status}")
    print(f"Recommendation: {report.recommendation}")
    print(f"Accuracy Delta: {report.delta['execution_accuracy']*100:+.1f}%")
    
    if args.output:
        report.to_json(args.output)
        md_path = args.output.replace(".json", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        print(f"\n[Gate] Reports saved to {args.output} and {md_path}")

    if args.comparisons:
        # Save comprehensive comparison (includes both_failed)
        comparison = []
        base_dict = {res["id"]: res for res in base_results["results"]}
        adapter_dict = {res["id"]: res for res in adapter_results["results"]}
        
        for cid in {res["id"] for res in base_results["results"]}:
            b = base_dict[cid]
            a = adapter_dict[cid]
            
            # Classification logic
            if a["is_correct"] and not b["is_correct"]: cls = "fixed"
            elif b["is_correct"] and not a["is_correct"]: cls = "regressed"
            elif not b["is_correct"] and not a["is_correct"]: cls = "both_failed"
            else: cls = "both_succeeded"

            comparison.append({
                "id": cid,
                "category": b.get("category"),
                "classification": cls,
                "base_correct": b["is_correct"],
                "adapter_correct": a["is_correct"],
                "base_output": b["generated"],
                "adapter_output": a["generated"]
            })
        
        with open(args.comparisons, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"[Gate] Comprehensive comparison saved to {args.comparisons}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Adapter Quality Gate v1.1 (Hardened)")
    parser.add_argument("--base_model", type=str, required=True, help="Path/ID of the original teacher base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path/ID of the teacher LoRA adapter")
    parser.add_argument("--benchmark", type=str, default="sql_50", help="Benchmark to use (currently only sql_50)")
    parser.add_argument("--output", type=str, help="Path to save the JSON/Markdown report")
    parser.add_argument("--comparisons", type=str, help="Path to save comprehensive case-by-case comparison (replaces --failures)")
    
    # Threshold args
    parser.add_argument("--primary_metric", type=str, default="execution_accuracy")
    parser.add_argument("--positive_delta_threshold", type=float, default=0.03)
    parser.add_argument("--weak_positive_threshold", type=float, default=0.00)
    parser.add_argument("--negative_delta_threshold", type=float, default=-0.01)
    parser.add_argument("--max_regression_rate", type=float, default=0.10)
    parser.add_argument("--max_empty_output_rate", type=float, default=0.05)
    parser.add_argument("--max_repetition_rate", type=float, default=0.10)
    
    args = parser.parse_args()
    run_quality_gate(args)
