import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.benchmarks.registry import BenchmarkRegistry
from neural_scalpel.core.diagnostic import AdapterTransferDiagnosticReport, TargetEvaluationResult
from pathlib import Path

def run_evaluate(args):
    print(f"[Eval] Starting Target Evaluation Gate (v2.1)...")
    print(f" - Target Base: {args.target_model}")
    print(f" - Projected Adapter: {args.adapter}")
    
    # Strict report check
    if args.report_path:
        report_file = Path(args.report_path)
        if not report_file.exists():
            raise FileNotFoundError(f"Diagnostic report not found: {args.report_path}. "
                                    f"Please run 'diagnose-adapter' first or check the path.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 1. Evaluate Projected Adapter
    print(f" - Pass 1/2: Evaluating Projected Adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    
    evaluator = BenchmarkRegistry.get_evaluator(args.benchmark, model, tokenizer)
    suite = BenchmarkRegistry.get_suite(args.benchmark)
    adapter_results = evaluator.evaluate_suite(suite)
    
    # 2. Evaluate Baseline
    print(f" - Pass 2/2: Evaluating Target Base Model...")
    with model.disable_adapter():
        base_results = evaluator.evaluate_suite(suite)
        
    # 3. Calculate Delta and Classification
    print(f" - Analyzing behavioral delta...")
    t_eval = TargetEvaluationResult()
    t_eval.total_cases = len(suite)
    
    t_eval.base_metrics = {
        "execution_accuracy": base_results["stats"]["execution_accuracy"],
        "execution_success": base_results["stats"]["execution_success_rate"]
    }
    t_eval.adapter_metrics = {
        "execution_accuracy": adapter_results["stats"]["execution_accuracy"],
        "execution_success": adapter_results["stats"]["execution_success_rate"]
    }
    for m in t_eval.base_metrics:
        t_eval.delta[m] = t_eval.adapter_metrics[m] - t_eval.base_metrics[m]
        
    base_correct_ids = {res["id"] for res in base_results["results"] if res["is_correct"]}
    adapter_correct_ids = {res["id"] for res in adapter_results["results"] if res["is_correct"]}
    
    t_eval.failure_classification = {
        "fixed": len(adapter_correct_ids - base_correct_ids),
        "regressed": len(base_correct_ids - adapter_correct_ids),
        "both_succeeded": len(base_correct_ids & adapter_correct_ids),
        "both_failed": t_eval.total_cases - len(base_correct_ids | adapter_correct_ids)
    }
    t_eval.regression_rate = t_eval.failure_classification["regressed"] / t_eval.total_cases
    
    # Verdict Logic (Improved with thresholds)
    if (t_eval.delta["execution_accuracy"] > args.positive_delta_threshold 
        and t_eval.regression_rate <= args.max_regression_rate):
        t_eval.verdict = "POSITIVE_TARGET_TRANSFER"
    elif t_eval.delta["execution_accuracy"] >= 0:
        t_eval.verdict = "NEUTRAL_TARGET_TRANSFER"
    else:
        t_eval.verdict = "TARGET_INTERFERENCE"
        
    # 4. Integrate with Diagnostic Report
    if args.report_path:
        print(f" - Updating diagnostic report: {args.report_path}")
        report = AdapterTransferDiagnosticReport.from_json(args.report_path)
        report.target_evaluation_gate = t_eval
        report.finalize_release_decision()
        report.save(args.report_path)
        
        print(f"\n--- Final Release Decision: {report.release_decision_gate.verdict} ---")
        print(f"Recommendation: {report.release_decision_gate.recommendation}")
        for reason in report.release_decision_gate.reasons:
            print(f" - {reason}")
    
    # 5. Save detailed results (Including base_results for analysis)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "target_evaluation": t_eval.__dict__,
        "base_results": base_results,
        "adapter_results": adapter_results
    }
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
        
    print(f"\n[Eval] Target evaluation complete. Results saved to {output_path}")

def add_evaluate_projected_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-projected",
        help="Evaluate a projected adapter on target model benchmarks (v2.1)."
    )

    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model")
    parser.add_argument("--adapter", required=True,
                        help="Path to projected adapter")
    parser.add_argument("--benchmark", default="sql_50",
                        help="Benchmark to run")
    parser.add_argument("--output", default="reports/target_eval/eval_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--report", dest="report_path",
                        help="Path to existing diagnostic_report.json to update")
    
    # v2.1 threshold tuning
    parser.add_argument("--positive-delta-threshold", type=float, default=0.0,
                        help="Minimum accuracy improvement required for POSITIVE verdict")
    parser.add_argument("--max-regression-rate", type=float, default=0.05,
                        help="Maximum allowed regression rate (0.0 to 1.0)")

    parser.set_defaults(func=run_evaluate)
