import os
from pathlib import Path
from neural_scalpel.core.diagnostic_runner import DiagnosticRunner
from neural_scalpel.commands.project_adapter import run_project
from neural_scalpel.commands.evaluate_projected import run_evaluate

class SimpleArgs:
    """Mock args object to call other command functions."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_safe_project(args):
    print(f"\n" + "="*60)
    print(f" Neural-Scalpel Safe-Project Orchestrator (v2.2)")
    print(f"="*60 + "\n")

    # Step 1: Diagnose
    print(f"[*] Stage 1: Running Multi-Stage Diagnostics...")
    report = DiagnosticRunner.execute(args)
    report_path = Path(args.output_dir) / "diagnostic_report.json"
    
    verdict = report.release_decision_gate.verdict
    print(f"\n[Diag] Diagnostic Verdict: {verdict}")
    
    if verdict not in ["PROJECTION_CANDIDATE", "RELEASE_READY"] and not args.force:
        print(f"\n[Aborting] Safe-Project stopped. Verdict is {verdict}.")
        print(f"Recommendation: {report.release_decision_gate.recommendation}")
        return

    # Step 2: Project
    print(f"\n[*] Stage 2: Structural Weight Projection...")
    projected_path = Path(args.output_dir) / "projected_adapter"
    project_args = SimpleArgs(
        source_adapter=args.source_adapter,
        target_model=args.target_model,
        output=str(projected_path),
        rank=args.rank,
        alpha=args.alpha
    )
    run_project(project_args)

    # Step 3: Evaluate
    print(f"\n[*] Stage 3: Target Evaluation & Final Release Decision...")
    eval_output = Path(args.output_dir) / "target_eval_results.json"
    eval_args = SimpleArgs(
        target_model=args.target_model,
        adapter=str(projected_path),
        benchmark=args.benchmark,
        output=str(eval_output),
        report_path=str(report_path),
        positive_delta_threshold=args.positive_delta_threshold,
        max_regression_rate=args.max_regression_rate
    )
    run_evaluate(eval_args)

    print(f"\n" + "="*60)
    print(f" Safe-Project Complete.")
    print(f" Results: {args.output_dir}")
    print(f"="*60 + "\n")

def add_safe_project_parser(subparsers):
    parser = subparsers.add_parser(
        "safe-project",
        help="Run the complete end-to-end pipeline (Diagnose -> Project -> Evaluate)."
    )

    # Common Args
    parser.add_argument("--source-base", dest="source_base_model", required=True,
                        help="Source base model (e.g., Qwen2.5-Coder-7B)")
    parser.add_argument("--source-adapter", required=True,
                        help="Path or ID of the source adapter")
    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model (e.g., Qwen2.5-Coder-0.5B)")
    parser.add_argument("--benchmark", default="sql_50",
                        help="Benchmark to use (default: sql_50)")
    parser.add_argument("--output-dir", default="runs/safe_project",
                        help="Directory to save all results")
    
    # Projection Args
    parser.add_argument("--rank", type=int, default=16, help="Target rank")
    parser.add_argument("--alpha", type=int, default=16, help="Target alpha")
    
    # Eval Thresholds
    parser.add_argument("--positive-delta-threshold", type=float, default=0.0)
    parser.add_argument("--max-regression-rate", type=float, default=0.05)
    
    # Safety
    parser.add_argument("--force", action="store_true", help="Force projection even if diagnostic fails")

    parser.set_defaults(func=run_safe_project)
