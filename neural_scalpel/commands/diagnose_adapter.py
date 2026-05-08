from neural_scalpel.core.diagnostic_runner import DiagnosticRunner

def run_diagnose(args):
    report = DiagnosticRunner.execute(args)
    
    print(f"\n--- Diagnostic Result: {report.release_decision_gate.verdict} ---")
    print(f"Recommendation: {report.release_decision_gate.recommendation}")
    for reason in report.release_decision_gate.reasons:
        print(f" - {reason}")
    print(f"\n[Diag] Complete. Report saved to {args.output_dir}/diagnostic_report.json")

def add_diagnose_adapter_parser(subparsers):
    parser = subparsers.add_parser(
        "diagnose-adapter",
        help="Run multi-stage adapter transfer diagnostics."
    )

    parser.add_argument("--source-base", dest="source_base_model", required=True, 
                        help="Path or ID of the source base model")
    parser.add_argument("--source-adapter", required=True, 
                        help="Path or ID of the source adapter (local or HF Hub)")
    parser.add_argument("--target", dest="target_model", 
                        help="Path or ID of the target base model for compatibility check")
    parser.add_argument("--benchmark", default="sql_50", 
                        help="Benchmark to use (e.g., sql_50)")
    parser.add_argument("--output-dir", default="reports/diagnostics", 
                        help="Directory to save the diagnostic report")
    parser.add_argument("--force", action="store_true", 
                        help="Force continue if non-fatal gates fail")

    parser.set_defaults(func=run_diagnose)
