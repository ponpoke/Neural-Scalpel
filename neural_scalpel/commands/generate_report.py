import json
import os
from pathlib import Path
from neural_scalpel.core.diagnostic import AdapterTransferDiagnosticReport

def generate_markdown_report(report: AdapterTransferDiagnosticReport, eval_data: dict = None) -> str:
    md = []
    md.append(f"# Neural-Scalpel Diagnostic & Evaluation Report")
    md.append(f"**Run ID:** `{report.run_id}`  ")
    md.append(f"**Timestamp:** {report.timestamp}  ")
    md.append(f"**Final Verdict:** `{report.release_decision_gate.verdict}`")
    
    md.append(f"\n## 1. Executive Summary")
    md.append(f"{report.release_decision_gate.recommendation}")
    for reason in report.release_decision_gate.reasons:
        md.append(f"- {reason}")
        
    md.append(f"\n## 2. Transfer Configuration")
    md.append(f"| Component | Entity |")
    md.append(f"|---|---|")
    md.append(f"| **Source Base** | `{report.source_base_model}` |")
    md.append(f"| **Source Adapter** | `{report.source_adapter}` |")
    md.append(f"| **Target Model** | `{report.target_model}` |")
    
    md.append(f"\n## 3. Diagnostic Gates (Stage 1-4)")
    md.append(f"### Stage 1: Metadata Gate")
    md.append(f"- **Status:** `{report.metadata_gate.status}`")
    md.append(f"- **Adapter Type:** {report.metadata_gate.adapter_type}")
    md.append(f"- **License:** {report.metadata_gate.license}")
    
    md.append(f"\n### Stage 2: Source Quality Gate")
    sq = report.source_quality_gate
    md.append(f"- **Verdict:** `{sq.get('verdict', 'N/A')}`")
    md.append(f"- **Teacher Improvement:** {sq.get('accuracy_delta', 0)*100:+.2f}%")
    
    md.append(f"\n### Stage 3: Delta Health Gate")
    dh = report.delta_health_gate
    md.append(f"- **Verdict:** `{dh.verdict}`")
    md.append(f"- **Global Frobenius Norm:** {dh.global_frobenius_norm:.4f}")
    
    md.append(f"\n### Stage 4: Compatibility & Feasibility")
    comp = report.compatibility_gate
    feas = report.feasibility_gate
    md.append(f"- **Compatibility Score:** {comp.compatibility_score*100:.1f}%")
    md.append(f"- **Layer Mapping:** `{feas.layer_mapping_type}`")
    md.append(f"- **GQA Aware Required:** {feas.gqa_aware_required}")
    
    if eval_data:
        md.append(f"\n## 4. Target Evaluation Results (Stage 5)")
        te = eval_data.get("target_evaluation", {})
        md.append(f"### Performance Delta")
        md.append(f"| Metric | Base | Adapter | Delta |")
        md.append(f"|---|---|---|---|")
        
        base_metrics = te.get("base_metrics", {})
        adapter_metrics = te.get("adapter_metrics", {})
        delta = te.get("delta", {})
        
        for m in base_metrics:
            md.append(f"| {m} | {base_metrics[m]*100:.2f}% | {adapter_metrics[m]*100:.2f}% | {delta.get(m, 0)*100:+.2f}% |")
            
        md.append(f"\n### Behavioral Classification")
        fc = te.get("failure_classification", {})
        md.append(f"- **Fixed (Improved):** {fc.get('fixed', 0)}")
        md.append(f"- **Regressed (Interference):** {fc.get('regressed', 0)}")
        md.append(f"- **Both Succeeded:** {fc.get('both_succeeded', 0)}")
        md.append(f"- **Both Failed:** {fc.get('both_failed', 0)}")
        md.append(f"- **Regression Rate:** {te.get('regression_rate', 0)*100:.2f}%")

    md.append(f"\n## 5. Required Artifacts")
    for art in report.release_decision_gate.required_artifacts:
        md.append(f"- [ ] `{art}`")
        
    return "\n".join(md)

def run_generate_report(args):
    print(f"[Report] Loading results from {args.run_dir}...")
    report_path = Path(args.run_dir) / "diagnostic_report.json"
    eval_path = Path(args.run_dir) / "target_eval_results.json"
    
    if not report_path.exists():
        raise FileNotFoundError(f"Diagnostic report not found at {report_path}")
        
    report = AdapterTransferDiagnosticReport.from_json(str(report_path))
    
    eval_data = None
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
            
    md_content = generate_markdown_report(report, eval_data)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"[Report] Final report generated: {output_path}")

def add_generate_report_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-report",
        help="Generate a detailed markdown report from safe-project results."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing run results")
    parser.add_argument("--output", default="reports/final_report.md", help="Output path for the markdown report")
    parser.set_defaults(func=run_generate_report)
