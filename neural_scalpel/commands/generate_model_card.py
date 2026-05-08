import json
from pathlib import Path
from neural_scalpel.core.diagnostic import AdapterTransferDiagnosticReport

def generate_model_card(report: AdapterTransferDiagnosticReport, eval_data: dict = None) -> str:
    te = eval_data.get("target_evaluation", {}) if eval_data else {}
    adapter_acc = te.get("adapter_metrics", {}).get("execution_accuracy", 0) * 100
    delta = te.get("delta", {}).get("execution_accuracy", 0) * 100
    
    # License fallback
    license_str = report.metadata_gate.license
    if not license_str or license_str.upper() == "UNKNOWN":
        license_str = "other"
    
    # HF Link Logic
    source_name = report.source_adapter
    if "/" in source_name and not Path(source_name).exists():
        source_link = f"[{source_name}](https://huggingface.co/{source_name})"
    else:
        source_link = f"`{source_name}`"
        
    md = []
    md.append(f"---")
    md.append(f"license: {license_str.lower()}")
    md.append(f"base_model: {report.target_model}")
    md.append(f"library_name: peft")
    md.append(f"pipeline_tag: text-generation")
    md.append(f"tags:")
    md.append(f"- neural-scalpel")
    md.append(f"- adapter-transfer")
    md.append(f"- lora")
    md.append(f"---")
    
    md.append(f"\n# {Path(report.source_adapter).name}-projected")
    md.append(f"This is a projected LoRA adapter transplanted from `{report.source_base_model}` to `{report.target_model}` using **Neural-Scalpel**.")
    
    md.append(f"\n## Model Details")
    md.append(f"- **Transfer Type:** Structural Projection")
    md.append(f"- **Source Adapter:** {source_link}")
    md.append(f"- **Target Architecture:** {report.target_model}")
    md.append(f"- **Diagnostic Verdict:** `{report.release_decision_gate.verdict}`")
    
    md.append(f"\n## Performance (Target Benchmarks)")
    md.append(f"Evaluated using Neural-Scalpel v2.3 Target Evaluation Gate.")
    md.append(f"- **Accuracy:** {adapter_acc:.2f}%")
    md.append(f"- **Delta vs Base:** {delta:+.2f}%")
    
    md.append(f"\n## Usage")
    md.append(f"You can load this adapter using standard `peft` or the Neural-Scalpel CLI.")
    
    md.append(f"```python")
    md.append(f"from peft import PeftModel")
    md.append(f"from transformers import AutoModelForCausalLM")
    md.append(f"\nbase_model = AutoModelForCausalLM.from_pretrained(\"{report.target_model}\")")
    md.append(f"model = PeftModel.from_pretrained(base_model, \"your-username/{Path(report.source_adapter).name}-projected\")")
    md.append(f"```")
    
    md.append(f"\n## Evaluation Methodology")
    md.append(f"This adapter was validated using the Neural-Scalpel multi-stage diagnostic pipeline. For full details, see the accompanying `diagnostic_report.json`.")
    
    md.append(f"\n---")
    md.append(f"*Developed with [Neural-Scalpel](https://github.com/ponpoke/Neural-Scalpel)*")
    
    return "\n".join(md)

def run_generate_model_card(args):
    print(f"[ModelCard] Loading results from {args.run_dir}...")
    report_path = Path(args.run_dir) / "diagnostic_report.json"
    eval_path = Path(args.run_dir) / "target_eval_results.json"
    
    if not report_path.exists():
        raise FileNotFoundError(f"Diagnostic report not found at {report_path}")
        
    report = AdapterTransferDiagnosticReport.from_json(str(report_path))
    
    eval_data = None
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
            
    card_content = generate_model_card(report, eval_data)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_content)
        
    print(f"[ModelCard] Model card generated: {output_path}")

def add_generate_model_card_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-model-card",
        help="Generate a Hugging Face Model Card (README.md) for the projected adapter."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing run results")
    parser.add_argument("--output", default="README.md", help="Output path for the model card")
    parser.set_defaults(func=run_generate_model_card)
