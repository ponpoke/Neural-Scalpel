import shutil
import json
from pathlib import Path

def generate_citation_cff(report: dict) -> str:
    source_adapter = report.get("source_adapter", "unknown")
    target_model = report.get("target_model", "unknown")
    
    cff = []
    cff.append("cff-version: 1.2.0")
    cff.append(f"message: \"If you use this projected adapter, please cite it as follows.\"")
    cff.append(f"title: \"Projected Adapter: {Path(source_adapter).name} for {target_model}\"")
    cff.append("authors:")
    cff.append("  - family-names: \"Neural-Scalpel-Autogen\"")
    cff.append(f"version: \"1.0.0\"")
    cff.append(f"date-released: \"2026-05-07\"")
    cff.append(f"url: \"https://github.com/ponpoke/Neural-Scalpel\"")
    return "\n".join(cff)

def run_package_release(args):
    print(f"[Package] Packaging release from {args.run_dir}...")
    
    run_dir = Path(args.run_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy Core Adapter Weights (Strict Selection)
    print(" - Copying adapter weights and config...")
    for name in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]:
        src = adapter_dir / name
        if src.exists():
            shutil.copy2(src, output_dir)
            
    # 2. Copy Scientific Evidence (Reports)
    print(" - Gathering scientific evidence...")
    reports_to_copy = [
        "diagnostic_report.json",
        "target_eval_results.json",
        "final_analysis.md"
    ]
    for r in reports_to_copy:
        src = run_dir / r
        if src.exists():
            shutil.copy2(src, output_dir)
            
    # 3. Handle README.md (Model Card)
    readme_src = run_dir / "README.md"
    if not readme_src.exists():
        readme_src = adapter_dir / "README.md"
        
    if readme_src.exists():
        shutil.copy2(readme_src, output_dir / "README.md")
        
    # 4. Read report for metadata & citation
    print(" - Generating metadata and citation...")
    report_json = run_dir / "diagnostic_report.json"
    report_data = {}
    if report_json.exists():
        with open(report_json, "r") as f:
            report_data = json.load(f)
            
    # Generate CITATION.cff
    if report_data:
        cff_content = generate_citation_cff(report_data)
        with open(output_dir / "CITATION.cff", "w") as f:
            f.write(cff_content)
            
    # 5. Metadata for Zenodo/Publishing (Enhanced)
    decision = report_data.get("release_decision_gate", {})
    pub_meta = {
        "framework": "Neural-Scalpel v2.4.0",
        "distribution_type": "Projected-PEFT-Adapter",
        "source_adapter": report_data.get("source_adapter"),
        "target_model": report_data.get("target_model"),
        "diagnostic_verdict": decision.get("verdict"),
        "recommendation": decision.get("recommendation"),
        "created_by": "neural-scalpel package-release",
        "contents": [f.name for f in output_dir.iterdir()]
    }
    with open(output_dir / "projection_metadata.json", "w") as f:
        json.dump(pub_meta, f, indent=2)
        
    print(f"\n[Package] SUCCESS: Release package created at {output_dir}")
    print(f"Items included: {len(list(output_dir.iterdir()))} files.")

def add_package_release_parser(subparsers):
    parser = subparsers.add_parser(
        "package-release",
        help="Package all artifacts (weights, reports, model card) into a single distribution folder."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing run reports")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing projected weights")
    parser.add_argument("--output-dir", required=True, help="Folder to create the release package")
    parser.set_defaults(func=run_package_release)
