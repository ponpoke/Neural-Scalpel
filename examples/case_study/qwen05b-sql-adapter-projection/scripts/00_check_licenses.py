import os
import json
import argparse
from pathlib import Path

def check_licenses(source_model_id, target_model_id, adapter_id, real_check=False):
    """
    Checks or simulates a license check for the models and adapter.
    """
    mode = "REAL" if real_check else "SIMULATED"
    print(f"[Phase 0] Checking licenses ({mode}) for:")
    print(f"  Source Model: {source_model_id}")
    print(f"  Target Model: {target_model_id}")
    print(f"  Adapter:      {adapter_id}")

    if real_check:
        # In a real scenario, this would use the Hugging Face API
        # Since this is a CLI script, we'll keep it simple but mark it as REAL if verified
        print("Warning: Real license check requires manual verification of HF model cards.")
        report = {
            "status": "MANUAL_VERIFICATION_REQUIRED",
            "timestamp": "2026-05-06T20:45:00Z",
            "mode": "REAL",
            "conclusion": "Manual verification of licenses is required before redistribution."
        }
    else:
        report = {
            "status": "SIMULATED",
            "timestamp": "2026-05-06T20:45:00Z",
            "mode": "SIMULATED",
            "models": {
                source_model_id: {"license": "Apache-2.0", "redistributable": True},
                target_model_id: {"license": "Apache-2.0", "redistributable": True},
            },
            "adapter": {
                adapter_id: {"license": "Apache-2.0", "redistributable": True}
            },
            "conclusion": "This is a SIMULATED license check. Do not publish weights based on this report."
        }

    report_path = Path("reports/license_check.md")
    os.makedirs(report_path.parent, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# License Check Report ({mode})\n\n")
        if not real_check:
            f.write(f"> [!CAUTION]\n")
            f.write(f"> **SIMULATED REPORT**: Do not use for legal compliance.\n\n")
        f.write(f"**Status:** {report['status']}\n\n")
        f.write(f"## Summary\n{report['conclusion']}\n\n")
        f.write(f"## Details\n")
        f.write(f"| Artifact | License | Redistributable |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| {source_model_id} | {'Apache-2.0' if not real_check else 'TBD'} | {'✅' if not real_check else '❓'} |\n")
        f.write(f"| {target_model_id} | {'Apache-2.0' if not real_check else 'TBD'} | {'✅' if not real_check else '❓'} |\n")
        f.write(f"| {adapter_id} | {'Apache-2.0' if not real_check else 'TBD'} | {'✅' if not real_check else '❓'} |\n")

    print(f"Report saved to {report_path}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Perform manual-verification-ready check")
    parser.add_argument("--source", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    args = parser.parse_args()

    check_licenses(args.source, args.target, args.adapter, real_check=args.real)
