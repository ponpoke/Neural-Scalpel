import os
import json
import argparse
from pathlib import Path
import torch

def check_integrity(payload_path, manifest_path, real_check=False):
    mode = "REAL" if real_check else "SIMULATED"
    print(f"[Phase 3] Checking payload integrity ({mode}): {payload_path}")
    
    if real_check:
        if not os.path.exists(payload_path):
            print(f"Error: Payload file not found: {payload_path}")
            return None
            
        try:
            from safetensors.torch import load_file
            tensors = load_file(payload_path, device="cpu")
            
            # Basic checks
            is_finite = True
            for k, v in tensors.items():
                if not torch.isfinite(v).all():
                    is_finite = False
                    break
            
            results = {
                "payload_path": payload_path,
                "mode": "REAL",
                "num_tensors": len(tensors),
                "is_finite": is_finite,
                "status": "PASS" if is_finite else "FAIL"
            }
        except Exception as e:
            print(f"Error during integrity check: {e}")
            return None
    else:
        results = {
            "payload_path": payload_path,
            "mode": "SIMULATED",
            "no_nan": True,
            "no_inf": True,
            "shape_check_passed": True,
            "manifest_check_passed": True,
            "status": "SCAFFOLD",
            "note": "This is a SIMULATED integrity check. No real files were inspected."
        }
    
    report_path = Path("reports/payload_integrity.json")
    os.makedirs(report_path.parent, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Integrity check results saved to {report_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Perform real payload integrity check")
    parser.add_argument("--payload", default="routes/qwen05b_sql_projection/qwen05b_sql_payload.safetensors")
    parser.add_argument("--manifest", default="routes/qwen05b_sql_projection/qwen05b_sql.scalpel_route")
    args = parser.parse_args()

    check_integrity(args.payload, args.manifest, real_check=args.real)
