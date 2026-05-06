import os
import sys
import argparse
import subprocess
from pathlib import Path

def prepare_payload(lora_id, target_model, output_dir, real_run=False, export_peft=False, scale_gamma=1.0):
    mode = "REAL" if real_run else "SIMULATED"
    print(f"[Phase 2] Preparing projected payload ({mode}) for {lora_id} -> {target_model}")
    
    if real_run:
        cmd = [
            sys.executable,
            "../../../../scripts/prepare_actual_lora_payload.py",
            "--lora_id", lora_id,
            "--target-model", target_model,
            "--output_dir", output_dir,
            "--scale-gamma", str(scale_gamma)
        ]
        if export_peft:
            cmd.append("--export-peft")
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("Payload generation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during payload generation: {e}")
            sys.exit(1)
    else:
        print(">>> SIMULATION MODE: No real payload will be generated.")
        print(">>> To generate a real payload, use the --real flag.")
        os.makedirs(output_dir, exist_ok=True)
        # We still create dummy files for the pipeline to continue testing the reporting flow,
        # but we mark them as dummy in the report.
        (Path(output_dir) / "qwen05b_sql_payload.safetensors.simulated").touch()
        (Path(output_dir) / "qwen05b_sql.scalpel_route.simulated").touch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Execute real payload projection")
    parser.add_argument("--lora_id", default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection")
    parser.add_argument("--export-peft", action="store_true", help="Export as PEFT-compatible LoRA adapter")
    parser.add_argument("--scale-gamma", type=float, default=1.0, help="Scale factor for the projected delta")
    args = parser.parse_args()

    prepare_payload(args.lora_id, args.target_model, args.output_dir, real_run=args.real, export_peft=args.export_peft, scale_gamma=args.scale_gamma)
