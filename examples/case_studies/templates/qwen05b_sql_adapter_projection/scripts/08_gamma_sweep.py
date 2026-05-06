import os
import subprocess
import json
import argparse
import shutil
import pandas as pd
from pathlib import Path

def run_sweep(gammas, lora_id, target_model):
    results = []
    print(f"Starting Gamma Sweep for Signal Detection: {gammas}")
    
    sweep_root = Path("reports/gamma_sweep")
    if sweep_root.exists():
        shutil.rmtree(sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)
    
    # Prepare environment with cross-platform PYTHONPATH
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    pp_paths = [".", "../../../../"]
    if existing_pp:
        pp_paths.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(pp_paths)
    
    for g in gammas:
        gamma_tag = str(g).replace(".", "p")
        print(f"\n[SWEEP] Testing Gamma = {g} (Tag: {gamma_tag})")
        
        adapter_path = Path(f"routes/qwen05b_sql_projection/gamma_{gamma_tag}")
        gamma_report_dir = sweep_root / f"gamma_{gamma_tag}"
        gamma_report_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Project with current gamma
        # Note: scale-gamma=0.0 should effectively disable the adapter
        proj_cmd = [
            "python", "scripts/02_prepare_payload_calibrated.py",
            "--lora_id", lora_id,
            "--target_model", target_model,
            "--scale-gamma", str(g),
            "--output_dir", str(adapter_path)
        ]
        subprocess.run(proj_cmd, check=True, env=env)
        
        # 2. Evaluate (Inference)
        eval_cmd = [
            "python", "scripts/04_eval_before_after.py",
            "--adapter_path", str(adapter_path / "peft_adapter_calibrated")
        ]
        subprocess.run(eval_cmd, check=True, env=env)
        
        # 3. Calculate Metrics
        metrics_cmd = [
            "python", "scripts/05_real_metrics.py",
            "--results_json", "reports/real_eval_results.json"
        ]
        subprocess.run(metrics_cmd, check=True, env=env)
        
        # 4. Isolate Reports
        shutil.copy("reports/real_eval_results.json", gamma_report_dir / "real_eval_results.json")
        shutil.copy("reports/eval_summary_real.json", gamma_report_dir / "eval_summary_real.json")
        shutil.copy("reports/eval_summary_real.md", gamma_report_dir / "eval_summary_real.md")
        shutil.copy("reports/before_after_real.md", gamma_report_dir / "before_after_real.md")

        # 5. Collect Metrics for CSV
        with open(gamma_report_dir / "eval_summary_real.json", "r") as f:
            m = json.load(f)
            results.append({
                "gamma": g,
                "exact_same_rate_raw": m.get("exact_same_rate_raw"),
                "exact_same_rate_normalized": m.get("exact_same_rate_normalized"),
                "projected_sql_signal_rate": m.get("projected", {}).get("sql_signal_rate"),
                "projected_repetition_rate": m.get("projected", {}).get("repetition_rate"),
                "projected_empty_rate": m.get("projected", {}).get("empty_output_rate"),
                "projected_avg_length": m.get("projected", {}).get("avg_length"),
                "behavioral_status": m.get("behavioral_status"),
            })

    # Save sweep summary
    df = pd.DataFrame(results)
    df.to_csv(sweep_root / "sweep_summary.csv", index=False)
    print(f"\n[SUCCESS] Gamma sweep complete. Summary saved to {sweep_root}/sweep_summary.csv")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_id", default="onurerkan/qwen2.5-0.5b-alpaca-lora-demo")
    parser.add_argument("--target_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gammas", default="0.0,1.0,2.0,4.0,8.0,16.0,32.0")
    args = parser.parse_args()
    
    target_gammas = [float(x) for x in args.gammas.split(",")]
    run_sweep(target_gammas, args.lora_id, args.target_model)
