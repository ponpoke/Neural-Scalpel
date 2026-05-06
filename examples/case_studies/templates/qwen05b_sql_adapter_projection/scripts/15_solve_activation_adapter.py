import torch
import json
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def solve_ridge(X, Y, alpha=1.0):
    """
    Solve Ridge Regression: min ||X W - Y||^2 + alpha ||W||^2
    X: target base activation, Y: desired delta
    """
    d_in = X.shape[1]
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    
    A = torch.matmul(X.t(), X) + alpha * torch.eye(d_in, device=X.device, dtype=torch.float64)
    B = torch.matmul(X.t(), Y)
    
    try:
        W = torch.linalg.solve(A, B)
    except torch.linalg.LinAlgError:
        W = torch.matmul(torch.linalg.pinv(A), B)
        
    return W.to(torch.float32)

def calculate_error(X, Y, W):
    Y_pred = torch.matmul(X, W)
    return torch.norm(Y - Y_pred, p='fro') / (torch.norm(Y, p='fro') + 1e-9)

def solve_activation_adapter(activations_path, desired_delta_path, output_dir, alpha=1.0):
    print(f"Loading paired activations and desired deltas...")
    data = torch.load(activations_path)
    delta_data = torch.load(desired_delta_path)
    
    target_base = data["streams"]["target_base"]
    transported_deltas = delta_data["transported_deltas"]
    
    n_samples = data["metadata"]["num_samples"]
    if n_samples < 5:
        raise ValueError("At least 5 paired samples are required for train/heldout split.")

    indices = list(range(n_samples))
    train_idx, heldout_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print(f"Solving Activation-space Adapter (Train: {len(train_idx)}, Heldout: {len(heldout_idx)})...")
    
    adapter_payload = {}
    layer_reports = []
    
    for t_name in sorted(target_base.keys(), key=lambda x: int(x.split(".")[-1])):
        if t_name not in transported_deltas:
            continue
            
        X_all = target_base[t_name]
        Y_all = transported_deltas[t_name]
        
        X_train, X_heldout = X_all[train_idx], X_all[heldout_idx]
        Y_train, Y_heldout = Y_all[train_idx], Y_all[heldout_idx]
        
        # Solve
        W = solve_ridge(X_train, Y_train, alpha=alpha)
        
        # Finite check for weights
        if not torch.isfinite(W).all():
            print(f"WARNING: NaN/Inf detected in solved weights for {t_name}")
            continue
            
        # Eval
        train_err = calculate_error(X_train, Y_train, W).item()
        heldout_err = calculate_error(X_heldout, Y_heldout, W).item()
        
        # Metrics
        y_pred = torch.matmul(X_all, W)
        rel_output_norm = (torch.norm(y_pred, p=2) / (torch.norm(X_all, p=2) + 1e-9)).item()
        
        adapter_payload[t_name] = {
            "weight": W,
            "train_error": train_err,
            "heldout_error": heldout_err,
            "rel_output_norm": rel_output_norm
        }
        
        layer_reports.append({
            "layer": t_name,
            "train_error": train_err,
            "heldout_error": heldout_err,
            "rel_output_norm": rel_output_norm,
            "status": "SOLVED" if heldout_err < 0.5 else "LOW_CONFIDENCE"
        })

    # Summary
    if not layer_reports:
        raise RuntimeError("No activation adapter layers were solved. Check transported_deltas keys or finite check failures.")

    mean_heldout_err = sum(r["heldout_error"] for r in layer_reports) / len(layer_reports)
    verdict = "ACTIVATION_ADAPTER_SOLVED"
    if mean_heldout_err > 0.5:
        verdict = "LOW_CONFIDENCE_ACTIVATION_ADAPTER"

    summary = {
        "evaluation_type": "activation_adapter_solve",
        "method": "ridge_regression",
        "alpha": alpha,
        "split_random_state": 42,
        "mean_heldout_error": mean_heldout_err,
        "solve_status": verdict,
        "layer_reports": layer_reports,
        "warnings": [
            "This is an activation-space adapter, not yet a deployable PEFT LoRA.",
            "Validates if desired delta is linearly recoverable from target hidden states."
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(adapter_payload, Path(output_dir) / "activation_adapter_weights.pt")
    
    with open(Path(output_dir) / "activation_adapter_solve_report.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(Path(output_dir) / "activation_adapter_solve_report.md", "w") as f:
        f.write("# Activation-space Adapter Solve Report (Phase 5-F-1)\n\n")
        f.write(f"**Status**: `{verdict}` | **Mean Heldout Error**: {mean_heldout_err:.6f}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Phase 5-F-1 validates whether the transported behavioral signal is linearly recoverable from the target model's internal hidden states. This is a critical prerequisite for full PEFT LoRA extraction.\n\n")
        
        f.write("## Layer-wise Solve Performance\n")
        f.write("| Layer | Train Error | Heldout Error | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for r in layer_reports:
            f.write(f"| {r['layer']} | {r['train_error']:.4f} | {r['heldout_error']:.4f} | `{r['status']}` |\n")

    print(f"Solve complete. Mean Heldout Error: {mean_heldout_err:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--delta", default="routes/qwen05b_sql_projection/transported_delta/target_behavior_delta_desired.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/activation_adapter")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    solve_activation_adapter(args.activations, args.delta, args.output_dir, alpha=args.alpha)
