import torch
import json
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def solve_ridge(X, Y, alpha=1.0):
    """
    Solve Ridge Regression: min ||X P - Y||^2 + alpha ||P||^2
    X: (n, d_in), Y: (n, d_out)
    Solution: P = (X^T X + alpha I)^-1 X^T Y
    """
    d_in = X.shape[1]
    # Use float64 for better numerical stability during matrix inversion
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    
    A = torch.matmul(X.t(), X) + alpha * torch.eye(d_in, device=X.device, dtype=torch.float64)
    B = torch.matmul(X.t(), Y)
    
    try:
        P = torch.linalg.solve(A, B)
    except torch.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        P = torch.matmul(torch.linalg.pinv(A), B)
        
    return P.to(torch.float32)

def calculate_reconstruction_error(X, Y, P):
    """Calculate Relative Frobenius Norm Error."""
    Y_pred = torch.matmul(X, P)
    error = torch.norm(Y - Y_pred, p='fro') / (torch.norm(Y, p='fro') + 1e-9)
    return error.item()

def learn_alignment_maps(activations_path, correspondence_path, output_dir, alpha=1.0):
    print(f"Loading data...")
    data = torch.load(activations_path)
    with open(correspondence_path, "r") as f:
        corr_data = json.load(f)
    
    streams = data["streams"]
    source_base = streams["source_base"]
    target_base = streams["target_base"]
    correspondence = corr_data["correspondence"]
    
    n_samples = data["metadata"]["num_samples"]
    if n_samples < 5:
        raise ValueError("At least 5 paired samples are required for train/heldout split.")

    indices = list(range(n_samples))
    train_idx, heldout_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print(f"Learning alignment maps (Train: {len(train_idx)}, Heldout: {len(heldout_idx)})...")
    
    alignment_payload = {}
    layer_reports = []
    
    for entry in correspondence:
        t_name = entry["target_layer"]
        s_name = entry["best_source_layer"]
        
        # Prepare Tensors
        X_all = source_base[s_name]
        Y_all = target_base[t_name]
        
        X_train, X_heldout = X_all[train_idx], X_all[heldout_idx]
        Y_train, Y_heldout = Y_all[train_idx], Y_all[heldout_idx]
        
        # Solve Ridge
        P = solve_ridge(X_train, Y_train, alpha=alpha)
        
        # Evaluate
        train_err = calculate_reconstruction_error(X_train, Y_train, P)
        heldout_err = calculate_reconstruction_error(X_heldout, Y_heldout, P)
        
        alignment_payload[t_name] = {
            "source_layer": s_name,
            "weight": P,
            "alpha": alpha,
            "train_error": train_err,
            "heldout_error": heldout_err
        }
        
        layer_reports.append({
            "target_layer": t_name,
            "source_layer": s_name,
            "train_error": train_err,
            "heldout_error": heldout_err,
            "status": "STABLE" if heldout_err < 0.5 else "LOW_CONFIDENCE"
        })

    # Summary
    mean_heldout_err = sum(r["heldout_error"] for r in layer_reports) / len(layer_reports)
    verdict = "ALIGNMENT_MAPS_LEARNED"
    if mean_heldout_err > 0.5:
        verdict = "LOW_CONFIDENCE_MAPS"
        
    summary = {
        "evaluation_type": "alignment_map_learning",
        "method": "ridge_regression",
        "alpha": alpha,
        "split_random_state": 42,
        "train_fraction": 0.8,
        "heldout_fraction": 0.2,
        "num_train": len(train_idx),
        "num_heldout": len(heldout_idx),
        "mean_heldout_error": mean_heldout_err,
        "alignment_status": verdict,
        "layer_reports": layer_reports,
        "warnings": [
            "Errors are relative Frobenius norms. < 0.5 is considered stable.",
            "This map aligns source_base to target_base; it does not yet transport deltas."
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(alignment_payload, Path(output_dir) / "alignment_maps.pt")
    
    with open(Path(output_dir) / "alignment_map_report.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(Path(output_dir) / "alignment_map_report.md", "w") as f:
        f.write("# Alignment Map Learning Report (Phase 5-D)\n\n")
        f.write(f"**Status**: `{verdict}` | **Mean Heldout Error**: {mean_heldout_err:.4f}\n\n")
        
        f.write("## Layer-wise Alignment Performance\n")
        f.write("| Target Layer | Source Layer | Train Error | Heldout Error | Status |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for r in layer_reports:
            f.write(f"| {r['target_layer']} | {r['source_layer']} | {r['train_error']:.4f} | {r['heldout_error']:.4f} | `{r['status']}` |\n")

    print(f"Learning complete. Mean Heldout Error: {mean_heldout_err:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--correspondence", default="routes/qwen05b_sql_projection/alignment/layer_correspondence_report.json")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/alignment_maps")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    learn_alignment_maps(args.activations, args.correspondence, args.output_dir, alpha=args.alpha)
