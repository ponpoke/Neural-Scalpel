import torch
import json
import argparse
import os
from pathlib import Path

def solve_ridge(X, Y, alpha=1.0):
    d_in = X.shape[1]
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    A = torch.matmul(X.t(), X) + alpha * torch.eye(d_in, device=X.device, dtype=torch.float64)
    B = torch.matmul(X.t(), Y)
    try:
        W = torch.linalg.solve(A, B)
    except:
        W = torch.matmul(torch.linalg.pinv(A), B)
    return W.to(torch.float32)

def low_rank_decompose_for_peft(W, rank):
    """
    W: (d_in, d_out), used as Y approx X @ W
    Goal: X @ A.T @ B.T approx X @ W
    Return:
      A: (rank, d_in)
      B: (d_out, rank)
    """
    # Clamp rank
    rank = min(rank, W.shape[0], W.shape[1])
    
    U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    
    U_r = U[:, :rank]          # (d_in, r)
    S_r = S[:rank]             # (r,)
    Vh_r = Vh[:rank, :]        # (r, d_out)

    sqrtS = torch.sqrt(S_r)

    # Need W approx A.T @ B.T
    # Let A.T = U_r @ diag(sqrtS) -> A = diag(sqrtS) @ U_r.T (rank, d_in)
    # Let B.T = diag(sqrtS) @ Vh_r -> B = Vh_r.T @ diag(sqrtS) (d_out, rank)
    A = (sqrtS[:, None] * U_r.T).to(torch.float32)
    B = (Vh_r.T * sqrtS[None, :]).to(torch.float32)
    
    return A, B

def solve_peft_lora(module_activations_path, output_dir, rank=16, alpha=1.0, lora_alpha=None):
    if lora_alpha is None:
        lora_alpha = rank * 2 # Default to 2.0 scaling for consistency
        
    print(f"Loading module activations and desired deltas...")
    data = torch.load(module_activations_path)
    module_inputs = data["module_inputs"]
    desired_deltas = data["desired_deltas"]
    
    lora_state_dict = {}
    layer_reports = []
    
    print(f"Solving PEFT-style LoRA (Rank={rank}, Alpha={alpha}, LoRA-Alpha={lora_alpha})...")
    
    for layer_key, delta in desired_deltas.items():
        idx = int(layer_key.split(".")[-1])
        module_name = f"model.layers.{idx}.mlp.down_proj"
        
        if module_name not in module_inputs:
            continue
            
        X = module_inputs[module_name] # (n, d_in)
        Y = delta # (n, d_out)
        
        # Solve Full Delta W (d_in, d_out)
        W_full = solve_ridge(X, Y, alpha=alpha)
        
        # Decompose for PEFT
        A, B = low_rank_decompose_for_peft(W_full, rank)
        
        # Key naming: PEFT typically maps 'lora_A.weight' to the active adapter internally
        lora_state_dict[f"base_model.model.model.layers.{idx}.mlp.down_proj.lora_A.weight"] = A
        lora_state_dict[f"base_model.model.model.layers.{idx}.mlp.down_proj.lora_B.weight"] = B
        
        # Eval
        W_approx = torch.matmul(A.t(), B.t())
        Y_pred = torch.matmul(X, W_approx)
        error = torch.norm(Y - Y_pred) / (torch.norm(Y) + 1e-9)
        
        layer_reports.append({
            "layer": idx,
            "module": "mlp.down_proj",
            "input_dim": X.shape[1],
            "output_dim": Y.shape[1],
            "rank": A.shape[0],
            "lora_A_shape": list(A.shape),
            "lora_B_shape": list(B.shape),
            "reconstruction_error": error.item()
        })

    if not layer_reports:
        raise RuntimeError("No LoRA layers were solved. Check module inputs and desired deltas.")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(lora_state_dict, Path(output_dir) / "adapter_model.bin")
    
    config = {
        "base_model_name_or_path": data["metadata"]["model_id"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": ["down_proj"],
        "modules_to_save": None,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "peft_config_version": "0.1",
        "adapter_name": "default"
    }
    with open(Path(output_dir) / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    mean_err = sum(r["reconstruction_error"] for r in layer_reports) / len(layer_reports)
    summary = {
        "status": "MODULE_LORA_SOLVED",
        "rank": rank,
        "lora_alpha": lora_alpha,
        "mean_reconstruction_error": mean_err,
        "layer_reports": layer_reports
    }
    with open(Path(output_dir) / "lora_solve_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] PEFT LoRA solved (Alpha={lora_alpha}). Mean Error: {mean_err:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="routes/qwen05b_sql_projection/module_activations/module_inputs.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/peft_lora")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args()
    
    solve_peft_lora(args.inputs, args.output_dir, rank=args.rank, alpha=args.alpha, lora_alpha=args.lora_alpha)
