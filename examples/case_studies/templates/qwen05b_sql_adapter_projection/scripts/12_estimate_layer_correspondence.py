import torch
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def gram_linear_cka(X, Y):
    """
    Compute Linear CKA using Gram matrices (efficient for large hidden dimensions).
    X: (n, d1)
    Y: (n, d2)
    Complexity: O(n^2 * d) for Gram, then O(n^3) or O(n^2) for CKA.
    Much faster than feature-matmul when d >> n (e.g., 4096 >> 50).
    """
    # 1. Center the features (n, d)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # 2. Compute Gram matrices (n, n)
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())
    
    # 3. Center the Gram matrices (centering matrix H = I - 1/n)
    def center_gram(G):
        n = G.shape[0]
        row_mean = G.mean(dim=1, keepdim=True)
        col_mean = G.mean(dim=0, keepdim=True)
        grand_mean = G.mean()
        return G - row_mean - col_mean + grand_mean
    
    Kc = center_gram(K)
    Lc = center_gram(L)
    
    # 4. HSIC and CKA
    # HSIC(K, L) = tr(Kc * Lc) / (n-1)^2
    # CKA(K, L) = tr(Kc * Lc) / (||Kc||_F * ||Lc||_F)
    hsic_kl = (Kc * Lc).sum()
    norm_k = torch.norm(Kc, p='fro')
    norm_l = torch.norm(Lc, p='fro')
    
    if norm_k < 1e-12 or norm_l < 1e-12:
        return 0.0
        
    cka_score = hsic_kl / (norm_k * norm_l + 1e-9)
    return cka_score.item()

def estimate_layer_correspondence(input_path, output_dir):
    print(f"Loading paired activations from {input_path}...")
    data = torch.load(input_path)
    
    metadata = data["metadata"]
    streams = data["streams"]
    
    source_base = streams["source_base"]
    target_base = streams["target_base"]
    
    source_layer_names = sorted(source_base.keys(), key=lambda x: int(x.split(".")[-1]))
    target_layer_names = sorted(target_base.keys(), key=lambda x: int(x.split(".")[-1]))
    
    n_samples = metadata["num_samples"]
    print(f"Calculating similarity matrix ({len(source_layer_names)}x{len(target_layer_names)}) using Gram-CKA (n={n_samples})...")
    
    correspondence = []
    
    for t_name in tqdm(target_layer_names, desc="Target Layers"):
        t_act = target_base[t_name]
        layer_scores = []
        
        for s_name in source_layer_names:
            s_act = source_base[s_name]
            score = gram_linear_cka(s_act, t_act)
            layer_scores.append({"source_layer": s_name, "cka": score})
            
        top_matches = sorted(layer_scores, key=lambda x: x["cka"], reverse=True)
        
        correspondence.append({
            "target_layer": t_name,
            "top_matches": top_matches[:5],
            "best_source_layer": top_matches[0]["source_layer"],
            "best_cka": top_matches[0]["cka"]
        })

    mean_top_cka = sum(c["best_cka"] for c in correspondence) / len(correspondence)
    
    verdict = "LAYER_CORRESPONDENCE_ESTIMATED"
    if mean_top_cka < 0.2:
        verdict = "LOW_CONFIDENCE_CORRESPONDENCE"
        
    summary = {
        "evaluation_type": "layer_correspondence_estimation",
        "method": "gram_linear_cka",
        "num_samples": n_samples,
        "mean_top1_cka": mean_top_cka,
        "correspondence_status": verdict,
        "correspondence": correspondence,
        "warnings": [
            "CKA estimates are heuristic and based on last-token hidden states only.",
            "Small sample sizes may lead to unstable mappings."
        ],
        "does_not_validate": ["alignment map quality", "target transfer success"]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(Path(output_dir) / "layer_correspondence_report.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(Path(output_dir) / "layer_correspondence_report.md", "w") as f:
        f.write("# Layer Correspondence Analysis (Phase 5-C)\n\n")
        f.write(f"**Status**: `{verdict}` | **Mean Top-1 CKA**: {mean_top_cka:.4f}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> This mapping is a preliminary heuristic estimate based on representation similarity. Success of behavioral transfer depends on the next phase (Map Learning).\n\n")
        
        f.write("## Best Correspondence Map\n")
        f.write("| Target Layer | Best Source Match | CKA Score |\n| :--- | :--- | :---: |\n")
        for c in correspondence:
            f.write(f"| {c['target_layer']} | {c['best_source_layer']} | {c['best_cka']:.4f} |\n")

    print(f"Estimation complete. Mean CKA: {mean_top_cka:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/alignment")
    args = parser.parse_args()
    estimate_layer_correspondence(args.input, args.output_dir)
