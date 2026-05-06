import torch
import json
import argparse
import os
from pathlib import Path

def transport_behavioral_delta(delta_path, maps_path, activations_path, output_dir):
    print(f"Loading Source Deltas, Alignment Maps, and Target Activations...")
    source_data = torch.load(delta_path) # contains metadata and deltas
    maps = torch.load(maps_path) # contains {target_layer: {source_layer, weight, ...}}
    paired_data = torch.load(activations_path)
    
    target_base = paired_data["streams"]["target_base"]
    source_deltas = source_data["deltas"]
    
    transported_deltas = {}
    layer_reports = []
    global_rel_delta_sum = 0
    
    print(f"Transporting behavioral deltas to target manifold...")
    
    for t_name, map_info in maps.items():
        s_name = map_info["source_layer"]
        P = map_info["weight"] # (d_source, d_target)
        
        if s_name not in source_deltas:
            print(f"Warning: Source layer {s_name} delta not found. Skipping {t_name}.")
            continue
            
        ds = source_deltas[s_name] # (n, d_source)
        dt_desired = torch.matmul(ds, P) # (n, d_target)
        
        transported_deltas[t_name] = dt_desired
        
        # Metrics relative to Target Base
        ht_base = target_base[t_name]
        dt_norm = torch.norm(dt_desired, p=2, dim=-1).mean().item()
        ht_norm = torch.norm(ht_base, p=2, dim=-1).mean().item()
        rel_delta = dt_norm / (ht_norm + 1e-9)
        
        layer_reports.append({
            "target_layer": t_name,
            "source_layer_used": s_name,
            "delta_l2_mean": dt_norm,
            "relative_delta": rel_delta,
            "map_heldout_error": map_info["heldout_error"]
        })
        global_rel_delta_sum += rel_delta

    # Summary Logic
    n_layers = len(layer_reports)
    mean_rel_delta = global_rel_delta_sum / n_layers if n_layers > 0 else 0
    
    verdict = "TARGET_DELTA_TRANSPORTED"
    if mean_rel_delta < 1e-5:
        verdict = "WEAK_TRANSPORTED_SIGNAL"
    elif mean_rel_delta > 0.5:
        verdict = "OVERAMPLIFIED_TRANSPORTED_SIGNAL"

    summary = {
        "evaluation_type": "behavioral_delta_transport",
        "num_layers": n_layers,
        "mean_relative_delta": mean_rel_delta,
        "transport_status": verdict,
        "layer_reports": layer_reports,
        "warnings": [
            "This produces a desired target activation delta, not yet a deployable adapter.",
            "Relative delta indicates the signal strength compared to target base activations."
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"metadata": source_data["metadata"], "transported_deltas": transported_deltas}, Path(output_dir) / "target_behavior_delta_desired.pt")
    
    with open(Path(output_dir) / "delta_transport_report.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(Path(output_dir) / "delta_transport_report.md", "w") as f:
        f.write("# Behavioral Delta Transport Report (Phase 5-E)\n\n")
        f.write(f"**Status**: `{verdict}` | **Mean Relative Delta**: {mean_rel_delta:.6f}\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> Phase 5-E transports source-side behavioral activation deltas into the target representation space. This produces a **desired target activation delta**, which serves as the teacher signal for the final adapter solve (Phase 5-F).\n\n")
        
        f.write("## Layer-wise Transport Summary\n")
        f.write("| Target Layer | Source Layer | Delta L2 | Rel. Delta | Map Error |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for r in layer_reports:
            f.write(f"| {r['target_layer']} | {r['source_layer_used']} | {r['delta_l2_mean']:.6f} | {r['relative_delta']:.6f} | {r['map_heldout_error']:.4f} |\n")

    print(f"Transport complete. Verdict: {verdict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", default="routes/qwen05b_sql_projection/analysis/source_behavior_delta.pt")
    parser.add_argument("--maps", default="routes/qwen05b_sql_projection/alignment_maps/alignment_maps.pt")
    parser.add_argument("--activations", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/transported_delta")
    args = parser.parse_args()
    
    transport_behavioral_delta(args.delta, args.maps, args.activations, args.output_dir)
