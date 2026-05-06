import torch
import json
import argparse
import os
from pathlib import Path

def extract_behavioral_delta(input_path, output_dir):
    print(f"Loading paired activations from {input_path}...")
    data = torch.load(input_path)
    
    metadata = data["metadata"]
    streams = data["streams"]
    
    source_base = streams["source_base"]
    source_lora = streams["source_lora"]
    captured_prompts = metadata.get("captured_prompts", [])
    
    num_samples = metadata["num_samples"]
    print(f"Analyzing behavioral delta for {num_samples} samples...")
    
    layer_metrics = []
    global_delta_sum = 0
    finite_check = "PASS"
    zero_delta_layers = []
    
    processed_deltas = {}
    
    # 1. Per-Layer Analysis
    for layer_name in sorted(source_base.keys(), key=lambda x: int(x.split(".")[-1])):
        h_base = source_base[layer_name] 
        h_lora = source_lora[layer_name]
        
        # Shape Check
        if h_base.shape != h_lora.shape:
            raise ValueError(f"Shape mismatch at {layer_name}: base={h_base.shape}, lora={h_lora.shape}")
            
        # NaN/Inf Check
        if not torch.isfinite(h_base).all() or not torch.isfinite(h_lora).all():
            finite_check = "FAIL"
            print(f"WARNING: NaN/Inf detected at {layer_name}")
        
        # Calculate Delta
        delta = h_lora - h_base
        processed_deltas[layer_name] = delta
        
        # Metrics
        delta_norm = torch.norm(delta, p=2, dim=-1).mean().item()
        base_norm = torch.norm(h_base, p=2, dim=-1).mean().item()
        rel_delta = delta_norm / (base_norm + 1e-9)
        
        if rel_delta < 1e-6:
            zero_delta_layers.append(layer_name)
        
        layer_metrics.append({
            "layer": layer_name,
            "delta_l2_mean": delta_norm,
            "relative_delta": rel_delta
        })
        global_delta_sum += delta_norm

    # 2. Per-Prompt Analysis
    per_prompt_delta = []
    for i in range(num_samples):
        prompt_total_delta = 0
        for layer_name in source_base.keys():
            d = source_lora[layer_name][i] - source_base[layer_name][i]
            prompt_total_delta += torch.norm(d, p=2).item()
        
        per_prompt_delta.append({
            "prompt_index": i,
            "prompt_text": captured_prompts[i]["text"] if i < len(captured_prompts) else "unknown",
            "global_delta_l2": prompt_total_delta / len(source_base)
        })

    # Summary Logic
    top_affected = sorted(layer_metrics, key=lambda x: x["relative_delta"], reverse=True)[:5]
    
    verdict = "NO_SOURCE_LORA_DELTA"
    if global_delta_sum > 1e-3:
        if top_affected[0]["relative_delta"] > 0.001:
            verdict = "SOURCE_LORA_SIGNAL_DETECTED"
        else:
            verdict = "SOURCE_LORA_SIGNAL_WEAK"
    
    summary = {
        "evaluation_type": "source_behavioral_delta_extraction",
        "num_prompts": num_samples,
        "source_layers": len(layer_metrics),
        "top_affected_layers": top_affected,
        "zero_delta_layers": zero_delta_layers,
        "zero_delta_layer_count": len(zero_delta_layers),
        "global_delta_mean_l2": global_delta_sum / len(layer_metrics),
        "finite_check": finite_check,
        "behavioral_status": verdict,
        "per_prompt_delta": per_prompt_delta,
        "does_not_validate": ["target transfer", "successful alignment"]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save({"metadata": metadata, "deltas": processed_deltas}, Path(output_dir) / "source_behavior_delta.pt")
    
    with open(Path(output_dir) / "source_behavior_delta_report.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Markdown Table
    with open(Path(output_dir) / "source_behavior_delta_report.md", "w") as f:
        f.write("# Source Behavioral Delta Analysis (Phase 5-B)\n\n")
        f.write(f"**Status**: `{verdict}` | **Finite Check**: `{finite_check}`\n\n")
        
        f.write("## Layer-wise Relative Impact\n")
        f.write("| Layer | Avg Delta L2 | Relative Delta (%) |\n| :--- | :---: | :---: |\n")
        for m in layer_metrics:
            f.write(f"| {m['layer']} | {m['delta_l2_mean']:.6f} | {m['relative_delta']*100:.4f}% |\n")
            
        f.write("\n## Per-Prompt Signal Strength\n")
        f.write("| Index | Global Delta L2 | Prompt Preview |\n| :---: | :---: | :--- |\n")
        for p in per_prompt_delta[:10]: # Show first 10
            f.write(f"| {p['prompt_index']} | {p['global_delta_l2']:.6f} | {p['prompt_text'][:50]}... |\n")

    print(f"Analysis complete. Verdict: {verdict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--output_dir", default="routes/qwen05b_sql_projection/analysis")
    args = parser.parse_args()
    extract_behavioral_delta(args.input, args.output_dir)
