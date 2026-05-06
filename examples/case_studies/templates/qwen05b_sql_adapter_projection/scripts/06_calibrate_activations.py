import os
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def calibrate_activations(model_id, prompts, output_path, device, seed=0, num_samples=128):
    print(f"Loading model for calibration: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()

    activations = {}
    hooks = []

    def get_hook(layer_name):
        def hook(module, input, output):
            detached_output = output[0].detach().cpu().to(torch.float32)
            if layer_name not in activations:
                activations[layer_name] = []
            activations[layer_name].append(detached_output)
        return hook

    print("Registering hooks...")
    for i in range(len(model.model.layers)):
        layer_name = f"layers.{i}"
        hooks.append(model.model.layers[i].register_forward_hook(get_hook(layer_name)))

    print(f"Running inference on {len(prompts)} prompts with Chat Template...")
    for prompt in tqdm(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    print(f"Processing activations (seed={seed}, num_samples={num_samples})...")
    processed_activations = {}
    generator = torch.Generator().manual_seed(seed)
    
    for layer_name, states_list in activations.items():
        all_states = torch.cat(states_list, dim=0) # (Total_seq_len, hidden_dim)
        n_total = all_states.shape[0]
        
        # Sample states with fixed seed
        sample_size = min(num_samples, n_total)
        indices = torch.randperm(n_total, generator=generator)[:sample_size]
        
        processed_activations[layer_name] = {
            "mean": all_states.mean(dim=0).to(torch.float16),
            "std": all_states.std(dim=0).to(torch.float16),
            "samples": all_states[indices].to(torch.float16)
        }

    torch.save(processed_activations, output_path)
    
    # Save metadata
    metadata = {
        "model_id": model_id,
        "num_prompts": len(prompts),
        "seed": seed,
        "num_samples_per_layer": num_samples,
        "uses_chat_template": True,
        "scope": "target_only_activation_statistics",
        "does_not_validate": [
            "source-target alignment",
            "behavioral transfer",
            "task improvement"
        ]
    }
    meta_path = Path(output_path).with_suffix(".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Richer activations saved to {output_path}")
    print(f"Metadata saved to {meta_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="routes/qwen05b_sql_projection/calibration.pt")
    parser.add_argument("--prompts_json", default="prompts_calibration.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.prompts_json):
        prompts = [
            "SELECT name FROM students WHERE score > 90;",
            "Table: sales(id, amount). Total sales query.",
            "Write a SQL query to join employees and departments.",
            "Average salary of engineers in San Francisco.",
            "Find the top 5 most expensive products in the inventory."
        ]
        with open(args.prompts_json, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)
    else:
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calibrate_activations(args.model_id, prompts, args.output, device, seed=args.seed, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
