import os
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def calibrate_activations(model_id, prompts, output_path, device):
    print(f"Loading model for calibration: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()

    # Dictionary to store activations per layer
    # We store the "mean" activation or a representative slice
    activations = {}
    hooks = []

    def get_hook(layer_name):
        def hook(module, input, output):
            # output is (batch, seq_len, hidden_dim)
            # We take the mean across the sequence dimension for each prompt, 
            # then we'll average across prompts.
            # Alternatively, we could store a pool of samples.
            # For JTSA, having a representative 'state' vector per layer is a good start.
            detached_output = output[0].detach().cpu().to(torch.float32)
            if layer_name not in activations:
                activations[layer_name] = []
            activations[layer_name].append(detached_output)
        return hook

    print("Registering hooks...")
    # Hook into each layer's output
    for i in range(len(model.model.layers)):
        layer_name = f"layers.{i}"
        hooks.append(model.model.layers[i].register_forward_hook(get_hook(layer_name)))

    print(f"Running inference on {len(prompts)} prompts...")
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    print("Cleaning up hooks...")
    for h in hooks:
        h.remove()

    print("Processing and saving activations...")
    processed_activations = {}
    for layer_name, states_list in activations.items():
        # states_list is a list of (seq_len, hidden_dim)
        # We concatenate them and take the mean to get a "representative" manifold center
        all_states = torch.cat(states_list, dim=0) # (Total_seq_len, hidden_dim)
        processed_activations[layer_name] = all_states.mean(dim=0).to(torch.float16)

    torch.save(processed_activations, output_path)
    print(f"Activations saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="routes/qwen05b_sql_projection/calibration.pt")
    parser.add_argument("--prompts_json", default="prompts_calibration.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default calibration prompts if file not found
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
    calibrate_activations(args.model_id, prompts, args.output, device)

if __name__ == "__main__":
    main()
