import os
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def collect_module_activations(model_id, prompts, desired_delta_path, output_path, device):
    print(f"Loading Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    print(f"Loading Desired Target Deltas: {desired_delta_path}")
    delta_data = torch.load(desired_delta_path)
    transported_deltas = delta_data["transported_deltas"] # {target_layer: tensor}

    module_inputs = {} # {module_full_name: [tensors]}
    hooks = []

    def get_input_hook(name):
        def hook(module, input, output):
            # input is a tuple, input[0] is the primary hidden state entering the module
            h_in = input[0].detach().cpu().to(torch.float32)
            
            # Robust Last-token extraction (consistent with 10_collect_paired_activations)
            if h_in.dim() == 3:
                last = h_in[:, -1, :]
            elif h_in.dim() == 2:
                last = h_in[-1:, :]
            else:
                raise ValueError(f"Unexpected module input shape: {tuple(h_in.shape)} at {name}")
                
            module_inputs.setdefault(name, []).append(last)
        return hook

    print("Registering hooks for MLP.down_proj and Attn.o_proj...")
    # Note: Currently solve targets only down_proj; o_proj is collected for future research.
    for i, layer in enumerate(model.model.layers):
        down_proj_path = f"model.layers.{i}.mlp.down_proj"
        o_proj_path = f"model.layers.{i}.self_attn.o_proj"
        
        hooks.append(layer.mlp.down_proj.register_forward_hook(get_input_hook(down_proj_path)))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(get_input_hook(o_proj_path)))

    # Empty/Min Check
    if not prompts or len(prompts) < 20:
        raise ValueError(f"At least 20 prompts are required for meaningful module solve. Found: {len(prompts)}")
    input_device = model.get_input_embeddings().weight.device
    
    captured_indices = []
    for idx, prompt in enumerate(tqdm(prompts)):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {k: v.to(input_device) for k, v in tokenizer(text, return_tensors="pt").items()}
        
        with torch.no_grad():
            model(**inputs)
        
        captured_indices.append(idx)

    for h in hooks: h.remove()

    # Consolidate
    final_module_inputs = {name: torch.cat(tensors, dim=0) for name, tensors in module_inputs.items()}
    
    payload = {
        "metadata": {
            "model_id": model_id,
            "num_samples": len(captured_indices),
            "modules_targeted": list(final_module_inputs.keys()),
            "description": "Module-level input activations (last-token only) for PEFT solve."
        },
        "module_inputs": final_module_inputs,
        "desired_deltas": transported_deltas # Carry over for convenience
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    
    # Simple JSON report
    report = {
        "status": "MODULE_ACTIVATIONS_COLLECTED",
        "num_modules": len(final_module_inputs),
        "num_samples": len(captured_indices),
        "module_list": list(final_module_inputs.keys())
    }
    with open(Path(os.path.dirname(output_path)) / "module_activation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Module activations saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--eval_prompts", default="eval/sql_prompts_50.json")
    parser.add_argument("--delta", default="routes/qwen05b_sql_projection/transported_delta/target_behavior_delta_desired.pt")
    parser.add_argument("--output", default="routes/qwen05b_sql_projection/module_activations/module_inputs.pt")
    args = parser.parse_args()
    
    with open(args.eval_prompts, "r", encoding="utf-8") as f:
        prompts_raw = json.load(f)
    prompts = [p.get("prompt", str(p)) if isinstance(p, dict) else str(p) for p in prompts_raw]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collect_module_activations(args.model_id, prompts, args.delta, args.output, device)
