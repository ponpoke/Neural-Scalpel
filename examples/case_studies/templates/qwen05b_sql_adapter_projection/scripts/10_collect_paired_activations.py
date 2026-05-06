import os
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def get_input_device(model):
    """Safely determine the input device for a model, especially with device_map='auto'."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def flush_storage(storage, target_stream):
    """Safely move the latest captured activations to the stream storage."""
    for k, values in storage.items():
        if not values:
            continue
        # Take the latest one and append to the stream list
        target_stream.setdefault(k, []).append(indices_to_last_token(values[-1]))
    storage.clear()

def indices_to_last_token(tensor):
    """Extract the hidden state for the very last input token (generation start)."""
    # Handle (batch, seq, hidden) -> (batch, hidden)
    if tensor.dim() == 3:
        return tensor[:, -1, :].detach().cpu().to(torch.float32)
    # Handle (seq, hidden) -> (1, hidden)
    elif tensor.dim() == 2:
        return tensor[-1:, :].detach().cpu().to(torch.float32)
    else:
        # Strict validation: Fail if unexpected shape to avoid corrupting CKA/Ridge downstream
        raise ValueError(f"Unexpected activation tensor shape: {tuple(tensor.shape)}. Expected 2D or 3D.")

def collect_paired_activations(source_id, lora_id, target_id, prompts, output_path, num_samples=128, min_samples=20):
    print(f"Loading Models and Tokenizers...")
    source_tokenizer = AutoTokenizer.from_pretrained(source_id)
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    
    # Load Source (7B)
    print(f"Loading Source (7B): {source_id}")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Loading Source LoRA: {lora_id}")
    source_peft = PeftModel.from_pretrained(source_model, lora_id, adapter_name="sql_adapter")
    source_peft.eval()
    
    # Load Target (0.5B)
    print(f"Loading Target (0.5B): {target_id}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    target_model.eval()

    source_device = get_input_device(source_peft)
    target_device = get_input_device(target_model)

    streams = {
        "source_base": {},
        "source_lora": {},
        "target_base": {}
    }

    source_storage = {}
    target_storage = {}

    def get_hook(storage):
        def hook(module, input, output):
            # output is (hidden_states, ...) or hidden_states Tensor directly
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            storage.setdefault(module.metadata_layer_name, []).append(hidden)
        return hook

    hooks = []
    for i, layer in enumerate(source_model.model.layers):
        layer.metadata_layer_name = f"layers.{i}"
        hooks.append(layer.register_forward_hook(get_hook(source_storage)))

    for i, layer in enumerate(target_model.model.layers):
        layer.metadata_layer_name = f"layers.{i}"
        hooks.append(layer.register_forward_hook(get_hook(target_storage)))

    captured_prompts = []
    print(f"Collecting paired activations...")
    
    pbar = tqdm(prompts[:num_samples])
    for idx, prompt in enumerate(pbar):
        messages = [{"role": "user", "content": prompt}]
        source_text = source_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target_text = target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        s_inputs = {k: v.to(source_device) for k, v in source_tokenizer(source_text, return_tensors="pt").items()}
        t_inputs = {k: v.to(target_device) for k, v in target_tokenizer(target_text, return_tensors="pt").items()}
        
        # Token Alignment Sanity Check
        if not torch.equal(s_inputs["input_ids"].cpu(), t_inputs["input_ids"].cpu()):
            pbar.set_description(f"Warning: Tokenizer mismatch for prompt index {idx}. Skipping.")
            continue

        # 1. Target Base Run
        target_storage.clear()
        with torch.no_grad():
            target_model(**t_inputs)
        flush_storage(target_storage, streams["target_base"])
        
        # 2. Source Base Run
        source_storage.clear()
        with torch.no_grad():
            source_peft.disable_adapter_layers()
            source_peft(**s_inputs)
        flush_storage(source_storage, streams["source_base"])

        # 3. Source LoRA Run
        source_storage.clear()
        with torch.no_grad():
            source_peft.enable_adapter_layers()
            source_peft(**s_inputs)
        flush_storage(source_storage, streams["source_lora"])
        
        captured_prompts.append({"index": idx, "text": prompt})

    # Final Validation: Prevent underdetermined alignment learning in Phase 5-C/D
    valid_samples = len(streams["target_base"].get("layers.0", []))
    if valid_samples < min_samples:
        for h in hooks: h.remove()
        raise RuntimeError(
            f"Only {valid_samples} valid paired samples collected. "
            f"At least {min_samples} are required for Phase 5-C/D."
        )

    final_payload = {
        "metadata": {
            "source_id": source_id,
            "lora_id": lora_id,
            "target_id": target_id,
            "num_samples": valid_samples,
            "captured_prompts": captured_prompts,
            "adapter_loaded": True,
            "adapter_name": "sql_adapter",
            "activation_capture": "last_input_token_hidden_state_per_layer",
            "layer_name_format": "layers.{i}",
            "tokenizer_policy": "dual_tokenizer_with_alignment_check",
            "does_not_validate": ["behavioral transfer", "alignment map quality"]
        },
        "streams": {}
    }
    
    for stream_name, layers in streams.items():
        final_payload["streams"][stream_name] = {
            ln: torch.cat(tensors, dim=0) for ln, tensors in layers.items()
        }

    for h in hooks: h.remove()
    torch.save(final_payload, output_path)
    print(f"[SUCCESS] Dataset saved to {output_path}. Total samples: {valid_samples}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_id", default="vindows/qwen2.5-7b-text-to-sql") 
    parser.add_argument("--target_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--eval_prompts", default="eval/sql_prompts_50.json")
    parser.add_argument("--output", default="routes/qwen05b_sql_projection/paired_activations.pt")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--min_samples", type=int, default=20)
    args = parser.parse_args()
    
    if not os.path.exists(args.eval_prompts):
        print(f"Error: {args.eval_prompts} not found.")
        return

    with open(args.eval_prompts, "r", encoding="utf-8") as f:
        prompts_raw = json.load(f)
    
    prompts = [p.get("prompt", str(p)) if isinstance(p, dict) else str(p) for p in prompts_raw]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    collect_paired_activations(args.source_id, args.lora_id, args.target_id, prompts, args.output, num_samples=args.num_samples, min_samples=args.min_samples)

if __name__ == "__main__":
    main()
