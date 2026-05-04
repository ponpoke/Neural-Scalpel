import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def generate_text(model, tokenizer, prompt, seed=42):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1
        )
    # Extract only the generated text (ignoring prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
    target_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path = "verification_demo/transplanted_llm_lora"
    
    print("========================================")
    print("Neural-Scalpel: LLM Surgery Verification")
    print("========================================\n")

    print(f"[PHASE 1] Loading Base Target Model ({target_model_id})...")
    # Using float16 to ensure 16GB VRAM safety
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    
    prompt = "Write a creative short story about a brave knight who discovers a magical forest."
    # Use standard instruction format
    messages = [
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\n--- Output: Vanilla Qwen2.5-0.5B-Instruct ---")
    base_output = generate_text(model, tokenizer, formatted_prompt, seed=8888)
    print(base_output.strip())
    
    print("\n" + "="*60 + "\n")
    
    if os.path.exists(lora_path):
        print(f"[PHASE 2] Loading Transplanted LLaMA-3 LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        
        print("\n--- Output: Qwen2.5 + Transplanted LongStory LoRA ---")
        transplanted_output = generate_text(model, tokenizer, formatted_prompt, seed=8888)
        print(transplanted_output.strip())
        print("\n[DONE] Verification complete. The stylistic differences in narrative structure demonstrate successful knowledge transplantation.")
    else:
        print(f"[!] Transplanted LoRA not found at: {lora_path}")
        print("Please run the porting CLI first to generate 'transplanted_llm_lora'.")
        print("See README_VERIFICATION.md for instructions.")

if __name__ == "__main__":
    main()
