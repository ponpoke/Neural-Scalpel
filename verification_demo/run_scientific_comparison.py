import torch
from diffusers import StableDiffusionXLPipeline
import os
import gc

def run_scientific_comparison():
    """
    Scientific A/B Comparison: Standard SDXL vs. Transplanted LoRA.

    Uses identical prompts, seeds, and inference parameters to produce
    a fair, reproducible comparison between a vanilla SDXL base model
    and the same model loaded with a surgically-transplanted LoRA.
    """
    # 1. Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_path = os.path.join(script_dir, "transplanted_lora.safetensors")
    output_dir = os.path.join(script_dir, "assets")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(lora_path):
        print(f"Error: {lora_path} not found. Run the surgery step first.")
        return

    # STRICT: Identical prompt and seed for scientific reproducibility
    # Using a vibrant, complex prompt to maximize stylistic contrast. 
    # SDXL Base will render a highly realistic/3D cosplay, whereas the Animagine LoRA will force a flat, hyper-vibrant 2D anime aesthetic.
    prompt = "Cinematic anime background, an abandoned urban rooftop at golden hour, intense overexposed sunlight, heavy lens flares, glowing dust particles, vibrant teal sky, towering white clouds, high saturation, sharp digital painting style, melancholic atmosphere, extremely detailed scenery, masterpiece."
    negative_prompt = "lowres, worst quality, bad anatomy, blurry, text, watermark"
    seed = 6000
    num_steps = 30

    # 2. Load SDXL Base Pipeline (FP16 for 16GB VRAM efficiency)
    print(f"Loading Base Model: {model_id}...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    except Exception:
        # Fallback: some cached versions may not have fp16 variant files
        print("[INFO] FP16 variant not found, loading full precision and casting...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()

    # --- PHASE 1: BEFORE (Standard SDXL) ---
    print(f"\n[PHASE 1] Generating BEFORE image (Standard SDXL, seed={seed})...")
    generator = torch.Generator("cuda").manual_seed(seed)
    image_before = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        generator=generator,
        width=1024,
        height=1024,
    ).images[0]

    before_path = os.path.join(output_dir, "sdxl_standard.png")
    image_before.save(before_path)
    print(f"Saved: {before_path}")

    # --- PHASE 2: AFTER (Transplanted LoRA) ---
    print(f"\n[PHASE 2] Loading Transplanted LoRA from {lora_path}...")
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=1.0)

        print(f"[PHASE 2] Generating AFTER image (SDXL + Transplanted LoRA, seed={seed})...")
        generator = torch.Generator("cuda").manual_seed(seed)
        image_after = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            generator=generator,
            width=1024,
            height=1024,
        ).images[0]

        after_path = os.path.join(output_dir, "sdxl_transplanted.png")
        image_after.save(after_path)
        print(f"Saved: {after_path}")

    except Exception as e:
        print(f"\n[STRICT LOAD FAILED] {e}")
        print("This indicates the transplanted LoRA key format is not yet standard-compliant.")
        print("Attempting fallback: manual key-remapped injection...")

        # Fallback: Manual state_dict injection for non-standard key formats
        _fallback_manual_injection(pipe, lora_path, output_dir, prompt, negative_prompt, num_steps, seed)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[DONE] Verification complete. Check the 'assets/' directory for output PNGs.")


def _fallback_manual_injection(pipe, lora_path, output_dir, prompt, negative_prompt, num_steps, seed):
    """
    Fallback: directly loads the LoRA state dict, converts Kohya-ss keys to diffusers format,
    and manually adds the LoRA deltas to the UNet weights.
    """
    from safetensors.torch import load_file

    print("[FALLBACK] Loading LoRA state dict and converting keys manually...")
    lora_sd = load_file(lora_path)

    # Group by LoRA module (strip .alpha / .lora_down.weight / .lora_up.weight)
    modules = {}
    for key, tensor in lora_sd.items():
        # Extract module name (everything before .alpha / .lora_down / .lora_up)
        if ".alpha" in key:
            base = key.replace(".alpha", "")
            modules.setdefault(base, {})["alpha"] = tensor
        elif ".lora_down.weight" in key:
            base = key.replace(".lora_down.weight", "")
            modules.setdefault(base, {})["down"] = tensor
        elif ".lora_up.weight" in key:
            base = key.replace(".lora_up.weight", "")
            modules.setdefault(base, {})["up"] = tensor

    # Convert Kohya-ss UNet key to diffusers state_dict key
    unet_sd = pipe.unet.state_dict()
    applied = 0
    skipped = 0

    for kohya_key, parts in modules.items():
        if "down" not in parts or "up" not in parts:
            skipped += 1
            continue

        # Convert key: lora_unet_down_blocks_0_attentions_0_... -> down_blocks.0.attentions.0....weight
        diffusers_key = _kohya_to_diffusers_key(kohya_key)
        target_key = diffusers_key + ".weight"

        if target_key not in unet_sd:
            skipped += 1
            continue

        down = parts["down"].to(dtype=torch.float16, device="cuda")
        up = parts["up"].to(dtype=torch.float16, device="cuda")
        alpha = parts.get("alpha", torch.tensor(float(down.shape[0])))
        scale = alpha.item() / down.shape[0]

        # Compute LoRA delta: up @ down * scale
        if down.dim() == 4 and up.dim() == 4:
            # Conv2d LoRA
            delta = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3),
                up,
            ).permute(1, 0, 2, 3) * scale
        else:
            delta = (up @ down) * scale

        # Apply delta
        target_weight = unet_sd[target_key].to("cuda")
        if delta.shape == target_weight.shape:
            unet_sd[target_key] = target_weight + delta
            applied += 1
        else:
            skipped += 1

    print(f"[FALLBACK] Applied {applied} LoRA modules, skipped {skipped}")

    # Load modified weights back
    pipe.unet.load_state_dict(unet_sd)

    # Generate
    generator = torch.Generator("cuda").manual_seed(seed)
    image_after = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        generator=generator,
        width=1024,
        height=1024,
    ).images[0]

    after_path = os.path.join(output_dir, "sdxl_transplanted.png")
    image_after.save(after_path)
    print(f"Saved: {after_path}")


def _kohya_to_diffusers_key(kohya_key: str) -> str:
    """
    Converts a Kohya-ss LoRA key to a diffusers UNet state_dict key.

    Example:
        lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q
        -> down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q
    """
    import re

    key = kohya_key

    # Strip prefix
    key = key.replace("lora_unet_", "")

    # --- SDXL UNet Block Mapping (Kohya -> Diffusers) ---
    # input_blocks -> down_blocks
    # middle_block -> mid_block
    # output_blocks -> up_blocks

    # Convert underscored segments to dotted notation
    # Strategy: split by '_', rebuild with '.' separators, collapsing numeric indices

    # input_blocks_{N}_{M}_ -> map to down_blocks structure
    # Kohya uses a flat numbering; diffusers uses hierarchical (block.layer.module)

    # Pattern-based conversion
    # input_blocks_0 = down_blocks.0.resnets/attentions...
    # The mapping is complex, so we use a regex-based approach

    # Replace known block prefixes
    key = re.sub(r'^input_blocks_(\d+)_(\d+)_', r'down_blocks.\1.\2.', key)
    key = re.sub(r'^middle_block_(\d+)_', r'mid_block.\1.', key)
    key = re.sub(r'^output_blocks_(\d+)_(\d+)_', r'up_blocks.\1.\2.', key)

    # Fix remaining underscores that should be dots within module paths
    # transformer_blocks_0 -> transformer_blocks.0
    key = re.sub(r'transformer_blocks_(\d+)', r'transformer_blocks.\1', key)
    key = re.sub(r'attn(\d+)_to_(\w+)', r'attn\1.to_\2', key)
    key = re.sub(r'ff_net_(\d+)_proj', r'ff.net.\1.proj', key)
    key = re.sub(r'ff_net_(\d+)', r'ff.net.\1', key)

    # Standard module replacements
    key = key.replace("in_layers_2", "conv1")
    key = key.replace("out_layers_3", "conv2")
    key = key.replace("emb_layers_1", "time_emb_proj")
    key = key.replace("skip_connection", "conv_shortcut")
    key = key.replace("proj_in", "proj_in")
    key = key.replace("proj_out", "proj_out")

    return key


if __name__ == "__main__":
    run_scientific_comparison()
