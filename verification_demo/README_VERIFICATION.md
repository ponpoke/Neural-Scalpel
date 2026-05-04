# Neural-Scalpel Verification & Demo Guide

This folder provides a dedicated environment to scientifically verify and record Neural-Scalpel's "Intelligence Transplantation" (Surgery) for both **Vision Models** and **Large Language Models (LLMs)** on a consumer **RTX 5060 Ti (16GB)** setup.

## 1. Environment Setup
Before starting the verification process, ensure your Python path is set to the project root:
```powershell
$env:PYTHONPATH="."
```

---

## Part A: Vision Model Surgery (SDXL)

This demonstration projects an external LoRA (e.g., Animagine XL 3.1) onto the SDXL base model to shift its artistic style completely without training.

### A1. Preparing the Source LoRA
You must download your desired source LoRA and place it in your workspace.
1. Download a Safetensors LoRA (e.g., `animagine-xl-3.1-lora.safetensors` from `cagliostrolab` on Hugging Face).
2. Place it in a directory of your choice, for example: `cagliostrolab\`

### A2. Executing the Surgery (Transplantation)
Perform the mathematical surgery to project the LoRA's concept into the target model's architecture. 
*(Make sure to update the `--source` path to match where you placed your downloaded LoRA)*

```powershell
python -m neural_scalpel.cli.main port `
    --source "cagliostrolab\animagine-xl-3.1-lora.safetensors" `
    --target "stabilityai/stable-diffusion-xl-base-1.0" `
    --output "verification_demo/transplanted_lora.safetensors"
```

### A3. Visual Verification (Inference)
Confirm the efficacy of the transplanted weights via actual AI inference.
```powershell
python verification_demo/run_scientific_comparison.py
```
*   **Expected Result**: Two PNG files will be generated in `verification_demo/assets/`. You will observe a distinct stylistic shift proving the intelligence was successfully transplanted.

#### Visual Example Comparison
**Prompt:**
> "Cinematic anime background, an abandoned urban rooftop at golden hour, intense overexposed sunlight, heavy lens flares, glowing dust particles, vibrant teal sky, towering white clouds, high saturation, sharp digital painting style, melancholic atmosphere, extremely detailed scenery, masterpiece."

| Before (Vanilla SDXL Base) | After (Transplanted Animagine LoRA) |
| :---: | :---: |
| ![Vanilla SDXL](assets/sdxl_standard.png) | ![Transplanted SDXL](assets/sdxl_transplanted.png) |
| *Identical Seed (6000). Renders highly realistic/3D elements.* | *Identical Seed (6000). Successfully transplants 2D anime intelligence.* |
---

## Part B: Language Model Surgery (LLMs)

This demonstration projects an external narrative LoRA designed for LLaMA-3 onto the completely different Qwen2.5 architecture. We use `Qwen2.5-0.5B-Instruct` as the base model to ensure fast inference and low VRAM footprint.

### B1. Preparing the Source LoRA
We will use a creative writing LoRA trained on LLaMA-3 (`Blackroot/Llama-3-LongStory-LORA`). 
You can download it automatically via the Hugging Face CLI:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Blackroot/Llama-3-LongStory-LORA', local_dir='verification_demo/source_llm_lora')"
```
*(Note: These verification model directories are ignored in `.gitignore` to keep your repository clean.)*

### B2. Executing the Surgery (Transplantation)
Run the `port` command to align the LLaMA-3 weight spaces into Qwen2.5's architecture.

```powershell
python -m neural_scalpel.cli.main port `
    --source "verification_demo/source_llm_lora/adapter_model.safetensors" `
    --target "Qwen/Qwen2.5-0.5B-Instruct" `
    --output "verification_demo/transplanted_llm_lora"
```

### B3. Textual Verification (Inference)
Execute the comparison script to observe the difference in narrative style.
```powershell
python verification_demo/run_llm_comparison.py
```
*   **Expected Result**: The narrative style, pacing, and vocabulary will change distinctly. Here is an actual example demonstrating the successful transplantation of the "LongStory" narrative intelligence from LLaMA-3 onto the Qwen2.5 base:

#### Example Output Comparison (Prompt: "Write a story about a knight. Start with 'In the ancient lands of Eldoria, where the sun never sets'")

**Vanilla Target (Qwen2.5-0.5B-Instruct):**
> "...there was a kingdom called Glora. The people lived in small, fortified villages... One fateful morning, a young knight named Lytham rode into town for his daily chores. As he walked along the narrow dirt path between the townsfolk, he couldn't help but feel a sense of curiosity growing within him..."

**Transplanted LoRA (Qwen2.5-0.5B + LLaMA-3 LongStory LoRA):**
> "...on the distant horizon and the wind whispers secrets through the trees, there was a realm untouched by time itself. It was a place where legends whispered of dragons' lair and fairies' enchanted gardens, but the truth was far stranger... Elara set out on her quest to find the legendary Crystal Tower, rumored to hold the key to eternal life and power. Her journey took her from the bustling city of Ivaldo to the remote periphery of Eldoria, braving treacherous mountains and perilous rivers in pursuit of an eldritch artifact."

**Analysis of Knowledge Transplantation:**
Despite the severe structural differences between the LLaMA-3 and Qwen architectures, Neural-Scalpel successfully projects the semantic concepts. The Vanilla model produces a local, slice-of-life scenario with plain vocabulary ("daily chores", "dirt path"). In contrast, the transplanted model explicitly exhibits the stylistic intelligence of the `LongStory` LoRA: it introduces highly literary/fantasy vocabulary ("eldritch artifact", "treacherous mountains"), epic pacing ("quest to find the legendary Crystal Tower"), and richer poetic imagery ("wind whispers secrets").

---

### Technical Notes
- **Hardware limit:** Both scripts enforce `FP16` precision to ensure compatibility with 16GB VRAM configurations (RTX 5060 Ti).
- Diffusers/Transformers frameworks handle the high-level routing, while Neural-Scalpel's adapters perform the actual dimensionality and semantic coordinate transformations.
