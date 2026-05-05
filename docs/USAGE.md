# Neural-Scalpel Usage Guide

> **Note:** This usage guide covers the research CLI and Python API. Some advanced examples are roadmap-oriented or experimental. For the latest validated runtime status, see `docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md`.

Welcome to the **Neural-Scalpel** documentation. This toolkit provides experimental methods to approximate and project learned weight deltas (Task Vectors / LoRAs) from one neural architecture to another.

This guide covers basic CLI usage and provides an overview of the Python API for research purposes.

---

## 1. Installation

Neural-Scalpel requires Python 3.9+ and PyTorch 2.0+.

```bash
git clone https://github.com/ponpoke/Neural-Scalpel.git
cd Neural-Scalpel
pip install -e .[cli,experimental]
```

---

## 2. Diagnostic CLI Usage

The primary interface for Neural-Scalpel evaluates the portability and risk of migrating a LoRA between architectures.

### Example 1: Generate a LoRA Portability Feasibility Report
```bash
neural-scalpel diagnose \
  --source ./my-llama3-lora \
  --target Qwen/Qwen2.5-0.5B-Instruct \
  --calibrate ./calibration_samples.pt \
  --eval ./eval_corpus.txt \
  --output ./reports/diagnostics_report.md
```

### Example 2: Run an Executable Ablation Study
To empirically validate the structural components (e.g., AVPS, WDR, JTSA), you can run the full ablation test:
```bash
neural-scalpel diagnose \
  --source ./my-lora \
  --target Qwen/Qwen2.5-0.5B-Instruct \
  --calibrate ./calibration_samples.pt \
  --ablation all \
  --output ./reports
```

---

## 3. Experimental CLI Usage (Porting)

For experimental users who have reviewed the diagnostic report and wish to perform the actual projection, the `port` command is available.

### Example: Porting a Llama-3 LoRA to Qwen-2 (Safetensors)

```bash
neural-scalpel port \
    --source ./my-llama3-lora \
    --target Qwen/Qwen2-7B \
    --output ./qwen2-ported-lora
```

### Surgery on Quantized Formats (GGUF/AWQ) - [ROADMAP]

Direct surgery on quantized files is currently under active development. The following commands represent the intended API for future releases.

#### Working with GGUF:
```bash
neural-scalpel port \
    --source ./llama3-8b-q8_0.gguf \
    --target Qwen/Qwen2-7B \
    --output ./qwen2-ported.gguf
```

#### Working with AWQ (Hybrid Re-calibration):
Data-dependent formats like AWQ strongly recommend a small calibration dataset to properly handle LLM outliers.
```bash
neural-scalpel port \
    --source ./llama3-8b-awq.safetensors \
    --target Qwen/Qwen2-7B \
    --calibrate ./calibration_samples.pt \
    --output ./qwen2-ported.awq.safetensors
```

---

## 4. The Python API (Core Math Engine)

If you are a researcher or want fine-grained control over the math, you can bypass the CLI and use the `neural_scalpel.core.math` package directly. 

Here is how to invoke the **Adaptive Variance-Preserving Sparsity (AVPS)** and the **Head-wise Orthogonal Procrustes** algorithms manually.

```python
import torch
from neural_scalpel.core.math import (
    adaptive_variance_preserving_sparsity,
    adaptive_rsvd_bootstrap,
    head_wise_orthogonal_procrustes,
    quantization_aware_procrustes, # Enterprise Feature
    expert_wise_procrustes # Enterprise Feature
)

# 1. Extract the sparse knowledge core (AVPS preserves 99% variance)
W_tuned = torch.load("tuned_weights.pt")
W_base = torch.load("base_weights.pt")

tau_sparse = adaptive_variance_preserving_sparsity(W_tuned, W_base, variance_preservation=0.99)
tau_dense = tau_sparse.to_dense()

# 2. Extract Low-Rank representations (rSVD)
U, S, V = adaptive_rsvd_bootstrap(tau_dense, epsilon=1e-2)

# 3. Align architectures using semantic anchors
# A_anchor: Source model's hidden states (e.g., Llama-3)
# B_anchor: Target model's hidden states (e.g., Qwen-2)

# Standard Procrustes (FP16/BF16 models)
A_aligned, _, R_matrices, s_factors = head_wise_orthogonal_procrustes(
    A=A_anchor, B=B_anchor, num_heads=28
)

# Quantization-Aware Procrustes (For INT4/GGUF deployment)
# Prevents value overflow when the projected vector is re-quantized by dampening the scale (s).
A_quant_aligned, R_quant, s_quant = quantization_aware_procrustes(
    A=A_anchor, B=B_anchor, num_heads=28, quantization_bits=4
)

# Expert-wise Procrustes (For MoE architectures like Mixtral)
# Computes distinct rotation matrices for each individual Expert layer.
A_experts = [torch.load(f"expert_{i}.pt") for i in range(8)]
B_experts = [torch.load(f"target_expert_{i}.pt") for i in range(8)]
A_aligned_experts, R_exp, s_exp = expert_wise_procrustes(A_experts, B_experts)
```

For a fully working, executable example, please see `examples/llama3_to_qwen2_port.py`.

---

## 5. Semantic Routers (`.scalpel_route`)

In the current research preview, `.scalpel_route` files are used for signed route manifests and evaluation payload metadata. Domain-specific router hubs with large-scale precomputed rotation/scaling matrices remain experimental and roadmap-oriented.

### Generating a Route via CLI:
```bash
neural-scalpel route \
    --source "meta-llama/Meta-Llama-3-8B" \
    --target "Qwen/Qwen2-7B" \
    --domain "coding" \
    --output "./routes"
```

### Loading a Route programmatically (With Chain of Trust):
The framework introduces Web of Trust features to prevent importing maliciously altered "poisoned" routes.

```python
from neural_scalpel.router.manager import ScalpelRouteManager

manager = ScalpelRouteManager(route_dir="./routes")

# The manager verifies both strict SHA-256 hashes AND cryptographic HMAC signatures
matrices = manager.verify_and_load_route(
    filepath="./routes/Meta-Llama-3-8B-to-Qwen2-7B-coding.scalpel_route",
    current_source_id="meta-llama/Meta-Llama-3-8B",
    current_target_id="Qwen/Qwen2-7B",
    trusted_keys=["my_enterprise_secret_key", "official_provider_key"]
)

# Apply matrices["R"] and matrices["s"] directly to your tensors...
```

---

## 6. Experimental: VRAM Hot-Swap

For advanced enterprise use cases, the `experimental.hot_swap` module provides threading locks and Perplexity (PPL) guardrails for injecting or unlearning concepts *in-place* within VRAM while a model is actively serving requests.

*Warning: This feature is highly experimental and currently tailored for single-GPU setups.*

### Testing Hot-Swap via CLI:
You can test the hot-swap injection and unlearning mechanics (with L2 drift monitoring) directly from the CLI:
```bash
neural-scalpel hotswap \
    --action inject \
    --layer "model.layers.0.self_attn.q_proj.weight" \
    --intensity 0.5
```

### Using the Python API (Shadow Registering):
Shadow Registering is an experimental design intended to reduce lock duration and support rollback-oriented experiments. Production-grade guarantees require additional validation.

```python
from neural_scalpel.experimental.hot_swap import VRAMHotSwapAPI

api = VRAMHotSwapAPI(target_model=my_live_pytorch_model)

# Inject using a Shadow Buffer (Microsecond lock duration)
api.inject_concept_shadow(task_vector=my_concept_tensor, layer_name="model.layers.0.self_attn.q_proj.weight")

# If the PPL Gateway detects Catastrophic Forgetting, trigger a rollback attempt with checksum verification:
if not api.ppl_gateway_monitor(current_ppl=12.5, baseline_ppl=6.2):
    api.transactional_rollback(layer_name="model.layers.0.self_attn.q_proj.weight")
```

---

## 7. Real Weights Verification
To verify the I/O pipeline on actual production weights (Layer 2 adapters), you can run our real weights validation script. It downloads a tiny `.safetensors` model from Hugging Face, projects it through the adapters (applying Soft-Routing Head Pooling and dimensional projection), and saves the newly formatted `.safetensors` file.

```bash
python examples/verify_real_safetensors.py
```

---

## 8. Validated vLLM Smoke Checks

### Qwen load smoke
```bash
PYTHONPATH=. python scratch/test_qwen.py
```

### vLLM engine probe
```bash
PYTHONPATH=. python scratch/probe_vllm.py
```

### Prepare an evaluation-only projected Alpaca payload
```bash
PYTHONPATH=. python scripts/prepare_actual_lora_payload.py \
  --lora_id onurerkan/qwen2.5-0.5b-alpaca-lora-demo \
  --output_dir routes/actual_loras \
  --target-model Qwen/Qwen2.5-0.5B
```

### Run qualitative real-LoRA route smoke check
```bash
PYTHONPATH=. python scripts/verify_alpaca_improvement.py \
  --payload routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors \
  --output reports/qualitative_alpaca_smoke_check.json \
  --dtype float16
```

### Run Phase 5-C Route-Window Benchmark
```bash
PYTHONPATH=. python scripts/bench_vllm_scalpel_overhead.py \
  --model Qwen/Qwen2.5-0.5B \
  --payload routes/actual_loras/qwen2.5-0.5b-alpaca-lora-demo_payload.safetensors \
  --prompts 50 \
  --max-tokens 32 \
  --output reports/bench_vllm_scalpel_overhead.json
```

### Run same-prompt Native LoRA rerun
```bash
PYTHONPATH=. python scripts/bench_vllm_native_lora.py \
  --model Qwen/Qwen2.5-0.5B \
  --lora-repo onurerkan/qwen2.5-0.5b-alpaca-lora-demo \
  --prompts 50 \
  --max-tokens 32 \
  --output reports/bench_vllm_native_lora.json
```

### Interpretation
* **`swap_count > 0`**: Proves successful route application.
* **`verified_rollbacks > 0`**: Proves bit-exact checksum-level rollback.
* **`exact_match=false`**: Indicates that while weights are identical, text-level output determinism was not established in that specific run due to vLLM non-determinism.