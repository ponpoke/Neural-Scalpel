# Neural-Scalpel Usage Guide

Welcome to the **Neural-Scalpel Ecosystem** documentation. Neural-Scalpel allows you to extract concepts (intelligence) learned by one neural architecture and transplant them into an entirely different architecture purely via mathematical projection (without training data or gradient descent).

This guide will walk you through the core concepts, from the CLI Auto-Wrapper to the programmatic Python API.

---

## 1. Installation

Neural-Scalpel requires Python 3.9+ and PyTorch 2.0+.

To install the framework locally for development and testing:

```bash
git clone https://github.com/ponpoke/Neural-Scalpel.git
cd Neural-Scalpel
pip install -e .[cli,experimental]
```

---

## 2. The Command Line Interface (CLI)

For most developers and MLOps engineers, the CLI is the fastest way to project a LoRA fine-tune from one model to another. The CLI utilizes the **Layer 2 Auto-Wrapper** to automatically detect model architectures and orchestrate the tensor conversions.

### Example: Porting a Llama-3 LoRA to Qwen-2

Suppose you have an Unsloth LoRA fine-tuned on Llama-3 (`./my-llama3-lora`), but you want to deploy it on Qwen-2 (`Qwen/Qwen2-7B`).

```bash
neural-scalpel port \
    --source ./my-llama3-lora \
    --target Qwen/Qwen2-7B \
    --domain general \
    --output ./qwen2-ported-lora
```

**What happens under the hood:**
1. The CLI reads the `config.json` of both Llama-3 and Qwen-2.
2. It detects the architectural mismatch (e.g., Llama-3 has 32 heads, Qwen-2 has 28).
3. It loads the `adapter_model.safetensors` from your source directory.
4. It passes the tensors to the **Layer 1 Math Engine** to structurally align and project the dimensions using Head-wise Orthogonal Procrustes and PCA Subspace Injection.
5. It writes the projected `.safetensors` to the `--output` directory, ready to be loaded via `peft.PeftModel.from_pretrained()`.

---

## 3. The Python API (Core Math Engine)

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

## 4. Semantic Routers (`.scalpel_route`)

To prevent catastrophic forgetting and improve domain accuracy, the Neural-Scalpel ecosystem uses **Domain-Specific Router Hubs** (Layer 3).

Instead of computing the semantic alignment dynamically (which requires representative prompt data), you can use pre-calculated `.scalpel_route` files. These files contain highly optimized Rotation ($R$) and Scaling ($s$) matrices calculated over massive domain-specific datasets (e.g., medical, coding).

### Generating a Route via CLI:
You can automatically generate a `.scalpel_route` file with strict SHA-256 validation:
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

## 5. Experimental: VRAM Hot-Swap

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
Shadow Registering (Double Buffering) eliminates thread deadlocks and allow for 100% strict transactional rollbacks.

```python
from neural_scalpel.experimental.hot_swap import VRAMHotSwapAPI

api = VRAMHotSwapAPI(target_model=my_live_pytorch_model)

# Inject using a Shadow Buffer (Microsecond lock duration)
api.inject_concept_shadow(task_vector=my_concept_tensor, layer_name="model.layers.0.self_attn.q_proj.weight")

# If the PPL Gateway detects Catastrophic Forgetting, trigger a 100% perfect rollback:
if not api.ppl_gateway_monitor(current_ppl=12.5, baseline_ppl=6.2):
    api.transactional_rollback(layer_name="model.layers.0.self_attn.q_proj.weight")
```

---

## 6. Real Weights Verification
To verify the I/O pipeline on actual production weights (Layer 2 adapters), you can run our real weights validation script. It downloads a tiny `.safetensors` model from Hugging Face, projects it through the adapters (applying Soft-Routing Head Pooling and dimensional projection), and saves the newly formatted `.safetensors` file.

```bash
python examples/verify_real_safetensors.py
```