# Neural-Scalpel Usage Guide

> **Note:** This usage guide covers the research CLI and Python API. Some advanced examples are roadmap-oriented or experimental. For the latest validated runtime status, see `docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md`.

Welcome to the **Neural-Scalpel** documentation. This toolkit provides experimental methods to approximate and project learned weight deltas (Task Vectors / LoRAs) from one neural architecture to another.

---

## 1. Installation

Neural-Scalpel requires Python 3.9+ and PyTorch 2.0+.

```bash
git clone https://github.com/ponpoke/Neural-Scalpel.git
cd Neural-Scalpel
pip install -e .[cli,experimental]
```

---

## 2. Adapter Transfer Diagnostic (v2.0.1)

The primary interface for Neural-Scalpel is the multi-stage diagnostic pipeline. This evaluates the portability, quality, and structural risk of migrating a LoRA between architectures.

```bash
# Using the official CLI
neural-scalpel diagnose-adapter \
  --source-base Qwen/Qwen2.5-Coder-7B-Instruct \
  --source-adapter jk200201/qwen2.5-coder-7b-sql-dpo \
  --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --benchmark sql_50 \
  --output-dir reports/diagnostics/qwen_coder_dpo_to_05b
```

> [!NOTE]
> `diagnose-adapter` currently runs source quality evaluation on CPU for stability. For large 7B+ models, this may be slow. GPU execution support is planned for a future release.

### Understanding Diagnostic Verdicts

The diagnostic pipeline issues a final verdict in `diagnostic_report.json`:

| Verdict | Meaning | Recommended Action |
|---|---|---|
| **`PROJECTION_CANDIDATE`** | Source is high-quality and architecture is compatible. | Proceed to structural projection and target evaluation. |
| **`RELEASE_READY`** | End-to-end success confirmed on the target benchmark. | Publish as a validated research artifact; deployment still requires environment-specific validation. |
| **`SOURCE_READY`** | Source is good, but target compatibility is unknown or failed. | Check architecture mapping/GQA settings. |
| **`RESEARCH_ONLY`** | High regression rate or unstable delta detected. | Do not use for production; analyze failure modes. |

> [!IMPORTANT]
> `PROJECTION_CANDIDATE` is not a release verdict. It means the adapter is eligible for projection and target-side evaluation. `RELEASE_READY` requires positive target benchmark results.

---

## 3. LoRA Projection (Experimental Wrapper)

For users who have reviewed the diagnostic report and wish to perform the actual projection, use the `project-adapter` command. This is currently an experimental wrapper for the Structural Projection engine.

```bash
neural-scalpel project-adapter \
    --source-base Qwen/Qwen2.5-Coder-7B-Instruct \
    --source-adapter jk200201/qwen2.5-coder-7b-sql-dpo \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --rank 16 \
    --alpha 16 \
    --output ./qwen25-05b-sql-projected
```

---

## 4. Target Evaluation Gate (v2.1)

Once projected, evaluate the adapter on the target model. Use the `--report` option to integrate results and finalize the release decision.

```bash
neural-scalpel evaluate-projected \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --adapter ./qwen25-05b-sql-projected \
    --benchmark sql_50 \
    --report reports/diagnostics/qwen_coder_dpo_to_05b/diagnostic_report.json \
    --positive-delta-threshold 0.0 \
    --max-regression-rate 0.05 \
    --output reports/target_eval/sql_results.json
```

---

## 5. The Python API (Core Math Engine)

If you are a researcher or want fine-grained control over the math, you can bypass the CLI and use the `neural_scalpel.core.math` package directly. 

```python
import torch
from neural_scalpel.core.math import (
    adaptive_variance_preserving_sparsity,
    adaptive_rsvd_bootstrap,
    head_wise_orthogonal_procrustes
)

# 1. Extract the sparse knowledge core (AVPS preserves 99% variance)
W_tuned = torch.load("tuned_weights.pt")
W_base = torch.load("base_weights.pt")

tau_sparse = adaptive_variance_preserving_sparsity(W_tuned, W_base, variance_preservation=0.99)
tau_dense = tau_sparse.to_dense()

# 2. Extract Low-Rank representations (rSVD)
U, S, V = adaptive_rsvd_bootstrap(tau_dense, epsilon=1e-2)

# 3. Align architectures using semantic anchors
A_aligned, _, R_matrices, s_factors = head_wise_orthogonal_procrustes(
    A=A_anchor, B=B_anchor, num_heads=28
)
```

---

## 6. Semantic Routers (`.scalpel_route`)

In the current research preview, `.scalpel_route` files are used for signed route manifests and evaluation payload metadata.

### Loading a Route programmatically (With Chain of Trust):
```python
from neural_scalpel.router.manager import ScalpelRouteManager

manager = ScalpelRouteManager(route_dir="./routes")

matrices = manager.verify_and_load_route(
    filepath="./routes/Meta-Llama-3-8B-to-Qwen2-7B-coding.scalpel_route",
    current_source_id="meta-llama/Meta-Llama-3-8B",
    current_target_id="Qwen/Qwen2-7B",
    trusted_keys=["my_enterprise_secret_key", "official_provider_key"]
)
```

---

## 7. Experimental: VRAM Hot-Swap

For advanced enterprise use cases, the `experimental.hot_swap` module provides threading locks and Perplexity (PPL) guardrails.

---

## 8. Real Weights Verification
To verify the I/O pipeline on actual production weights (Layer 2 adapters):

```bash
python examples/verify_real_safetensors.py
```