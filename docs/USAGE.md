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
# Standard Procrustes (FP16/BF16 models)
A_aligned, _, R_matrices, s_factors = head_wise_orthogonal_procrustes(
    A=A_anchor, B=B_anchor, num_heads=28
)
```

---

## 5. Semantic Routers (`.scalpel_route`)

In the current research preview, `.scalpel_route` files are used for signed route manifests and evaluation payload metadata.

### Loading a Route programmatically (With Chain of Trust):
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
```

---

## 6. Experimental: VRAM Hot-Swap

For advanced enterprise use cases, the `experimental.hot_swap` module provides threading locks and Perplexity (PPL) guardrails.

*Warning: This feature is highly experimental and currently tailored for single-GPU setups.*

---

## 7. Real Weights Verification
To verify the I/O pipeline on actual production weights (Layer 2 adapters):

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

---

## 9. Phase 5 Controlled Runtime Validation

These commands reproduce the controlled runtime validation used for the current validated prototype status.

> Note: These tests require a working vLLM environment, CUDA-capable GPU, and prepared evaluation payloads. They are not required for basic CLI usage.

### Phase 5-D: Repeated Median Benchmark

```bash
PYTHONPATH=. python scripts/run_phase_5d_median.py \
  --runs 3 \
  --prompts 50 \
  --output reports/phase_5d_repeated_median.json
```

Expected interpretation:
* Scalpel v2 median throughput is compared against Base and Native LoRA.
* `swap_count > 0` confirms route application.
* `verified_rollbacks > 0` confirms checksum-verified rollback events.

### Phase 5-E-2: 3+ route mixed-batch safety

> **Requirement:** Requires prepared Alpaca and SQL route payloads/manifests under the expected `routes/actual_loras` paths, or equivalent script arguments.

```bash
PYTHONPATH=. python scripts/run_phase_5e_3plus_mixed_batch.py \
  --requests 1000 \
  --max-tokens 16 \
  --output reports/phase_5e_3plus_mixed_batch.json
```

Expected pass criteria:
* at least three routes are requested
* `route_violations == 0`
* `quarantine_events == 0`
* `worker_is_healthy == true`

### Phase 5-E-3: Worst-case alternating route stress

Two-route alternation:
```bash
PYTHONPATH=. python scripts/run_phase_5e_alternating.py \
  --requests 1000 \
  --routes __base__,qwen2.5-0.5b-alpaca-lora-demo \
  --output reports/phase_5e_alternating_2route.json
```

---

## 10. External Proxy Fallback

External Proxy Fallback is a compatibility-risk mitigation path for environments where the internal vLLM plugin is unavailable, unsupported, or disabled.

### Serving mode selection

Use `SCALPEL_SERVING_MODE` to choose the serving path:

```bash
export SCALPEL_SERVING_MODE=internal        # use internal vLLM plugin; fail closed if incompatible
export SCALPEL_SERVING_MODE=external_proxy  # force external proxy mode
export SCALPEL_SERVING_MODE=auto            # internal if compatible, otherwise external proxy if configured
export SCALPEL_SERVING_MODE=fail_closed     # refuse to serve; reserved 'native_lora' currently fails closed here
```

Current modes:

| Mode | Status | Behavior |
| :--- | :--- | :--- |
| `internal` | controlled validation | uses internal vLLM route-window plugin |
| `external_proxy` | implemented / smoke-validated | forwards to external backend URLs |
| `auto` | implemented | falls back to external proxy when internal compatibility fails |
| `fail_closed` | implemented | refuses to start |

### Backend Registry

`BackendRegistry` only resolves `route_id -> backend_url`. It does **not** replace the main Neural-Scalpel `RouteRegistry`, which remains responsible for route existence, tenant authorization, revocation, and quarantine checks.

```python
from neural_scalpel.serving.backend_registry import BackendRegistry

registry = BackendRegistry()
registry.register_backend(
    "qwen2.5-0.5b-alpaca-lora-demo",
    "http://127.0.0.1:8001/v1/completions",
)
```

### Server integration

`create_app()` accepts an optional `ServingEngine`.

```python
from neural_scalpel.serving.server import create_app

# The engine handles the actual forwarding logic
app = create_app(
    runtime=runtime,
    registry=route_registry,
    engine=engine,
)
```

### Live proxy smoke test

```bash
PYTHONPATH=. python tests/smoke_test_proxy_forwarding.py
```

Expected result:
```text
PASS: Live proxy forwarding to local HTTP backend verified.
```

---

## 11. Interpreting Validation Reports

### What counts as strong runtime evidence?

#### Internal Plugin Evidence:
- `swap_count > 0`: route application occurred
- `rollback_count > 0`: rollback path executed
- `verified_rollbacks > 0`: checksum-verified rollback events occurred
- `route_violations == 0`: no mixed-route scheduling violation was observed
- `quarantine_events == 0`: no worker quarantine occurred
- `worker_is_healthy == true`: runtime remained healthy

#### External Proxy Evidence:
- `backend_url` resolved correctly: route-to-backend mapping worked
- HTTP request reached the expected backend endpoint and returned successfully
- Unhealthy backends were correctly tracked and rejected
- Route policy checks in `RouteRegistry` still passed before forwarding

### What these tests do not prove

These tests do not prove:
- universal adapter quality improvement
- dataset-level task improvement
- production readiness
- SLA-grade reliability
- compatibility with arbitrary vLLM versions
- multi-GPU / multi-node safety

The final constrained Production Candidate gate remains the 24h persistent-route soak.