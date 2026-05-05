# Neural-Scalpel

**No-Retraining LoRA Migration & Diagnostic Toolkit**

[![Version](https://img.shields.io/badge/version-1.0.0--alpha-orange)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-153%20passed-brightgreen)](tests/TEST_REPORT.md)
[![Verification](https://img.shields.io/badge/Status-Prototype--Validated-blue)](docs/LOGIC_CONSISTENCY_REPORT.md)

Neural-Scalpel is an experimental no-retraining LoRA migration toolkit for projecting learned adapter weights (Task Vectors / LoRAs) across partially compatible neural architectures.

It does not guarantee universal adapter conversion. Instead, it combines mathematical task-vector projection with diagnostic gates that evaluate whether a migrated adapter is stable, risky, or unsuitable for deployment.

In short: Neural-Scalpel attempts no-retraining adapter migration, then tells you whether the result is safe enough to trust.

> **⚠️ RESEARCH DISCLAIMER**
> Neural-Scalpel performs no gradient-based retraining, but it is not data-free.
> LLM projections require calibration activations to preserve emergent outlier dimensions.
> 
> This framework does not guarantee universal "intelligence transfer."
> Successful migration depends on architectural homology, calibration quality, and downstream validation.

---

## Why It Matters

Modern teams often accumulate LoRA assets tied to older base models. When refreshing to newer, cheaper, faster, or more capable models, those adapters often become stranded.

Neural-Scalpel helps answer:

- Can this LoRA be migrated without immediate retraining?
- How much language-modeling stability is lost after projection?
- Is the result better than naive padding or random projection?
- Should we port, retrain, or discard this adapter?
- What risks block production deployment?

*Use Neural-Scalpel when you want to test whether an existing LoRA can survive a base-model refresh without immediate retraining.*

---

## Project Maturity

Neural-Scalpel has completed a production-readiness evaluation prototype.
It is an alpha-stage research project.

### Stable / Verified
- **Portability Diagnostics:** Signed route manifests, payload verification, PPL/KL evaluations.
- **Experimental Hot-Swap Runtime:** PyTorch-native atomic injection/rollback.
- **External vLLM Proxy Validation:** Strict route isolation and homogeneous batching enforcement validated against a live vLLM backend in controlled tests.
- **Real-Model Endurance:** 16K Qwen2.5-0.5B PyTorch-native Hot-Swap requests with zero route leakage and zero rollback failures.
- **Actual LoRA Evaluation:** Preserved exact-match coding performance while injecting domain-specific (Text-to-SQL / Alpaca) distributions via safetensors payloads.
- **Internal vLLM Mock Design:** Architecture designed for `RouteAwareScheduler` and `RouteTaggedKVBlock`.

### Roadmap / Future Work
- Native vLLM plugin source code patching
- Internal Scheduler / KV Cache integration in a live vLLM environment
- Long-running multi-tenant production pilots
- GGUF/AWQ direct surgery

### ⚠️ Not Production Ready
- **Internal vLLM Plugin:** Phase 0-6 monkey patch implementation is complete, but live Linux/vLLM validation (Phase 7+) is pending. It is not yet proven safe under continuous batching in a real engine.
- **SLA-Grade Serving:** This is a research prototype. It should not be used as a multi-tenant SLA proxy in an enterprise environment without further hardening.

---

## Recommended Workflow

1. Run `diagnose` to estimate whether no-retraining migration is feasible.
2. Review the generated portability report.
3. Run `diagnose --ablation all` if structural confidence is required.
4. Run `port` only if the adapter passes diagnostic gates.
5. Run downstream task validation before production use.

Recommended principle:
> Diagnose first. Port second. Deploy only after downstream validation.

---

## 1. What This Toolkit Can and Cannot Do

### ✅ Verified / Supported Capabilities
* **No-Retraining Adapter Migration:** Attempts to project LoRA / task-vector deltas into a target architecture without gradient-based fine-tuning.
* **Portability Diagnostics:** Generates feasibility reports using PPL, KL divergence, calibration coverage, architecture homology, and adapter drift metrics.
* **Mathematical Subspace Alignment:** Successfully maps attention heads and MLP layers between architectures with low Procrustes error in localized tests.
* **Qualitative Style Shifts (Vision):** Can project styling LoRAs (e.g., Animagine XL) from an SDXL base to another structurally compatible model, resulting in observable stylistic changes without retraining.
* **Memory-Efficient Processing:** Streams multi-gigabyte delta computations layer-by-layer, keeping VRAM usage under 16GB.

### ❌ Known Limitations & Failure Modes
*   **Zero-Dataset Collapse:** Projecting LLM adapters without a representative calibration dataset destroys massive emergent outliers, causing the model to output gibberish. **Gradient-free does not mean data-free.**
*   **OOD Approximation Failure:** JTSA and HAMA rely on Taylor approximations. Extreme Out-Of-Distribution (OOD) prompts will break the structural alignment, leading to hallucination or logical collapse.
*   **No Empirical Task Guarantee:** A mathematically perfect projection does not guarantee that the target model can execute the complex logic (e.g., math, coding) learned by the source model.

---

## 2. Evaluation Metrics

We strictly divide our evaluation to avoid conflating mathematical success with downstream model capability.

| Category | Metric | Status / Result |
| :--- | :--- | :--- |
| **Structural Metric** | Procrustes Relative Error | $1.3392 \times 10^{-6}$ localized hidden-state alignment |
| **LM Metric** | PPL Degradation | +0.06% on a localized 4,000-token calibration/eval set |
| **Ablation Metric** | `diagnose --ablation all` | Calibrated JTSA+WDR outperforms naive/random/procrustes-only baselines |
| **Downstream Metric** | HumanEval subset | 27.0% pass@1 on N=100 small subset; full HumanEval/GSM8K pending |

*For full details on our testing methodology and failure modes, read the [Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md).*

---

## 3. Qualitative Visual Demo

This visual A/B test demonstrates a successful weight-delta projection from an SDXL Anime LoRA onto a standard SDXL base model without retraining.

| Baseline (Vanilla SDXL Base) | Projected (SDXL Base + Converted LoRA) |
| :---: | :---: |
| ![Before](verification_demo/assets/sdxl_standard.png) | ![After](verification_demo/assets/sdxl_transplanted.png) |
| *Identical Prompt & Seed (6000).* | *Identical Prompt & Seed (6000). The converted adapter successfully forces the anime aesthetic.* |

*Note: This visual demo demonstrates adapter projection behavior in an SDXL-compatible setting. Full SDXL-to-FLUX visual validation remains highly experimental and is tracked in the compatibility matrix.*

---

## 4. Core Algorithms

1. **Hard-WDR (Wasserstein Discrete Routing):** Attempts to map specialized Attention Heads via Sinkhorn-Knopp optimization.
2. **JTSA & HAMA (Hessian-Aware Manifold Alignment):** Pre-compensates for non-linear GeGLU/SwiGLU distortions using Taylor approximations across a **calibrated activation manifold**.
3. **Adaptive Variance-Preserving Sparsity (AVPS):** Truncates noise in the Task Vector prior to SVD, preserving 99% of L2 energy to enable calculation on consumer GPUs.
4. **Synchronized Tensor Swapping:** Experimental PyTorch-level pointer swapping for runtime adapter injection, using `torch.cuda.synchronize()` to implement basic rollback semantics (Note: single-digit to low-double-digit millisecond swap overhead in controlled tests).

---

## 5. Compatibility Matrix

To ensure realistic expectations, Neural-Scalpel maintains a strict compatibility matrix.

| Source Adapter | Target Model | Status | Evidence | Recommended Use |
|---|---|---|---|---|
| LLaMA-3 LoRA | Qwen2.5-0.5B | Experimental | PPL/KL localized | Research only |
| SDXL LoRA | SDXL-compatible | Qualitative pass | Visual A/B | Style testing |
| SDXL LoRA | FLUX | Highly experimental | Limited qualitative | Not for production |
| GGUF/AWQ | Any | Roadmap | Not implemented | N/A |
| Text LoRA | Vision Model | Unsupported | Cross-modality failure | Do not use |

## 6. Experimental Hot-Swap Runtime

Neural-Scalpel includes an experimental PyTorch-native Hot-Swap Runtime for testing signed `.scalpel_route` injection with rollback, audit logging, tenant isolation, and route-aware serving.

**Current validation:**
- Signed route registry with HMAC-SHA256 verification
- Fail-closed policy gates (tenant, license, revocation, quarantine)
- Checksum rollback verification (bit-exact restoration)
- Structured JSON-L audit logging (100% event coverage)
- External FastAPI proxy with strict temporal route isolation
- Live stress-tested with `Qwen2.5-0.5B` and actual Text-to-SQL / Alpaca LoRAs
- Internal vLLM integration architecture designed and mocked

**⚠️ What is NOT Production Ready:**
- **Full vLLM Internal Plugin:** The architecture is validated via mocks, but actual vLLM continuous batching source code has not been patched.
- **SLA-Grade Serving:** Not ready for uncontrolled public enterprise traffic.
- **1-GPU Multi-Tenant Scale:** Cannot yet serve hundreds of concurrent routes seamlessly without internal KV cache integration.

*For full details on what works and what doesn't, see the [Production Readiness Report](docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md).*

---

## 7. Commercialization Roadmap

The core research framework and diagnostic CLI are licensed under the Apache License 2.0.

Future commercial offerings may focus on:
- No-retraining LoRA migration feasibility reports
- Adapter asset reuse during model refresh cycles
- Batch portability scoring for large LoRA libraries
- Private/on-premise migration diagnostics
- License risk checks and audit-ready reports
- Safe-mode adapter generation and rollback strategies

No production-grade guarantees are currently provided in v1.0.0-alpha.

---

## 8. Quick Start

```bash
# 1. Run a capability transfer evaluation (e.g., Text-to-SQL)
python tests/bench_text_to_sql.py

# 2. Run the Hot-Swap Runtime external proxy with vLLM
VLLM_BACKEND_URL="http://localhost:8000/v1/completions" uvicorn neural_scalpel.serving.vllm_proxy:app --port 8080
```

---

## 9. Documentation & Reports
- **[Hot-Swap Runtime Production Readiness Report](docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md):** 🚀 *Read this first!* Contains the endurance test results, actual LoRA evaluations, and integration benchmarks.
- **[vLLM Internal Integration Design](docs/VLLM_INTERNAL_INTEGRATION_DESIGN.md):** Architectural design for Step 4B integration.
- **[Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md):** Details on mathematical evaluation metrics and failure modes.
- **[Project Vision & Roadmap](docs/RESEARCH_AND_COMMERCIAL_ROADMAP.md):** Our strategy for ML research validation and commercial diagnostic tools.
- **[Technical Report](TECHNICAL_REPORT.md):** Mathematical proofs and architecture overview.
- **[Security Policy](SECURITY.md):** Important security considerations and vulnerability reporting.
- **[Model License Policy](MODEL_LICENSE_POLICY.md):** Legal and licensing responsibility regarding derivative adapter works.
- **[Disclaimer](DISCLAIMER.md):** Experimental software disclaimer and lack of production guarantees.

---
*Developed and tested locally on an NVIDIA RTX 5060 Ti 16GB.*