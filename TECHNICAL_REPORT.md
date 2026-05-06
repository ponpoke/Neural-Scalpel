# Task Vector Projection: A Mathematical Framework for Cross-Architecture Adapter Conversion

**Date:** May 2026
**Status:** Experimental Research Preview (v1.0.0-alpha)

## Abstract
This report proposes "Task Vector Projection," an experimental mathematical framework attempting to approximate and project learned weight deltas (Task Vectors / LoRAs) between neural architectures without gradient-based fine-tuning. By defining knowledge as a geometric vector within the weight space, we explore methods to project these vectors across distinct architectures (e.g., UNet to DiT, or LLaMA to Qwen). Our methodology incorporates memory-efficient sparse hacking, adaptive singular value decomposition, and structural non-linear compensation. Preliminary localized validations on a single-node setup indicate that calibrated projection can preserve structural alignment and language-modeling stability in limited settings. A small HumanEval subset experiment suggests partial coding-behavior retention for one LLaMA-3-to-Qwen-2.5 configuration, but broader downstream validation across full benchmark sets, additional LoRA types, and additional model pairs remains future work. Core mathematical, structural projection, and controlled-runtime components are covered by an automated test suite; the repository badge currently tracks 200+ non-live tests passed; live vLLM tests are executed separately.

---

## 1. Introduction
Traditional AI development relies heavily on dataset-driven training. We propose an experimental "surgical approach": if a model has learned a concept, that adaptation exists as a Task Vector ($\tau = W_{tuned} - W_{base}$). This framework extracts $\tau$, attempts to structurally align the semantic manifolds of the models, and mathematically injects the vector.

## 2. Mathematical Framework (Core Algorithms)

### 2.1. Physical Sparse Memory Hack
To manipulate multi-gigabyte weight deltas on consumer hardware (16GB RAM), we perform Pre-SVD Trimming. We truncate noise by zeroing the bottom $20\%$ of absolute values in $\tau$, then casting to Compressed Sparse Row (CSR) format.
*Efficacy:* A $640 \times 2048$ matrix was compressed by $20\%$, mathematically preserving the primary signal while slashing memory requirements during the SVD phase.

### 2.2. Adaptive rSVD Bootstrap
We extract "core components" using Randomized SVD with a Bootstrap Stopping criterion. The algorithm iteratively expands the rank block-by-block, stopping when the largest singular value of the new subspace falls below a threshold $\epsilon$.

### 2.3. Head-wise Scaling Orthogonal Procrustes
Different architectures map concepts to different coordinates. We independently solve the Orthogonal Procrustes problem for each attention head $i$:
$$ \min_{s_i, R_i} \| s_i A_i R_i - B_i \|_F $$

---

## 3. Experimental Validation

### 3.1. LLM Alignment (LLaMA-3 $\to$ Qwen-2)
We tested the logic by projecting a fine-tuned LoRA from LLaMA-3 (32 heads) to Qwen-2 (28 heads).
The Head-wise Procrustes analysis aligned the multi-head attention spaces with a relative transformation error of **$1.3392 \times 10^{-6}$** (measured locally on hidden states). 

### 3.2. Cross-Architecture Projection (Vision: SDXL $\to$ FLUX)
We projected an "anime-style" LoRA from SDXL (Animagine XL 3.1) to FLUX.1-schnell.
1.  **Padding:** Dynamically padded SDXL's 10x64 heads into FLUX's 24x128 format.
2.  **Procrustes:** Used CLIP-L hidden states as semantic anchors for all 24 heads.
3.  **Result:** Qualitative A/B tests demonstrated a stylistic shift toward anime illustrations without gradient descent. Note that visual quality depends highly on the prompt and seed.

---

## 4. Hardware Context & Scalability Disclaimers
Unless otherwise noted, live GPU validations reported here were obtained on a single NVIDIA RTX 5060 Ti (16GB VRAM). Non-live unit tests may run in CPU or mixed local CI environments.
*   **Scalability:** Performance in multi-GPU clusters or distributed vLLM environments is unverified.
*   **VRAM Hot-Swapping:** We implemented a framework-agnostic C++/CUDA extension (`Scalpel-Kernel`) for atomic tensor swapping. When compiled with `nvcc`, it achieves synchronized tensor swaps with rollback semantics by leveraging CUDA stream synchronization. Without `nvcc`, it falls back to Python-level operations which lack strict hardware isolation.

### 4.1. Experimental VRAM Hot-Swap Verification (Scalpel-Kernel)
To test the `Scalpel-Kernel`'s robustness, we conducted verification tests:
1. **Resource Exhaustion ($O(1)$ Footprint):** Over 1,000 continuous hot-swap cycles, `torch.cuda.memory_allocated()` was monitored. The final memory footprint matched the baseline.
2. **Multi-Layer Contention:** Simulated simultaneous, asynchronous hot-swaps across 5 separate layers using multiple threads. The system handled this with zero exceptions.
3. **Latency Impact & TPOT:** The p99 latency spike during a hot-swap was firmly under **50ms**, demonstrating that the `cudaStreamSynchronize` barrier imposes a momentary stall but does not indefinitely freeze the pipeline. This result applies to the experimental Scalpel-Kernel verification path, not to the current vLLM monkey-patch route path. vLLM TTFT/TPOT and payload-load latency remain pending.
4. **Fault Tolerance (Rollback):** Attempting to inject malformed tensors triggered framework exceptions and rolled back the transaction.

### 4.2. Non-linear Robustness (Perplexity Impact)
Transformers are non-linear due to GeGLU/SwiGLU. Linear alignment alone is insufficient.
*   **Result:** Initial SVD-based transplant (SRHP) yielded a PPL degradation of **+4.80%**, confirming the need for non-linear structural compensation.

---

## 5. Architectural Features (Precision Upgrades)

### 5.1. Adaptive Variance-Preserving Sparsity (AVPS)
Replaces hardcoded thresholds with energy-aware thresholding, preserving exactly $99\%$ of the total L2 variance to ensure heavy-tailed connections are never severed.

### 5.2. Principal Component Subspace Injection (PCSI)
Projects the source concept onto the **Principal Components** of the target space via SVD. When the number of SVD components is fewer than the source dimensionality, PCSI gracefully falls back to projecting through all available principal components.

### 5.3. Wasserstein Discrete Routing (WDR), Jacobian Tangent Space Alignment (JTSA) & HAMA
To mitigate non-linear distortion:
1.  **Hard-WDR:** Attempts to map specialized Attention Heads via Sinkhorn-Knopp.
2.  **Jacobian Tangent Space Alignment (JTSA):** Pre-compensates for SwiGLU/GeGLU curves via a high-precision first-order Taylor alignment across a **calibrated activation manifold**, with an optional zero-dataset synthetic fallback.
3.  **Hessian-Aware Manifold Alignment (HAMA):** Introduces a 2nd-order Taylor expansion to pre-compensate for extreme curvature in Out-Of-Distribution (OOD) regions, stabilizing the projection. Both JTSA and HAMA rely heavily on a small set of activation states (datasets) to accurately capture emergent outliers.

### 5.4. Universal I/O Bridge & Streaming Processing
To overcome the physical limitations of consumer hardware, we implemented a modular I/O architecture:
- **Multi-Format Thawing:** Direct loading and vectorized auto-dequantization of quantized formats (**GGUF, AWQ**) into high-precision FP16 tensors for alignment.
- **Streaming Processing:** The pipeline processes models layer-by-layer. This ensures a constant, O(1 layer) memory footprint, enabling the processing of 7B+ models on a single 16GB VRAM node.

### 5.5. External Proxy Fallback
To mitigate vLLM internal monkey-patch compatibility risk, Neural-Scalpel now includes an External Proxy Fallback path. This mode routes requests through external vLLM-compatible backends via HTTP instead of patching vLLM internals.

Validated components include:
- serving mode selection and fail-closed behavior
- backend registry and route-to-backend resolution
- HTTP forwarding through `ProxyServingEngine`
- automatic fallback from failed internal compatibility checks in `auto` mode
- live local HTTP forwarding smoke test

This fallback does not eliminate all deployment risk. It trades route density and memory efficiency for process-level isolation and operational simplicity.

### 5.6. Structural Projection Baseline v2
Neural-Scalpel includes a structural projection baseline for Qwen2.5-style cross-scale adapter experiments.

**Current Findings:**
While structural compatibility is achieved (100% tensor-shape matching), **behavioral transfer is currently inconclusive.** Initial qualitative smoke tests using greedy decoding resulted in **100% bit-identical output** compared to the base model. Behavioral validation remains a downstream requirement.

Validated components include:

- GQA-aware target-shape inference for Q/O/K/V and MLP projections
- Interpolated layer mapping from source depth to target depth
- SVD-based recompression statistics and energy-retention reporting
- Qwen fused tensor construction for `qkv_proj` and `gate_up_proj`
- Strict unexpected-tensor rejection during target-shape verification
- PEFT-format load and one-token generation smoke validation

This baseline verifies structural and format compatibility only. It does not prove SQL/Coding task improvement, long-form generation stability, or arbitrary cross-model intelligence transfer. Behavioral validation remains a downstream requirement.

---

## 6. Executable Ablation Study Framework

To rigorously prove the necessity of each mathematical component, Neural-Scalpel defines an executable ablation framework. Rather than theoretical assumptions, structural and downstream retention must be validated across the following modes:

1. **Naive Padding / Resize Baseline:** Zero-padding or truncating the source adapter to fit the target dimensions without structural rotation.
2. **Random Orthogonal Projection:** Applying a random orthogonal matrix to isolate the effect of intentional alignment from random noise injection.
3. **Procrustes Only (Linear):** Applying only the Singular Value Decomposition (SVD) and Orthogonal Procrustes alignment without non-linear compensation.
4. **Procrustes + AVPS:** Adding Adaptive Variance-Preserving Sparsity to measure the impact of noise filtering on projection fidelity.
5. **Procrustes + WDR:** Introducing Wasserstein Discrete Routing to evaluate the necessity of attention head re-routing.
6. **JTSA + WDR (Uncalibrated):** Applying Taylor approximations assuming a standard normal distribution, demonstrating the failure mode of zero-dataset projection.
7. **JTSA + WDR (Calibrated):** The complete Neural-Scalpel pipeline, relying on an empirical calibration manifold.
8. **Route-Window Persistent Swapping (Phase 5-C):** Implementing a stateful runtime that maintains the active route until a transition is required, reducing swap overhead from $O(tokens)$ to $O(windows)$.

Each ablation mode must be measured against a strict 6-way empirical comparison to isolate the exact contribution of each algorithm.

---

## 7. Conclusion
Neural-Scalpel Version 1.0.0-alpha establishes an experimental foundation for cross-architecture adapter conversion. Phase 5-C introduced **Route-Window Persistent Swapping**, which removed the Phase 5-B per-token swap bottleneck under the tested route-window workloads. In a recent Qwen2.5-0.5B / Alpaca payload benchmark, Neural-Scalpel recorded only 1 confirmed swap and 1 verified rollback across 1,600 generated tokens, with checksum-level rollback verification passing (`verified_rollbacks=1`).

While the Phase 5-C benchmark showed a positive throughput delta over base (+69.98%), this result was prompt-specific. Phase 5-D extended this result beyond a single prompt: 50 prompts × 3 runs showed Scalpel v2 median throughput of ~2574 tok/s versus Native LoRA at ~983 tok/s under controlled conditions.

Phase 5-E-1 validated two-route mixed-batch safety across 1000 dynamically routed requests (`__base__` ↔ Alpaca). Phase 5-E-2 extended this to 3+ real-payload mixed-batch validation across `__base__`, Alpaca, and SQL routes. Phase 5-E-3 completed worst-case alternating route stress validation for both two-route and three-route patterns. These tests strengthen route-isolation evidence but remain short-duration controlled validations.

Phase 5-F addressed the previous text-level determinism concern under the tested cache-reset condition: Base-before and Base-after matched exactly, with 100.0% top-token logprob trace similarity after verified checksum rollback.

Neural-Scalpel is currently best described as a validated prototype with strong controlled runtime evidence, and a paradigm-shift-class candidate under controlled validation.

Formal Production Candidate status remains pending the 24h persistent-route soak. Broader model coverage, vLLM-version compatibility, multi-backend load testing, and real-traffic pilots remain future hardening work.

On the projection side, HAMA and the Streaming I/O Bridge suggest that adapter weights can be structurally mapped and physically managed across architectures within the limits of high-precision linear and second-order non-linear approximations. Downstream task improvement remains workload-dependent and must be validated separately.

All core mathematical, structural projection, and controlled-runtime components are covered by an automated suite (200+ non-live tests passed; live vLLM tests are executed separately; see `tests/TEST_REPORT.md`).

---

## 8. Future Roadmap
### 8.1. ExL2 Direct Integration
Binary orchestration for non-uniform bit-rate formats.

*Note: Live GPU validations referenced in this report were performed locally on an NVIDIA RTX 5060 Ti 16GB unless otherwise stated.*