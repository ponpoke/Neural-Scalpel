# Task Vector Projection: A Mathematical Framework for Cross-Architecture Adapter Conversion

**Date:** May 2026
**Status:** Phase 6: Full SQL Capability Evaluation (Active)
**Last Hardened:** Phase 5-G Core API (May 2026)

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

### 5.6 Structural Projection Baseline v2 (Negative Baseline)

Initial experiments using target-only statistical alignment (calibrating the target model's internal activations without reference to the source model's specific shift) resulted in a negative baseline:
- **Zero Behavioral Shift**: The projected weights failed to move the target model's output distribution.
- **Identity Result**: Greedy decoding produced 100% bit-identical outputs to the base model.
- **Conclusion**: Structural compatibility alone is insufficient for functional signal transfer. Manifold statistics must be aligned relative to the source model's behavioral shift.

Validated components of this baseline included:
- GQA-aware target-shape inference for Q/O/K/V and MLP projections
- Interpolated layer mapping from source depth to target depth
- SVD-based recompression and energy-retention reporting
- Qwen fused tensor construction for `qkv_proj` and `gate_up_proj`
- Strict unexpected-tensor rejection during target-shape verification
- PEFT-format load and one-token generation smoke validation

### 5.7 Paired Activation Behavioral Alignment

To overcome the limitations of the negative baseline, a paired source-target manifold alignment pipeline was introduced (Phase 5). This method treats intelligence transplantation as a translation problem between disparate latent spaces.

#### 5.7.1 The Pipeline
The process involves:
1. **Paired Activation Collection**: Capturing hidden states from both models on common calibration prompts.
2. **Behavioral Delta Extraction**: Capturing the specific activation shift $\Delta H_s$ caused by the source adapter.
3. **Alignment Map Learning**: Solving for a translation matrix $P$ such that $H_s P \approx H_t$ using Ridge regression.
4. **Delta Transport**: Projecting the source delta into the target space: $\Delta H_t = \Delta H_s P$.
5. **Weight Solving**: Solving for the target weight change $\Delta W_t$ via Ridge solver: $X_t \Delta W_t \approx \Delta H_t$.
6. **PEFT Export**: Compressing the full-rank solution into a low-rank (rank=16) LoRA adapter using SVD.

#### 5.7.2 Observation & Breakthrough
Unlike the target-only baseline, paired alignment produced:
- **Non-zero KL Divergence**: Measurable shifts in the target model's logit distribution.
- **Stable Behavioral Shift**: The emergence of advanced SQL structures (CTEs, Window functions) in the 0.5B target model.
- **Coherence Preservation**: Successful signal delivery without triggering catastrophic repetition collapse at calibrated scales ($\alpha=16$).

This demonstrates that while structural projection provides the "shape" of the adapter, paired manifold translation provides a measurable adapter signal that can reach the target model's generation behavior under the tested setup.

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

The paired behavioral alignment experiments suggest that structural compatibility alone is insufficient for cross-scale behavioral transfer. However, paired manifold translation combined with transported behavioral deltas produced preliminary runtime evidence of downstream behavioral modification in a 7B-to-0.5B SQL alignment experiment.

Unlike previous attempts that resulted in bit-identical outputs, this method achieved measurable logit shifts and the emergence of advanced task-specific structures (e.g., CTEs and Window functions) while maintaining generation stability.

This result remains experimental and workload-dependent, but it establishes a new research direction for **activation-space behavioral transplantation**. By aligning the manifolds of disparate models, these results suggest that higher-order behavioral structures may be partially transported across model scales without gradient-based fine-tuning, but task-level correctness remains unverified. Future work will focus on scaling these results to larger benchmarks and verifying execution accuracy across a broader range of architectures.

---

## 8. Future Roadmap
### 8.1. ExL2 Direct Integration
Binary orchestration for non-uniform bit-rate formats.

---

## 9. Recent Progress (May 2026 Update)

### 9.1 Phase 5-G: Core API Hardening
The experimental behavioral alignment scaffold has been promoted to a robust Core API.
- **Validation Standard:** Introduced a unified `ValidationReport` with status enums (`PASS`, `WARNING`, `FAIL`) and severity-based gate tracking.
- **Numerical Guards:** Implemented mandatory NaN/Inf detection in both activation collection and Ridge solving stages.
- **Flexible Mapping:** The system now supports explicit `module_to_delta_layer` mapping, allowing for non-trivial layer correspondences between heterogeneous architectures.
- **PEFT Abstraction:** LoRA export now supports custom key styles and adapter names, facilitating integration with diverse runtimes.

#### 9.2.3 Alpha Sweep and Scale Sensitivity
A systematic alpha sweep was conducted over `alpha={8, 16, 24, 32}` to map the scale sensitivity of the projected adapter.

| Setting | Accuracy | Delta | Execution Success | Delta | Syntax Valid |
|---|---:|---:|---:|---:|---:|
| Baseline | 32.0% | - | 38.0% | - | 37/50 |
| alpha=8 | 34.0% | +2.0% | 42.0% | +4.0% | 39/50 |
| **alpha=16** | **36.0%** | **+4.0%** | 44.0% | +6.0% | 40/50 |
| alpha=24 | 36.0% | +4.0% | 44.0% | +6.0% | 40/50 |
| alpha=32 | 34.0% | +2.0% | 46.0% | +8.0% | 41/50 |

**Observations:**
- **Systematic Response:** The model showed a clear response to adapter scaling. Execution accuracy peaked at `alpha=16–24`, while execution success continued to rise up to `alpha=32`.
- **Signal Saturation:** At `alpha=32`, we observed a decline in exact accuracy despite higher execution success. This indicates "over-steering," where the adapter forces the model into valid SQL syntax but occasionally compromises logical correctness.
- **Robustness:** At the balanced `alpha=16` setting, the adapter fixed 2 baseline failures and introduced **no observed regressions** against previously correct cases in the SQL-50 suite.

#### 9.2.4 Structural Projection vs Behavioral Alignment
A direct comparison was conducted between Structural Projection and Behavioral Alignment on the Qwen2.5 7B → 0.5B SQL-50 benchmark.

| Method | Accuracy | Delta | Exec Success | Delta | Syntax Valid |
|---|---:|---:|---:|---:|---:|
| Baseline 0.5B | 32.0% | - | 38.0% | - | 37/50 |
| **Structural Projection alpha=16** | **36.0%** | **+4.0%** | **44.0%** | **+6.0%** | **40/50** |
| Behavioral Alignment calibrated | 32.0% | +0.0% | 38.0% | +0.0% | 37/50 |
| Behavioral Alignment standard | 0.0% | -32.0% | 0.0% | -38.0% | 0/50 |

**Analytical Breakdown:**
- **Robustness:** Structural Projection acted as a conservative task-vector compression method, preserving the target model's native manifold while injecting a bounded source-derived weight delta.
- **Instability of Alignment:** Standard Behavioral Alignment (activation matching) collapsed in this extreme cross-scale setting. Even with calibration, it failed to improve task performance, indicating that simple activation imitation may be insufficient when representational capacities differ significantly.
- **Recommendation:** Structural Projection is the current recommended baseline for Qwen2.5 SQL adapter migration.

*Note: Live GPU validations referenced in this report were performed locally on an NVIDIA RTX 5060 Ti 16GB unless otherwise stated.*