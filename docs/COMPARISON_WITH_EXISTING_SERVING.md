# Comparison With Existing Serving Methods

## Status

**Evaluated: Neural-Scalpel vs. Native LoRA vs. Model Reload vs. Multi-Instance (VRAM).**

Measured:
- Neural-Scalpel (v2) swap latency p50: 79.96 ms (one-time per window)
- Neural-Scalpel (v2) rollback latency p50: 9.64 ms (one-time per window)
- vLLM Native LoRA throughput (single adapter, Qwen2.5-0.5B rerun): ~663 tok/s
- Neural-Scalpel (v2) throughput (single adapter, Qwen2.5-0.5B): 4086.68 tok/s
- Model reload latency p50: 7.55 s
- Model reload latency p90: 8.72 s

> [!NOTE]
> The Phase 5-C benchmark showed Neural-Scalpel v2 outperforming Native LoRA in throughput under controlled same-prompt conditions. The positive delta over base in that run (+69.98%) should be interpreted carefully as it was influenced by repetitive output dynamics. The primary result is the removal of per-token overhead.

Pending:
- Detailed multi-adapter mixed-batch Native LoRA comparison
- Real multi-process multi-instance throughput measurements
- Cross-method comparison under high-concurrency production workloads

This document currently combines measured Neural-Scalpel latency with architectural comparison of existing serving patterns.

## Overview

Serving multiple domain-specific behaviors (e.g., Code, SQL, Chat, Safety) efficiently is a core challenge in modern LLM deployments. When adapters (LoRAs) are available, operators typically choose between three legacy strategies. Neural-Scalpel introduces a fourth: **Route-Aware Weight Surgery**.

### 1. Neural-Scalpel (Route-Aware Hot-Swap)
Neural-Scalpel intercepts the forward pass within the serving engine (e.g., vLLM), temporarily injecting specialized adapter weights into a single resident base model, and verifies restoration through explicit rollback checks immediately after execution.

### 2. vLLM Native LoRA
The inference engine holds the base weights in GPU memory and maintains multiple small PEFT adapters in VRAM. During the forward pass, the engine mathematically routes inputs through the corresponding adapters simultaneously.

### 3. Model Reloading (Per Route)
A single GPU worker unloads the current model from VRAM and loads the specific full-parameter fine-tune or merged model required for the incoming request.

### 4. Multi-Instance Serving
Deploying multiple dedicated GPU workers, each holding a distinct specialized model or merged LoRA in memory, fronted by an API gateway router.

---

## Direct Benchmark Results

Measured on the same RTX 5060 Ti / vLLM 0.20.1 environment.

### Neural-Scalpel (Latency Evolution)

**Earlier precise-latency microbenchmark (Local Mock):**
- Swap latency p50: 0.59 ms
- Rollback latency p50: 2.19 ms
- Combined swap + rollback p50: **~2.78 ms**

**Phase 5-C real projected Alpaca payload route-window benchmark (vLLM):**
- Swap latency p50: **79.96 ms** (one-time cost per window)
- Rollback latency p50: **9.64 ms** (one-time cost per window)

### Model Reload

Measured as process-level vLLM reinitialization using a fresh Python subprocess for each reload iteration (using `facebook/opt-125m`):

- Reload latency p50: **7.55 sec**
- Reload latency p90: **8.72 sec**
- Reload latency avg: **8.03 sec**
- Warm generation latency p50: 0.44 sec
- Warm generation latency p90: 0.48 sec

Model reload route switching is approximately 2,716x higher latency than the earlier Neural-Scalpel microbenchmark swap+rollback path (~2.78ms), and approximately 84x higher than the Phase 5-C real-payload route-window swap+rollback cost (~89.6ms).

#### Phase 5-C Rerun (Same-Prompt Benchmark)
- **Base Model:** `Qwen/Qwen2.5-0.5B`
- **LoRA Adapter:** `onurerkan/qwen2.5-0.5b-alpaca-lora-demo`
- **Native LoRA Rerun Throughput:** **~663 tok/s**
- **Neural-Scalpel v2 Throughput:** **4086.68 tok/s**
- **Neural-Scalpel v2 Delta vs. Base:** **+69.98%**

> [!IMPORTANT]
> In this same-prompt, route-homogeneous benchmark, Neural-Scalpel v2 showed significantly higher measured throughput than Native LoRA. However, the positive delta over base should not be generalized as universal speed superiority; routed output in this run was highly repetitive. The core technical validation is that **route-window swapping reduced swap frequency to 0.000625 swaps/token**, effectively removing the Phase 5-B bottleneck.

### Phase 5-D Repeated Median Benchmark

| Method | Median Throughput | Notes |
|---|---:|---|
| Base | ~3813.84 tok/s | 50 prompts × 3 runs |
| Native LoRA | ~983.32 tok/s | Same prompt set / controlled rerun |
| Neural-Scalpel v2 | ~2574.31 tok/s | Route application and verified rollback enforced |

Neural-Scalpel v2 outperformed Native LoRA by +161.80% under this controlled 50-prompt repeated-median benchmark. This supports generality beyond the single-prompt Phase 5-C run, but should not be generalized to all models, adapters, concurrency levels, or prompt distributions.


### Multi-Instance VRAM Scaling

Measured single-instance vLLM memory footprint for `facebook/opt-125m`:

- Single-instance VRAM reserved: 15,166.0 MB
- Single-instance VRAM allocated: 14,772.8 MB

Estimated 3-instance footprint assuming linear scaling:

- 3-instance VRAM reserved: **45,498.0 MB**
- 3-instance VRAM allocated: **44,318.5 MB**

This estimate indicates that multi-instance serving would exceed the capacity of a 16GB-class GPU even for a small OPT-125M vLLM configuration under this setup.

> [!NOTE]
> Multi-instance values are estimated from measured single-instance VRAM. Actual multi-process throughput measurements remain pending.

---

## Quantitative & Architectural Trade-offs

| Metric | Neural-Scalpel (v2) | vLLM Native LoRA | Model Reload | Multi-Instance |
| ------ | -------------- | ---------------- | ------------ | -------------- |
| **VRAM Scaling** | O(1) + deltas | O(1) + adapters | O(1) per active worker | O(N); estimated 45.5GB reserved for 3 instances |
| **Route Switch Latency** | **~80 ms** per window | Near-zero per-request | 7.55s p50 / 8.72s p90 measured | 0 ms model switch |
| **Throughput (latest)** | **4086.68 tok/s** | **~663 tok/s** (rerun) | Poor | Pending |
| **Route Isolation** | Checksum-verified rollback | Adapter-level kernels | Process-level | Process/worker-level |
| **Architecture Limit** | Raw layer deltas | vLLM-supported LoRA | Any loadable model | Any deployed model |

### Deep Dive Analysis

#### A. VRAM Efficiency vs. Hardware Costs
- **Multi-Instance** scaling can be financially prohibitive as GPU count increases. Serving many specialized models through dedicated instances scales GPU memory roughly linearly with the number of resident model instances, unless additional quantization, offloading, or model-sharing techniques are used. In this benchmark, three instances of even a tiny model (OPT-125M) were estimated to exceed 45GB of VRAM.

#### B. The Latency Cost of Isolation
- **Model Reloading** ensures absolute isolation but incurs a seconds-scale reload penalty; in this environment, process-level vLLM reinitialization measured 7.55s p50 and 8.72s p90, ruining interactive latency (TTFT). 
- **vLLM Native LoRA** provides a highly optimized adapter execution path. Earlier refined single-adapter Qwen2.5-0.5B testing measured 1968.25 tok/s versus a 3501.4 tok/s baseline (-43.79%). In the later same-prompt Phase 5-C rerun used for direct comparison, Native LoRA measured ~663 tok/s. These values should not be mixed; the Phase 5-C rerun is the relevant comparison for the route-window benchmark.
- **Neural-Scalpel** applies raw deltas before the optimized base forward path and then rolls back. Phase 5-C successfully removed the performance-prohibitive per-token overhead observed in v1. It incurs an **~80ms** swap latency penalty (p50) once per route window to achieve strict isolation and verification.

#### C. The Security and Contamination Factor
- In **Native LoRA**, adapters share the same base VRAM execution context. It does not provide Neural-Scalpel-style explicit base-weight rollback verification, because adapters are applied through the engine's optimized adapter execution path rather than by temporarily mutating and restoring base weights.
- **Neural-Scalpel** treats weights as stateful resources. By taking a checksum before injection and validating the rollback, it verifies that the base weights are restored before processing the next route batch, reducing the risk of cross-route state contamination.

---

## Decision Matrix: When to use what?

### 🟢 Use Neural-Scalpel When:
1. You have a large library of specialized behaviors (SQL, Code, Domain-Chat) and cannot afford VRAM for Multi-Instance serving.
2. You require **strict route isolation and explicit verification** (the rollback mechanism provides checksum-based validation between requests).
3. Your weight deltas are generated through experimental methods (e.g., Task Vectors, Orthogonal Projection, Non-Standard PEFT) that are unsupported by Native LoRA kernels.

### 🟡 Use vLLM Native LoRA When:
1. Peak throughput and maximum concurrency are your sole priorities.
2. You only have a few standard Hugging Face PEFT LoRAs.
3. You do not require strict memory-checksum isolation between tenant executions.

### 🔴 Use Multi-Instance Serving When:
1. You are serving completely different base architectures (e.g., Llama-3 and Qwen2.5) simultaneously.
2. Hardware budget is unlimited and latency/throughput SLAs are absolutely critical.

### ❌ Avoid Model Reloading Unless:
1. You are operating an offline batch-processing pipeline where latency is irrelevant.
 (e.g., Task Vectors, Orthogonal Projection, Non-Standard PEFT) that are unsupported by Native LoRA kernels.

### 🟡 Use vLLM Native LoRA When:
1. Peak throughput and maximum concurrency are your sole priorities.
2. You only have a few standard Hugging Face PEFT LoRAs.
3. You do not require strict memory-checksum isolation between tenant executions.

### 🔴 Use Multi-Instance Serving When:
1. You are serving completely different base architectures (e.g., Llama-3 and Qwen2.5) simultaneously.
2. Hardware budget is unlimited and latency/throughput SLAs are absolutely critical.

### ❌ Avoid Model Reloading Unless:
1. You are operating an offline batch-processing pipeline where latency is irrelevant.
