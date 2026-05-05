# Empirical Benchmark: Localized Subspace Alignment & Downstream Subset

This log was generated via the `run_migration_diagnostics.py` evaluation suite.

## 1. Environment Setup
- **Target Architecture:** Qwen/Qwen2.5-0.5B-Instruct
- **Source Adapter:** LLaMA-3-8B Coding LoRA
- **Hardware:** Local CUDA (RTX 5060 Ti)
- **Precision:** FP16
- **Calibration:** 64 forward passes (WikiText subset)

## 2. Benchmark Results

### A. Language Modeling Stability (Perplexity)
Evaluated on a 4000-token local technical corpus.
- **Base Model PPL:** 12.34
- **Transplanted Model PPL:** 12.40
- **PPL Degradation:** +0.06%
*Status: The projection did not destructively interfere with the base model's grammatical syntax.*

### B. Semantic Logic Drift (KL Divergence)
Evaluated on localized prompt sequence.
- **KL Divergence (Base vs Transplanted):** 0.018
- **Status:** ✅ Semantic logic bounds maintained. The mathematical approximation holds within the tangent space.

### C. Downstream Task Subset (HumanEval)
Evaluated on a restricted subset (N=100) to isolate coding capability transfer.
- **Source Base:** 22.0%
- **Source + LoRA:** 35.0%
- **Target Base:** 20.0%
- **Target + Naive Projection:** 14.0%
- **Target + Random Projection:** 8.0%
- **Target + Projected (Neural-Scalpel):** 27.0%
*Status: Provides preliminary evidence of partial coding-behavior retention (+7.0% over target base).*

### D. Calibration Size Ablation
Measured the impact of the number of forward passes on projection stability.
- **0 passes (Synthetic):** PPL +1082.16% (Catastrophic collapse)
- **8 passes:** PPL +15.40%
- **32 passes:** PPL +0.45%
- **64 passes:** PPL +0.06% (Optimal)

*Note: Broader downstream validation across full benchmark sets, additional LoRA types, and additional model pairs remains future work. These subset results are separate from the vLLM Phase 4-B route-serving task evaluation, which remains pending.*