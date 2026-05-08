# Technical Report: Neural-Scalpel

## 1. Overview
Neural-Scalpel is a framework for cross-architecture intelligence transplantation. It extracts learned weight deltas (Task Vectors) from a source model and projects them onto a target architecture using a combination of Manifold Alignment and Structural Projection.

## 2. Core Engines

### 2.1 Structural Projection
Structural projection handles the geometric transformation of weight matrices.
- **AVPS (Adaptive Variance Preserving Sparsity)**: Identifies the high-variance sparse core of a weight delta.
- **rSVD Extraction**: Compresses the delta into a low-rank representation suitable for PEFT architectures.

### 2.2 Manifold Alignment
Manifold alignment ensures that the semantic behavior of the adapter is preserved across different architectural dimensions.
- **Head-wise Orthogonal Procrustes**: Aligns attention heads between source and target dimensions.
- **CKA Calibration**: Measures the feature-map similarity to verify that the projection hasn't collapsed the internal representations.

## 3. The 7-Stage Diagnostic Pipeline

Neural-Scalpel enforces a rigorous validation pipeline to ensure scientific integrity:

1. **Stage 1: Metadata Gate**: Verifies model lineage, licensing, and base-model hashes.
2. **Stage 2: Source Quality Gate**: Confirms the adapter is a "Positive Teacher" on its original architecture.
3. **Stage 3: Delta Health Gate**: Spectral analysis of weights to detect rank collapse or instability.
4. **Stage 4: Compatibility Gate**: Hidden-size ratio and tokenizer vocabulary verification.
5. **Stage 5: Feasibility Gate**: Configuration-level structural mapping (GQA-awareness, layer mapping).
6. **Stage 6: Target Evaluation Gate**: Final benchmarking on the target student model.
7. **Stage 7: Release Decision Gate**: Automated unified decision engine (RELEASE_READY / RESEARCH_ONLY).

---

## 4. Scientific Release Pipeline (v2.3)

To ensure reproducibility and scientific transparency, Neural-Scalpel v2.3 automates the generation of a **Chain of Evidence Report**. This report consolidates all internal metrics (CKA, rSVD rank, spectral health, and behavioral fixed/regressed counts) into a single, immutable audit trail for each transplantation run.

---

## 5. Interference-Aware Gating (v2.10)

Neural-Scalpel v2.10 introduces **Interference-Aware Gating (IAPG)**, a surgical refinement layer that mitigates destructive interference between task-specific weight deltas and target-base knowledge.

- **Module-Alpha-Map**: Enables per-module-family alpha scaling (e.g., Attention vs. MLP).
- **Strict Gating**: Physically excludes modules with `alpha=0` from the final adapter configuration, rather than including them with zero weights.
- **Sentinel-Safe Transfer**: Prioritizes structural integrity by monitoring "Sentinel Cases" (sensitive task primitives) to detect internal behavioral collapse before regressions occur.

---

---

## 6. Risk-Calibrated Safety Mapping (v2.11)

Neural-Scalpel v2.11 reframes alpha selection as a model-specific **Safety Mapping** problem rather than a universal constant prediction problem. The framework establishes a diagnostic foundation for constructing empirical maps of stable and unstable scaling regions.

Key features of the v2.11 workflow include:
- **Comprehensive Metadata Tracking**: Records both projection metadata (global alpha, module-alpha-map) and evaluation metadata (dtype, adapter merge state) to ensure 100% scientific reproducibility.
- **Effective Delta Metrics**: Calculates projected-delta norm ratios across all layers to estimate the physical "Safety Budget" of the target model.
- **Case-Level Regression Analysis**: Quantifies the delta in behavioral performance (Fixed vs. Regressed) to determine an empirical Verdict (WINNER/SAFE/BOUNDARY/UNSAFE).
- **Avoid-Band Identification**: Recognizes non-monotonic safety behavior, such as the localized failure zones observed in 0.5B models (e.g., Alpha=3.0 instability), and generates actionable safety maps with excluded scaling intervals.

### 6.1 Limitations

The current safety map is specific to the tested Qwen2.5-Coder 7B SQL DPO → Qwen2.5-Coder 0.5B SQL-50 setting. Avoid bands and safe alpha regions should not be assumed to transfer to other model families, tasks, decoding settings, or evaluation dtypes without re-running the safety mapping workflow.

---

## 7. Performance Metrics

Validated locally on an NVIDIA RTX 5060 Ti 16GB under SQL-50 case-study conditions.

- **Source Adapter Quality Gate:** Positive Teacher behavior observed for the tested Qwen2.5-Coder SQL-DPO adapter.
- **Internal Signal Retention:** Maintained **~92%** CKA similarity (internal representation proxy) for identity-sized projections. Note: This is a structural metric and does not directly equate to behavioral task performance.
- **v2.11 Safety Mapping Results:** In the Qwen2.5-Coder 7B→0.5B setup, attention-only projections were validated as SAFE at Alpha=[0.5, 1.0, 2.0, 4.0], with a localized **UNSAFE "Avoid Band" at [2.75, 3.25]**.
- **v2.10 Fixed-Extractor Result:** **+2.0pt accuracy improvement** (from 24.0% to 26.0%) with **zero regressions** and `joins_007=PASS` in the documented sentinel set.
- **Historical Structural Projection Result:** +4.0% improvement (from 32% to 36%) under the earlier SQL-50 extraction setup.
- **Cross-size Generalization:** Preliminary positive SQL-50 deltas observed across 1.5B and 3B targets under selected benchmark conditions.
- **Scope:** These metrics are benchmark-specific and do not guarantee general SQL or production performance. Regression may occur depending on target model local minima.