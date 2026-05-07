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

## 5. Performance Metrics

Validated locally on an NVIDIA RTX 5060 Ti 16GB under SQL-50 case-study conditions.

- **Source Adapter Quality Gate:** Positive Teacher behavior observed for the tested Qwen2.5-Coder SQL-DPO adapter.
- **Internal Signal Retention:** Maintained **~92%** CKA similarity (internal representation proxy) for identity-sized projections. Note: This is a structural metric and does not directly equate to behavioral task performance.
- **Qwen2.5-Coder-0.5B Target Delta:** **+4.0% accuracy improvement** (from 32% to 36%) in the documented SQL-50 case study.
- **Cross-size Generalization:** Preliminary positive SQL-50 deltas observed across 1.5B and 3B targets under selected benchmark conditions.
- **Scope:** These metrics are benchmark-specific and do not guarantee general SQL or production performance. Regression may occur depending on target model local minima.