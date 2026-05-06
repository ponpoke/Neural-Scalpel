# Neural-Scalpel: Advanced Adapter Transplantation Methodology

This document outlines the ideal research roadmap for projecting large-scale adapters (e.g., 7B/14B) into low-resource models (e.g., 0.5B) using the Neural-Scalpel framework.

## 1. Dimension Mismatch Resolution
**Ideal Solution: Activation-Calibrated Bidirectional Subspace Projection**

Instead of naive SVD resizing, we map the representation space by comparing how the source (7B) and target (0.5B) models process the same input.

- **Process**: Collect hidden states from both models using a *Task-balanced Calibration Set*.
- **Projection**: Estimate projection matrices $P_{in}$ (target $\to$ source) and $P_{out}$ (source $\to$ target) using Procrustes analysis.
- **Transformation**: $\Delta W_{target} \approx P_{out} @ \Delta W_{source} @ P_{in}$

## 2. Payload Optimization
**Ideal Solution: Target-shape Projection + Low-rank Recompression**

To maintain the efficiency of the "Scalpel" philosophy, the 12GB full-rank delta must be recompressed for distribution.

- **Artifact Separation**: 
    1. **Source Full-rank Delta**: Research intermediate (Large, private).
    2. **Target-shape Full-rank Delta**: Validation intermediate.
    3. **Target Low-rank Projected LoRA**: Public release (Small, efficient, ~50-100MB).
- **Metric-driven Rank Selection**: Compare Ranks (4, 8, 16, 32, 64) against reconstruction error and SQL task validity.

## 3. Depth Mismatch Resolution
**Ideal Solution: Activation-aware Layer Transport / Layer Folding**

Handle the 28-to-24 layer gap by measuring layer-wise representation similarity (CKA, Procrustes residual).

- **Weighted Folding**: Synthesize a target layer from multiple source layers based on their activation similarity and task-output impact.
- **Importance Scoring**: Prioritize layers with high weight norms or significant output drift.

## 4. Module-aware Projection
**Ideal Solution: Structural Alignment**

- **Attention**: Align head-to-head based on attention patterns.
- **MLP**: Align neuron-to-neuron based on activation statistics in the intermediate dimension.

## 5. Evaluation Framework
**Ideal Solution: 4-Layer Validation**

1. **Shape / Tensor Validation**: Ensure strict compatibility with target model runtime state_dict.
2. **Runtime Safety**: Verify atomic swapping and rollback integrity.
3. **Task Quality**: Measure SQL syntax validity, component match, and LLM judge preference.
4. **Failure Analysis**: Document hallucination patterns and base-model regressions.

---
*Note: This methodology serves as the "Ideal Form" for Neural-Scalpel case studies. Current implementations are moving from Baseline (Phase 1) toward these Advanced targets (Phase 2).*
