# Neural-Scalpel: Advanced Adapter Transplantation Methodology (Ideal State)

This document defines the "Ideal State" for projecting large-scale adapters (e.g., 7B/14B) into low-resource target models (e.g., 0.5B). Following the Phase 4 initial findings (100% Identity on 0.5B), this methodology prioritizes **Signal Extraction over Compression**.

---

## A. Paired Activation Alignment (The Bridge)

To bridge the gap between divergent architectures, we must move beyond target-only statistics.

1.  **Triple-Stream Collection**: Collect hidden states $H$ from three concurrent streams using identical prompts and token positions:
    - **$H_{source\_base}$**: Source model (e.g., 7B) without adapter.
    - **$H_{source\_lora}$**: Source model with the original high-quality adapter.
    - **$H_{target\_base}$**: Target model (e.g., 0.5B).
2.  **Explicit Mapping ($P$)**: Estimate projection matrices $P_{in}$ and $P_{out}$ such that $H_{source} @ P \approx H_{target}$ using:
    - **Orthogonal Procrustes**: For rotation-based alignment that preserves norms.
    - **Ridge Regression / CCA**: For flexible subspace matching.
3.  **Behavioral Delta ($\Delta H$)**: Calculate the *true behavioral signal* $\Delta H = H_{source\_lora} - H_{source\_base}$. The goal is to reproduce this $\Delta H$ in the target model.
4.  **Soft Layer Correspondence**: Map source layers to target layers using **CKA (Centered Kernel Alignment)** similarity instead of static linear folding.

## B. Manifold-Aware Projection (The Filter)

Ensure the transplanted signal is "audible" to the target model.

1.  **Target Manifold Projection**: Use the target model's PCA basis to filter the adapter delta. Keep only the components that align with the target's natural representation directions.
2.  **Activation-Weighted SVD**: When compressing weights, minimize the reconstruction error of the *activations* ($||X \Delta W - X B A||$) rather than just the weights.
3.  **Variance Scaling**: Scale adapter deltas based on the per-dimension standard deviation of target activations to prevent instability or signal suppression.

## C. Payload Optimization (The Footprint)

Once a behavioral signal is confirmed, minimize the adapter size (e.g., reduce 600MB $\to$ 50MB).

1.  **Module Selective Adapters**: Retain only the most impactful modules (e.g., Attention Q/V, MLP Gate/Up).
2.  **Layer Importance Pruning**: Discard layers where the adapter contribution to the final behavioral delta is negligible.
3.  **Delta Distillation**: Use the large projected adapter as a teacher to train a compact (Rank 8/16) target adapter via distillation on task-specific prompts.

## D. Evaluation & Validation (The Gate)

1.  **Signal Detection**: Measure the **Divergence Rate** from the base model. If Identity is 100%, the signal is missing.
2.  **Gamma Sweep Analysis**: Systematically vary $\gamma$ (0.5 to 32.0) to find the "Break-through Point" where behavior changes without causing model collapse (gibberish).
3.  **SQL Task Validity**: Move beyond heuristics to parse-based and execution-based metrics (e.g., using `sqlglot`).

---

> [!IMPORTANT]
> **Signal Extraction before Rank Compression.**
> The primary goal of Neural-Scalpel research is to prove that behavioral transfer is possible. Compression and quantization are secondary optimization steps to be performed once the "Signal" is audible in the target model.
