# Empirical Consistency & Logic Validation Report

This report tracks the mathematical and logical consistency of the Neural-Scalpel framework across its core components.

## 1. Core Mathematical Validation
Verified via automated unit tests in `tests/`:
- **Ridge Solver**: Accuracy of $XW = Y$ solutions with L2 regularization.
- **WDR (Wasserstein Discrete Routing)**: Sinkhorn convergence and mapping integrity.
- **SVD Decomposition**: Energy retention and low-rank reconstruction accuracy.
- **Coordinate Projection**: Procrustes error minimization verified across head-wise transformations.

## 2. Structural Projection Integrity
- **Shape Validation**: Automated rejection of malformed or mismatched tensor shapes during projection.
- **Fused Tensor Handling**: Correct construction of Qwen-style `qkv_proj` and `gate_up_proj`.
- **GQA Projection**: Verified head-wise expansion logic for Grouped Query Attention compatibility.

## 3. Behavioral Alignment Scaffold (New)
Validated in Phase 5/6:
- **Paired Activation Dataset**: Verification of sample count and manifold alignment.
- **Delta Transport**: Mathematical validation of $\Delta H_s P \to \Delta H_t$ consistency.
- **PEFT Export Validation**: Automated check of exported LoRA state dicts for PEFT/Transformers compatibility.
- **Scaling Curve**: Observation of the alpha-to-collapse phase transition curve.

## 4. Runtime & Serving Validation
Validated via live smoke tests and endurance logs:
- **Hook Injection**: Verified that activation injection occurs at the correct layer index.
- **Logit KL Divergence**: Monitoring for non-zero behavioral shifts during runtime.
- **Rollback Consistency**: Checksum-level verification that the model returns to 100% bit-identical base state after route removal.
- **Proxy Fallback**: HTTP forwarding integrity and fail-closed mechanism.

---

## 5. Summary of Test Status

| Category | Status | Notes |
| :--- | :--- | :--- |
| Core Math | **PASS** | 200+ non-live unit tests. |
| Structural Projection | **PASS** | Verified for Llama-3 and Qwen-2.5 families. |
| Behavioral Alignment | **EXPERIMENTAL** | Preliminary success in 7B-to-0.5B SQL path. |
| Runtime Hot-Swap | **PASS** | Verified via 10K swap endurance test. |
| Proxy Fallback | **PASS** | Smoke tests for HTTP forwarding complete. |

---

## 6. Known Failure Modes & Guardrails

- **Alpha Collapse**: High `lora_alpha` (>32) leads to repetition. The framework now warns when alpha exceeds rank.
- **Numerical Instability**: Ridge inversion can fail if activations are rank-deficient. The solver uses Float64 for internal inversion to mitigate this.
- **Version Lock**: vLLM internal patches are version-sensitive. `auto` serving mode falls back to external proxy if internal patching fails.

*Last Updated: May 2026*