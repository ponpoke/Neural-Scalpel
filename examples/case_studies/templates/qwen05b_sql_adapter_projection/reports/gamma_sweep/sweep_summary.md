# Gamma Sweep Summary Report: Signal Detection Phase 4

- **Date**: 2026-05-07
- **Setup**: Qwen2.5-0.5B-Instruct Target with Qwen2.5-7B SQL-LoRA Projection
- **Method**: Target-Activation-Conditioned (JTSA-style self-alignment)
- **Gammas Tested**: [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

## Quantitative Summary

| Gamma | exact_same_rate_raw | exact_same_rate_normalized | behavioral_status |
| :---: | :---: | :---: | :--- |
| 0.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 1.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 2.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 4.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 8.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 16.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |
| 32.0 | 1.0 | 1.0 | IDENTICAL_TO_BASE |

## Key Findings

1.  **Zero Behavioral Divergence**: Despite increasing the adapter scale to 32x the nominal value ($\gamma=32.0$), the output remained bit-identical to the base model.
2.  **Argmax Boundary Stability**: The base model's probability distribution is sufficiently dominant ("Behavioral Gravity") that the current projected adapter deltas fail to shift the top-1 token selection during greedy decoding.
3.  **Self-Alignment Limit**: Conditioning the projection only on the target model's activation manifold (without source activation reference) is insufficient to bridge the architectural gap between 7B and 0.5B for this task.

## Interpretation & Next Steps

This negative result is highly valuable as it establishes a "Negative Baseline." It confirms that **structural validity $\neq$ behavioral transfer**.

### Immediate Recommendation: Logit Delta Verification
Before proceeding to more complex alignment methods, we must determine if the adapter is exerting *any* influence on the logits. Even if the top-1 token is identical, a shift in the logit distribution (e.g., increased probability for SQL keywords) would prove the presence of a "Sub-threshold Signal."

### Long-term Direction: Paired Alignment
If logit deltas are minimal, the focus must shift to **Paired Activation Alignment (Phase 5)**, where we learn an explicit cross-model mapping matrix $P$ to properly translate the 7B behavioral signal into the 0.5B representation space.
