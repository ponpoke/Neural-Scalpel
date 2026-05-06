# Behavioral Alignment Research Notes: From Negative Baseline to Signal Delivery

These notes document the transition from the Phase 4 "Negative Baseline" to the Phase 5/6 "Preliminary Runtime Success" in the Qwen2.5-0.5B SQL alignment experiment.

## The Problem: The "Negative Baseline" (Phase 4)

Initial attempts to project a 7B SQL adapter into a 0.5B model using **target-only statistical calibration** (PCSI/WDR/JTSA without paired activations) failed completely.

- **Observation**: The target model's output distribution remained 100% unchanged. Greedy decoding produced bit-identical output to the base model.
- **Root Cause**: Structural projection ensures the adapter "fits" the target shape, but it does not ensure the "signal" is directed correctly within the target's latent manifold. Without a reference to the source model's activations, the translation is mathematically unanchored.

## The Breakthrough: Paired Activation Alignment (Phase 5)

We pivoted to a **paired-activation pipeline**, treating the problem as a translation between disparate latent manifolds ($M_{source} \to M_{target}$).

### 1. Manifold Translation via Ridge Regression
By running the same prompts through both models, we learn a translation matrix $P$ such that:
$$ H_{source} P \approx H_{target} $$
This anchors the source model's "experience" of a prompt to the target model's "experience" of the same prompt.

### 2. Behavioral Delta Transport
We extract the behavioral shift $\Delta H_s$ caused by the adapter in the source model and transport it:
$$ \Delta H_t = \Delta H_s P $$
This directs the "intelligence signal" into the specific region of the target's manifold that corresponds to the source's behavioral shift.

### 3. SVD-based LoRA Export
The full-rank weight solution for $\Delta H_t$ is decomposed using SVD into a rank-16 LoRA.
- **Result**: Reduced artifact size from ~600MB (full-rank) to **8.8MB** (PEFT).

## Experimental Results

### Behavioral Scaling Gate (Phase 5-G8)
We tested scaling the transported signal using `lora_alpha`.
- **$\alpha=8/16$**: Stable behavioral shift. KL divergence became non-zero, and token shifts occurred.
- **$\alpha=32$**: Repetition collapse. The model began repeating the same token indefinitely, indicating that the signal was overpowering the base model's coherence.

**Optimal Scale**: `alpha=16` was selected as the stable operating point.

### Qualitative Findings (Phase 6 Smoke Test)
In the 0.5B model, the following behaviors emerged:
- **Base Model**: Generated simple `SELECT * FROM table WHERE ...` queries.
- **Projected Model**: Began using CTEs (`WITH ... AS`) and Window Functions (`OVER (PARTITION BY ...)`) for the same prompts, showing qualitative movement toward more advanced SQL-style structures observed in the source-side behavior.

## Summary of Success

1. **Signal Delivery**: Demonstrated that an activation-space signal can produce measurable runtime effects across model scales under the tested setup.
2. **Runtime Integration**: Verified that these projected adapters can be loaded into the Neural-Scalpel vLLM runtime and produce measurable shifts.
3. **Efficiency**: Achieved a 98.5% reduction in adapter size compared to full-rank projections.

## Pending Work

- **Execution Validation**: Does the "advanced SQL" actually run and return correct data?
- **Benchmark Expansion**: Scaling the 4-sample smoke test to the full 50-sample SQL evaluation set.
- **Architecture Generalization**: Testing Llama-3 to Qwen-2.5 paths.

---
**Status**: Experimental Scaffold Validated. Preliminary success observed.
**Date**: May 2026
