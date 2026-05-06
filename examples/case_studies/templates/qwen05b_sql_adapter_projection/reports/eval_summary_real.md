# Quantitative Evaluation Summary (REAL SMOKE)

> [!WARNING]
> This is a **preliminary heuristic smoke test**. Results are inconclusive regarding SQL capability improvement.

Evaluated on 4 curated prompts.

| Metric | Base Qwen2.5-0.5B | Projected SQL Route | Delta |
| :--- | :---: | :---: | :---: |
| Basic SQL Signal (Heuristic) | 75% | 75% | +0% |
| Observed Repetition Rate | 0% | 0% | +0% |
| Avg Output Length | 502 | 502 | +0.0 |

**Identity Rates (Base vs Projected):**
- Exact Bit-Identical: 100.0%
- Normalized (Whitespace-Insensitive): 100.0%

# Observed Failure Cases (REAL)

> [!NOTE]
> These are observed failure modes from the current Phase 4 initial target-activation-conditioned projection attempt.

- **No observable adapter effect:** Base and Projected outputs were identical across the 4-prompt greedy smoke set at multiple scales ($\gamma=1.0, 4.0$).
- **Behavioral transfer not detected:** The projected calibrated adapter did not produce measurable SQL/Coding behavior change under greedy decoding.
- **Target-only calibration limit:** The current calibration uses target activations only and does not learn a paired source-target representation map.
- **Mean-only conditioning limit:** Although mean/std/samples/PCA are stored, the current projection pass uses only the per-layer mean for JTSA-style conditioning. Manifold-rich alignment remains future work.
