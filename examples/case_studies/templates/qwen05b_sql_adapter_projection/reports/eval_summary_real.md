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
> These are observed failure modes from the current 4-prompt greedy smoke test.

- **No observable adapter effect:** Base and Projected outputs were identical across the 4-prompt greedy smoke set at multiple scales ($\gamma=1.0, 4.0$).
- **Behavioral transfer not detected:** The current initial scaffold (target-only calibration / mean-only JTSA) did not measurably change SQL/Coding outputs under greedy decoding.
- **Target-only calibration limit:** Current alignment is a target-only self-alignment attempt and does not yet use paired source-target representation mapping.
- **SQL improvement not proven:** SQL outputs were already reasonable from the base model, and the projected adapter did not improve them in the smoke set.
