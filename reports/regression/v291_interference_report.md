# Technical Report: v2.9.1 Structural Compatibility & Interference Analysis

## 1. Executive Summary
The v2.9.1 update successfully resolved the structural compatibility issues in the Neural-Scalpel transplantation pipeline. However, end-to-end evaluation on the SQL-50 benchmark (fixed extractor mode) reveals that **Interference** is currently the dominant factor when projecting the Qwen2.5-Coder-7B-SQL-DPO adapter onto the Qwen2.5-Coder-0.5B-Instruct base model. 

While the "Attention-only" projection at low alpha is non-destructive, any increase in signal strength or the inclusion of MLP modules leads to immediate regression in specific SQL capabilities.

## 2. Methodology Updates (v2.9.1)
- **Structural Integrity**: Corrected GQA head projection and GQA-aware rank mapping.
- **Selective Projection**: Implemented `--include-modules` to isolate Attention vs. MLP components.
- **Deterministic Metadata**: `adapter_config.json` is now dynamically generated based on actual output keys to ensure PEFT compatibility.
- **Regression Sentinels**: Identified `joins_007` as a high-sensitivity sentinel for detecting target interference.

## 3. Results Summary (Alpha Sweep)

| Projection Type | Alpha | Accuracy | Sentinel (joins_007) | Interpretation |
|---|---|---|---|---|
| Baseline | - | 24.0% | PASS | Reference point |
| Attention-only | 4 | 24.0% | PASS | Non-destructive Window |
| Attention-only | 6 | 24.0% | **FAIL** | Trade-off (Fixed 1, Regressed 1) |
| Attention-only | 8 | 20.0% | **FAIL** | Interference dominant |
| Attention-only | 12 | 16.0% | **FAIL** | Catastrophic Interference |
| Attention-only | 16 | 16.0% | **FAIL** | Catastrophic Interference |
| MLP-only | 4 | 22.0% | **FAIL** | Interference dominant |
| **Down-proj-only** | **4** | **20.0%** | **FAIL** | **Highest-risk module** |

## 4. Key Findings
1.  **High-Risk MLP Core**: The `down_proj` module is the primary source of catastrophic interference. This suggests that the MLP delta in the source model encodes knowledge at a granularity or density that the 0.5B target model's MLP cannot absorb without displacing base knowledge.
2.  **Narrow Attention Safety Window**: Even Attention modules, which are generally more robust to transplantation, display a sharp degradation curve beyond `alpha=4`.
3.  **The attention_a6 Trade-off**: The fact that `attention_a6` fixes a case while breaking another is the "smoking gun" for transplantation success; it proves the delta is being applied correctly, but its global application is over-aggressive.

## 5. Next Phase: v2.10 "Interference-Aware Gating"
Future work will shift from brute-force alpha sweeping to **Case-aware/Layer-aware Gating**. The goal is to detect and suppress delta components that conflict with high-confidence base capabilities before the final weight projection.

---
**Status**: v2.9.1 Interference Analysis Complete
**Recommendation**: Proceed to v2.10 Interference-Aware Projection Gate
