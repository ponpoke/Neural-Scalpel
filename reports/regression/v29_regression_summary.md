# Neural-Scalpel v2.9 Regression Test Summary

## 1. Executive Summary
The v2.9-unreleased advanced projection branch has undergone a full artifact-level regression test. All core functionalities, including alpha-scaling, piecewise projection, and adaptive scaling logic, have been verified against the production hardening criteria.

**Overall Verdict: PASS**

## 2. Test Matrix & Results

| Test Case | Mode | Alpha | Config (lora_alpha) | Shape Integrity | Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Alpha Sweep 1 | linear | 8 | OK (8) | PASS | **PASS** |
| Alpha Sweep 2 | linear | 16 | OK (16) | PASS | **PASS** |
| Alpha Sweep 3 | linear | 24 | OK (24) | PASS | **PASS** |
| Alpha Sweep 4 | linear | 32 | OK (32) | PASS | **PASS** |
| Piecewise A/B | piecewise | 16 | OK (16) | PASS | **PASS** |
| Numerical | all | - | - | No NaN/Inf | **PASS** |

## 3. Key Improvements & Fixes
- **CLI/Config Sync**: Fixed a critical bug where the `--alpha` argument was ignored in the saved `adapter_config.json`. All future transplantations will now correctly respect the user-specified alpha.
- **LoRA Pair-Awareness**: Piecewise projection successfully reconstructs $B \times A$ deltas from the stream, ensuring architectural stability for MLP layers.
- **Experimental Safety**: `kernel` and `jacobian` modes correctly trigger `RuntimeWarning` and fallback to linear, preventing system collapse.

## 4. Artifact Details
- **Test Date**: 2026-05-08
- **Source Architecture**: Llama-3 style (Mocked 7B)
- **Target Architecture**: Qwen-2 style (Mocked 0.5B)
- **Regression Suite**: `scripts/v29_regression_suite.py`

## 5. Verified Artifacts Location
- `runs/regression_test/linear_a16/`
- `runs/regression_test/piecewise_a16/`
- `runs/regression_test/linear_a32/`
