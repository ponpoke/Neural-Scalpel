# Neural-Scalpel Diagnostic Failure Cases

This document logs specific failure conditions detected during the diagnostic projection.

## Case 1: Outlier Collapse
- **Mode:** JTSA + WDR (Uncalibrated)
- **Symptom:** PPL Degradation spiked to > 1000%.
- **Cause:** Attempted to execute Jacobian Tangent Space Alignment assuming a standard normal distribution, zeroing out massive emergent activation outliers required for reasoning.
- **Resolution:** A calibration set of at least 32 sequences is required.

## Case 2: Extreme Adapter Norm Drift
- **Mode:** Procrustes + AVPS
- **Symptom:** L2 Norm of the projected adapter increased by 2.4x compared to the source adapter.
- **Cause:** The target architecture uses a different dimensional scale, causing the Orthogonal Procrustes alignment to stretch the weight manifold aggressively.
- **Resolution:** Enable quantization-aware Procrustes or dampen the `s_factor`.
