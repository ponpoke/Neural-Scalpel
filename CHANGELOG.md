# Changelog

## v2.12.0 - 1.5B Manifold Harmonization

### Added
- **1.5B Architecture Support**: Optimized transplantation scaffold for 28-layer models (tested on Qwen2.5-Coder-1.5B).
- **Layer-Specific Alpha Scaling**: Added support for `module.layer=value` syntax in `module_alpha_map` for surgical injection.
- **Manifold Harmonization**: Implemented triple-layer tapered injection profile (Shallow/Mid/Deep).
- **Structural Gain Diagnostic**: New classification for non-regressive structural improvements (Syntax Valid improvements).
- **Law of Balance Hypothesis**: Empirical discovery of spatial symmetry requirements in early model tiers.

### Fixed
- Fixed JSON serialization of `module_alpha_map` when using tuple-based layer keys.
- Corrected regex-based layer extraction for models with complex internal naming schemes.

## v2.11.0 - Risk-Calibrated Safety Mapping

### Added
- Risk-calibrated safety map generation with avoid-band detection.
- Projection and evaluation metadata tracking for reproducible alpha analysis.
- Unit tests for v2.11 safety map generation.

### Fixed
- Restored backward-compatible diagnostic APIs used by v2.1–v2.9 pipelines.
- Restored `AdapterTransferDiagnosticReport`, `DeltaHealthAnalyzer`, and diagnostic result dataclasses.
- Fixed SVD-based spectral entropy calculation for delta health analysis.
- Restored adapter class aliases/stubs required by the comprehensive test suite.
- Hardened CLI behavior under pytest to avoid unintended Hugging Face Hub downloads.

## [v2.9.0-unreleased]
### Added
- **Hardened Delta Health (v2.6)**: Implemented normalized spectral entropy and effective rank metrics with robust regex-based layer extraction.
- **Traceable Adaptive Scaling (v2.7)**: Configurable scaling rules and per-layer scale reporting (`applied_scales`) for auditing.
- **Pair-Aware Piecewise Projection (v2.8)**: Order-agnostic LoRA-pair buffering and delta-based SVD reconstruction for high-fidelity MLP transfer. Corrected scale application logic.
- **Health-Aware Orchestration**: `safe-project` now automatically suggest/upgrades to `piecewise` mode based on health diagnostics.

### Experimental / Research Status
- `kernel` and `jacobian` modes are research stubs providing conservative linear fallbacks with warnings (B-1 Research Roadmap).
- Non-linear manifold alignment remains in active research and requires calibration activations for production use.

## [v2.5.0] - 2026-05-07
### Added
- **Package Validation**: Automated integrity checks for release packages.
- **Integrity Hashes**: SHA256 validation for all adapter artifacts.
- **Hardened CLI**: Process exits with code 1 on validation failure.
