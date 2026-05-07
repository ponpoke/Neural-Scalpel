# Changelog

All notable changes to this project will be documented in this file.

## [2.2.0] - 2026-05-07
### Added
- **`safe-project` Orchestrator**: Unified CLI command to run the entire pipeline (Diagnose -> Project -> Evaluate) in one shot.
- **Official Demo Pipeline**: Standardized verification path using Qwen2.5-Coder models.

## [2.1.0] - 2026-05-07
### Added
- **Target Evaluation Gate**: Automated dual-pass benchmarking (Base vs Adapter) on target architectures.
- **Behavioral Delta Analysis**: Classification of failures into `fixed` and `regressed` cases.
- **State Restoration**: Enhanced `from_json` to preserve multi-stage diagnostic results across sessions.
- **Configurable Thresholds**: CLI support for `--positive-delta-threshold` and `--max-regression-rate`.
- **Release Decision Promotion**: Automated path from `PROJECTION_CANDIDATE` to `RELEASE_READY`.

## [2.0.1] - 2026-05-07
### Added
- **7-Stage Diagnostic Scaffold**: Formalized pipeline (Metadata, Quality, Health, Compatibility, Feasibility).
- **Hardened CLI**: Modular command structure under `neural_scalpel/commands/`.
- **Hugging Face Hub Integration**: Automatic metadata and config fetching for adapters.
- **GQA-Aware Feasibility**: Structural checks for Grouped-Query Attention compatibility.

## [1.1.0-experimental] - 2026-05-04
### Added
- **Structural Projection Engine**: Core math for cross-architecture weight transplantation.
- **Manifold Alignment**: Head-wise Orthogonal Procrustes for intelligence alignment.
- **Initial Research Scripts**: `source_adapter_quality_gate.py` for teacher validation.
