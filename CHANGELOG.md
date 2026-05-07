# Changelog

## [v2.5.0] - 2026-05-07
### Added
- **Package Validation**: Automated integrity checks for release packages.
- **Integrity Hashes**: SHA256 validation for all adapter artifacts.
- **Hardened CLI**: Process exits with code 1 on validation failure.

## [2.4.0] - 2026-05-07
### Added
- **`package-release` Command**: Automated gathering of weights, scientific reports, and metadata into a distribution-ready folder.
- **Enhanced Metadata**: Publication packages now include `projection_metadata.json` with source/target and diagnostic verdict info.
- **Citation Generation**: Automatic creation of `CITATION.cff` from diagnostic metadata.

## [2.3.0] - 2026-05-07
### Added
- **`generate-report` Command**: Automated generation of detailed scientific Markdown reports from diagnostic and evaluation artifacts.
- **`generate-model-card` Command**: Automated Hugging Face Model Card (README.md) generation with YAML metadata.
- **Automation Pipeline**: Standardized evidence-to-publication workflow support.

## [2.2.0] - 2026-05-07
### Added
- **`safe-project` Orchestrator**: Unified CLI command to run the entire pipeline (Diagnose -> Project -> Evaluate) in one shot.
- **Safety Gates**: Automatic abort if diagnostic results do not reach `PROJECTION_CANDIDATE` threshold.
- **Official Demo Pipeline**: Standardized verification path using Qwen2.5-Coder models.

## [1.1.0] - 2026-05-06
### Added
- **Hardened Core Migration API**: Modularized AVPS, rSVD, and alignment engines into a reusable library.
- **Numerical Guards**: Added NaN/Inf detection and layer mapping validation.

