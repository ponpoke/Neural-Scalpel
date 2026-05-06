# Paired Alignment Core Migration Definition

## Status
**Experimental core migration.** This framework provides a pipeline for behavioral signal transport; however, behavioral transfer and task-level skill acquisition are not guaranteed and must be verified per use case.

## Objective
The goal is to formalize the Phase 5 paired-activation alignment logic into reusable Neural-Scalpel core modules. This enables researchers to:
1. Align latent spaces between different model architectures.
2. Extract behavioral deltas from source adapters.
3. Transport those deltas into a target model's manifold.
4. Solve and export the result as a PEFT-compatible LoRA.
5. Validate the signal delivery through a multi-stage gate system.

## Non-goals
- **No Guarantee of Skill Transfer**: Does not claim that "intelligence" or "skills" are automatically transferred.
- **No Automated Improvement**: Does not guarantee that the target model will perform better on specific tasks.
- **No Replacement for Benchmarking**: Does not replace the need for full scale task-level evaluations (e.g., SQL execution accuracy).
- **No Black-box Transfer**: Requires calibration data (paired prompts) and architecture-specific mapping.

## Core Components (API Boundaries)

### 1. Data Objects
- `PairedActivationDataset`: Encapsulates source/target hidden states across calibration prompts.
- `AlignmentMap`: Represents the learned transformation (e.g., Ridge/Procrustes) between source and target layers.
- `BehavioralDelta`: The extracted difference between source-base and source-with-adapter hidden states.
- `TransportedDelta`: The BehavioralDelta projected into the target manifold via the AlignmentMap.
- `ActivationAdapterSolution`: The solved weight delta (full-rank) that approximates the TransportedDelta.
- `PeftExportResult`: The final low-rank (SVD) weights and PEFT configuration.
- `ValidationReport`: Standardized output of the multi-stage gate system.

### 2. High-level API (Phase B/C/D)
- `scalpel.align()`: Learns the translation map $P$.
- `scalpel.extract_behavior_delta()`: Captures source-level shifts.
- `scalpel.transport_delta()`: Projects shifts to the target.
- `scalpel.solve_activation_adapter()`: Computes required weight changes.
- `scalpel.export_lora()`: Produces PEFT-compatible artifacts with SVD compression.
- `scalpel.validate_behavior()`: Executes the gate-based validation suite.

## Multi-stage Validation Gates
The framework must enforce the following gates to ensure signal integrity:
- **G1 (Signal Presence)**: Source adapter signal is non-zero and detectable.
- **G2 (Layer Correspondence)**: CKA or MSE indicates stable layer mapping.
- **G3 (Alignment Quality)**: Held-out validation error for the alignment map is within thresholds.
- **G4 (Transport Stability)**: Transported deltas are finite and maintain directional similarity to source deltas.
- **G5 (Solve Accuracy)**: Reconstruction error of the activation-space solver is minimized.
- **G6 (Runtime Smoke)**: Dynamic hook injection produces observable logit/Top-1 shifts.
- **G7 (Artifact Integrity)**: PEFT load succeeds and weights are non-zero.
- **G8 (Behavioral Shift)**: KL divergence and text shifts confirm signal delivery.
- **G9 (Task Evaluation)**: Task-specific metrics (e.g., SQL parse success) are evaluated independently.

## Terminology
- **Behavioral Alignment**: The process of aligning generation patterns rather than task performance.
- **Adapter Signal Transport**: Moving a weight-induced shift from one model's activation space to another.
- **Experimental Scaffold**: A foundation for further research, not a production-ready "intelligence engine".
