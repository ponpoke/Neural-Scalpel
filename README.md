# Neural-Scalpel: Experimental Behavioral Alignment & Hot-Swap Framework

Neural-Scalpel is a surgical transplantation framework for large language models. It enables **low-latency hot-swapping** of model weights and **experimental behavioral alignment** across model architectures and scales.

## Core Capabilities

- **Hot-Swap Runtime**: Zero-copy weight swapping for vLLM, enabling 10K+ route endurance with minimal latency overhead.
- **Structural Projection**: Project adapters across architectures (e.g., Llama-3 to Qwen-2) via head-wise routing and PCA-guided subspace injection.
- **Behavioral Alignment (Experimental)**: A paired-activation pipeline to transport behavioral signals from large source models to smaller target models.

---

## Case Study: Qwen2.5-0.5B SQL Behavioral Alignment

Neural-Scalpel now includes an experimental paired-activation behavioral alignment pipeline for cross-scale adapter transfer.

### Phase 4 Result (Target-only Alignment) - Negative Baseline
Initial target-only activation calibration produced:
- **100% bit-identical greedy outputs**
- No meaningful logit delta
- No detectable behavioral shift

This established a negative baseline demonstrating that target-only manifold statistics were insufficient for functional signal transfer.

### Phase 5 Result (Paired Activation Alignment)
A paired source-target alignment pipeline was introduced to bridge the manifold gap:
1. **Paired Activation Collection**: Capturing hidden states from both models on common prompts.
2. **Behavioral Delta Extraction**: Capturing the specific shift caused by the source adapter.
3. **CKA-based Layer Correspondence**: Estimating the optimal layer mapping.
4. **Ridge-based Manifold Translation**: Learning the transformation $P$ between latent spaces.
5. **Transported Delta Solving**: Computing target weight changes to replicate the transported signal.
6. **PEFT LoRA Export**: Exported a rank-16 PEFT adapter (~8.8MB), approximately 98.5% smaller than the earlier full-rank transported-delta artifact.

### Preliminary Runtime Success
Under calibrated PEFT scaling (`alpha=16`):
- **Top-1 behavioral shifts** were observed.
- **Runtime KL divergence** became non-zero.
- **Emergence of Advanced SQL**: The projected 0.5B model began generating CTEs (`WITH`) and Window Functions (`OVER`) where the base model used simple queries.
- **Stability**: Generation coherence was preserved without catastrophic repetition.

### Phase 6 Initial Smoke Evaluation
Initial 4-sample SQL smoke evaluation results:

| Metric | Base Model | Projected LoRA |
| :--- | :---: | :---: |
| SQL Parse Success | 75.0%* | 75.0%* |
| Advanced SQL Usage | 0.0% | 25.0% |
| Schema Hallucination | 0.0% | 0.0% |
*\*Including one non-SQL Python prompt.*

> [!IMPORTANT]
> **Status: Preliminary Runtime Success / Task Transfer Unverified**  
> These results provide preliminary evidence that paired activation alignment can alter downstream behavioral structure. However, they do **not** yet prove general SQL task improvement or execution accuracy.

---

## Status: Validated Prototype with Behavioral Scaffold

### Controlled-Validation Hot-Swap Runtime
- **Validated under controlled tests**: 10K+ route endurance and 6-hour mixed-route soak completed.
- **Pending**: Final 24h persistent-route soak before constrained Production Candidate status.
- **Architecture**: Atomic weight swapping integrated with vLLM engine.

### Experimental Behavioral Alignment Scaffold
The paired-activation pipeline is currently experimental.
- **Demonstrated**: Structural projection, activation transport, and preliminary behavioral shifts.
- **Pending**: Large-scale benchmark validation, execution accuracy proofs, and cross-family generalization.

---

## Documentation

- **[Alignment Definition](docs/PAIRED_ALIGNMENT_CORE_MIGRATION_DEFINITION.md):** Formal definition of the core migration API.
- **[Behavioral Alignment Notes](docs/BEHAVIORAL_ALIGNMENT_NOTES.md):** Research journey from negative baseline to preliminary success.
- **[Hot-Swap Readiness](docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md):** 🚀 Endurance and soak test results.
- **[Technical Report](TECHNICAL_REPORT.md):** Mathematical proofs and architecture overview.
- **[Usage Guide](docs/USAGE.md):** API workflow and CLI instructions.

---

## License

Neural-Scalpel is released under the Apache 2.0 License. See [LICENSE](LICENSE) and [MODEL_LICENSE_POLICY.md](MODEL_LICENSE_POLICY.md) for details.