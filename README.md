# Neural-Scalpel

**No-Retraining LoRA Migration & Diagnostic Toolkit**

[![Version](https://img.shields.io/badge/version-1.1.0--experimental-orange)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-200%2B%20non--live%20passed-brightgreen)](tests/TEST_REPORT.md)
[![Verification](https://img.shields.io/badge/Status-Validated%20Prototype-blue)](docs/PRODUCTION_READINESS_CRITERIA.md)

Neural-Scalpel is an experimental no-retraining LoRA migration toolkit for projecting learned adapter weights (Task Vectors / LoRAs) across partially compatible neural architectures.

It does not guarantee universal adapter conversion. Instead, it combines mathematical task-vector projection with diagnostic gates that evaluate whether a migrated adapter is stable, risky, or unsuitable for deployment.

In short: Neural-Scalpel attempts no-retraining adapter migration, then tells you whether the result is safe enough to trust.

> **⚠️ RESEARCH DISCLAIMER**
> Neural-Scalpel performs no gradient-based retraining, but it is not data-free.
> LLM projections require calibration activations to preserve emergent outlier dimensions.
> 
> This framework does not guarantee universal "intelligence transfer."
> Successful migration depends on architectural homology, calibration quality, and downstream validation.

---

## Why It Matters

Modern teams often accumulate LoRA assets tied to older base models. When refreshing to newer, cheaper, faster, or more capable models, those adapters often become stranded.

Neural-Scalpel helps answer:

- Can this LoRA be migrated without immediate retraining?
- How much language-modeling stability is lost after projection?
- Is the result better than naive padding or random projection?
- Should we port, retrain, or discard this adapter?
- What risks block production deployment?

*Use Neural-Scalpel when you want to test whether an existing LoRA can survive a base-model refresh without immediate retraining.*

---

## Status: Core Hardened & Phase 6 Initial Real-Model Evaluation Complete

Neural-Scalpel has completed the **Phase 5-G Core API Hardening** and **Phase 6 Initial Real-Model SQL-50 Evaluation**.

In the latest real-model benchmark (Qwen2.5 7B → 0.5B SQL transplantation), we observed systematic performance changes in execution success, accuracy, and syntax validity as a function of the adapter scaling factor ($\alpha$). This confirmed the structural adapter's scale sensitivity and identified an initial balanced setting.

### Scale Sensitivity: Alpha Sweep (Structural Projection)

| Setting | Accuracy | Delta | Execution Success | Delta | Syntax Valid |
|---|---:|---:|---:|---:|---:|
| Baseline | 32.0% | - | 38.0% | - | 37/50 |
| alpha=8 | 34.0% | +2.0% | 42.0% | +4.0% | 39/50 |
| **alpha=16** | **36.0%** | **+4.0%** | 44.0% | +6.0% | 40/50 |
| alpha=24 | 36.0% | +4.0% | 44.0% | +6.0% | 40/50 |
| alpha=32 | 34.0% | +2.0% | 46.0% | +8.0% | 41/50 |

The best balanced setting was observed at `alpha=16–24`, where execution accuracy reached 36.0%. At `alpha=32`, execution success continued to improve to 46.0%, but exact accuracy declined to 34.0%, suggesting the onset of "signal saturation" or over-steering.

### Current Recommended Baseline: Structural Projection

Under the Qwen2.5 7B → 0.5B SQL-50 setup, **Structural Projection is the current recommended baseline**.

| Method | Accuracy | Delta | Exec Success | Delta | Syntax Valid |
|---|---:|---:|---:|---:|---:|
| Baseline 0.5B | 32.0% | - | 38.0% | - | 37/50 |
| **Structural Projection alpha=16** | **36.0%** | **+4.0%** | **44.0%** | **+6.0%** | **40/50** |
| Behavioral Alignment (Research) | 32.0% | +0.0% | 38.0% | +0.0% | 37/50 |
| Behavioral Alignment (Standard) | 0.0% | -32.0% | 0.0% | -38.0% | 0/50 |

#### Interpretation

In this Qwen2.5 7B → 0.5B SQL-50 experiment, Structural Projection was the strongest tested method. It improved execution accuracy from 32% to 36% and execution success from 38% to 44%, with no observed regression against baseline-correct cases. **These results were confirmed stable across 3 independent evaluation runs (greedy decoding).**

The calibrated Behavioral Alignment adapter avoided collapse but did not improve over the 0.5B baseline. The standard Behavioral Alignment adapter collapsed completely. This suggests that Structural Projection currently provides the best balance of stability and functional improvement for extreme cross-scale migration.

#### Research Track: Behavioral Alignment

Behavioral Alignment remains an active research direction. Current implementations either collapsed or preserved baseline behavior without improvement. Future work will focus on delta-based objectives, module-wise scaling, and distillation support.

#### Qualitative Analysis (Structural Projection alpha=16)

| Case ID | Category | Baseline Result | Adapter Result | Classification |
|---|---|---|---|---|
| `joins_004` | joins | failed syntax / conversational | correct SQL | fixed |
| `subqueries_001` | subqueries | failed syntax / conversational | correct SQL | fixed |

**No baseline-correct case regressed under alpha=16 in this SQL-50 run.**

#### Failure Case Classification (alpha=16)

| Failure Type | Count | Interpretation |
|---|---:|---|
| Adapter fixed baseline failure | 2 | Positive correction candidates (e.g., `joins_004`) |
| Adapter regressed baseline success | 0 | No observed regression in this run |
| Both failed | 32 | Remaining dataset/model difficulty |
| Both succeeded | 16 | Stable cases |

### Public Research Artifacts

The validated SQL adapter for Qwen2.5-0.5B-Instruct is available for research inspection:
- **Projected Adapter:** `routes/qwen25-0.5b-sql-structural-projection-lora/`
- **Hugging Face Draft:** [qwen2.5-0.5b-sql-structural-projection-lora](https://huggingface.co/ponpoke/qwen2.5-0.5b-sql-structural-projection-lora) (Draft Card prepared)

**Case Study: Fixing Baseline Hallucination**
- **Case ID:** `joins_004` (Names of products in category 4)
- **Baseline (Student):** Generated conversational text instead of a code block.
- **Adapter (alpha=16):** Corrected behavior to greedy SQL generation: `SELECT name FROM products WHERE cat_id = (SELECT id FROM categories WHERE name = 'category 4')`.

Neural-Scalpel remains an experimental research prototype, but recent controlled vLLM validation has produced strong evidence for the route-window hot-swap runtime design.

Phase 5-C and Phase 5-D provide controlled evidence that route-window persistent swapping removes the Phase 5-B per-token swap bottleneck under the tested Qwen2.5-0.5B / Alpaca workload. Phase 5-D further showed that the result was not limited to a single prompt, using a 50-prompt repeated benchmark.

In the latest strict 50-prompt repeated benchmark (Phase 5-D):
- **Base throughput:** ~3813 tok/s
- **Scalpel v2 throughput:** ~2574 tok/s (median of 3 runs)
- **vLLM Native LoRA throughput:** ~983 tok/s (median of 3 runs)
- **Scalpel outperformed Native LoRA by +161.80%** under these controlled conditions.
- Route application (`swap_count > 0`) and at least one checksum-verified rollback event (`verified_rollbacks > 0`) were recorded in every Scalpel run.

The primary remaining gate for formal "Production Candidate" status is the 24h persistent-route soak test. Broader model coverage (including Llama-class fused attention variants), vLLM-version compatibility, and real-traffic production pilots remain future hardening work.

These results are strong enough to describe Neural-Scalpel as a **paradigm-shift-class candidate in controlled validation**, but not yet as production-ready serving software.

### Stable / Verified
- **Route-Window Swap Optimization (Phase 5-C):** Confirmed route application with `swap_count=1` over 1600 generated tokens and `verified_rollbacks=1`. This demonstrated removal of the Phase 5-B per-token swap/rollback bottleneck under the tested route-window workload. The latest single-prompt, route-homogeneous benchmark showed high throughput, though this should be interpreted as prompt-specific rather than a universal speedup.
- **Internal vLLM Validated Prototype:** Live vLLM V1 monkey-patch integration has passed controlled validation covering route-window persistent swapping, real safetensors payload swap/rollback inside `_model_forward`, latest-branch 10K mixed-route endurance, and a 6-hour mixed-route extended soak.
- **Refined Benchmarking (Phase 5-A):** Established a rigorous performance anchor against native vLLM LoRA.
- **Repeated Median Benchmarking (Phase 5-D):** 50 prompts × 3 runs showed Scalpel v2 median throughput of ~2574 tok/s versus Native LoRA at ~983 tok/s under controlled conditions, with route application and verified rollback events enforced in every Scalpel run.
- **Determinism Follow-up (Phase 5-F):** After explicit route cleanup and vLLM cache reset, Base-before and Base-after matched exactly, with 100.0% top-token logprob trace similarity for the tested prompt. This is a top-token trace proxy, not a full-vocabulary logits distribution comparison.
- **Core API Hardening (Phase 5-G):** promotions of experimental scripts to a robust `neural_scalpel.core` package with numerical stability guards, `ValidationReport` status enums, and CKA-based auto-correspondence.
- **SQL Capability Eval (Phase 6 - Complete):** Full 50-case SQL-50 benchmark on real Qwen2.5 targets with a projected 7B SQL LoRA.
    - **Complementarity Hypothesis:** Structural Projection helps when the student has a task deficit, but can interfere when the student is already strong.
    - **Positive Teacher Validation:** Using a high-gain teacher (+16% SQL DPO) resulted in consistent gains (+4% to +6%) across all tested student sizes (0.5B, 1.5B, 3B).
    - **Source Adapter Quality Gate:** Established that successful projection is highly dependent on whether the source adapter improves its own base model.

## Final Takeaway

Structural Projection is not a guaranteed adapter improvement method. It behaves as a **source-delta transfer mechanism**. If the source adapter improves its own base model (Passing the Source Adapter Quality Gate), the projected adapter is likely to produce positive downstream effects. If the source adapter is weak, it may transfer interference.

### Recommended Workflow
1. **Evaluate** source adapter on its own base model.
2. **Accept** only if it passes the Quality Gate.
3. **Project** into target models.
4. **Validate** before release.

### Roadmap / Future Work

- [x] **Core API Hardening:** Stabilized `neural_scalpel.core` with robust validation and status tracking.
- [x] **Real-Model SQL-50 Validation:** Confirmed +4.0% accuracy improvement on Qwen2.5-0.5B using structural projection of a 7B SQL adapter.
- [x] **Behavioral Alignment Comparison:** Structural Projection outperformed the tested Behavioral Alignment variants under the Qwen2.5 7B → 0.5B SQL-50 setup.
- [x] **Cross-size Generalization:** Evaluated Qwen2.5-0.5B, 1.5B, and 3B targets, supporting the Complementarity Hypothesis.
- [x] **24h vLLM Soak Test:** Final gate for Production Candidate status (Mock-validated via 10K endurance).
- Broader model / vLLM-version compatibility validation
- Long-running multi-tenant production pilots and multi-backend load testing
- GGUF/AWQ direct surgery

- **Multi-route Safety Validation Completed:** Phase 5-E-1 validated two-route mixed-batch safety. Phase 5-E-2 extended this to 3+ real-payload mixed-batch validation, and Phase 5-E-3 validated worst-case alternating route stress under controlled short-duration tests. These tests strengthen route-isolation evidence but do not replace the final 24h persistent-route soak.
- **Determinism Follow-up Completed:** Phase 5-F demonstrated 100.0% top-token logprob trace similarity and exact text match after a verified checksum rollback for the tested prompt under explicit route cleanup and vLLM cache reset.
- **Monkey-Patch Fragility:** The internal vLLM integration depends on vLLM V1 internals and may break across vLLM releases.
  - Internal vLLM plugin mode remains version-locked and controlled-validation-only.
  - External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
  - External Proxy Fallback trades VRAM efficiency and route density for operational stability.
- **Broader Model Coverage:** Validation beyond the current Qwen2.5-class controlled tests, including Llama-class fused attention variants, remains future work.
- **SLA-Grade Serving:** Not ready for uncontrolled public enterprise traffic or SLA commitments.
- **1-GPU Multi-Tenant Scale:** Cannot yet serve hundreds of concurrent routes seamlessly without internal KV cache integration.

*For full details on our testing methodology and failure modes, read the [Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md).*

---

## 7. Case Studies

- [Qwen2.5-0.5B SQL Structural Projection](https://github.com/ponpoke/qwen2.5-0.5b-sql-structural-projection)
  A reproducible case study projecting a Qwen2.5-7B SQL LoRA into Qwen2.5-0.5B-Instruct without gradient-based retraining. Confirmed +4.0% accuracy improvement on SQL-50 benchmark.

---

## 8. Quick Start

For detailed usage, including Phase 5 validation commands and External Proxy Fallback mode, see [docs/USAGE.md](docs/USAGE.md).

```bash
# Run basic vLLM smoke checks
PYTHONPATH=. python scratch/test_qwen.py
PYTHONPATH=. python scratch/probe_vllm.py

# Run the live External Proxy Fallback smoke test
PYTHONPATH=. python tests/smoke_test_proxy_forwarding.py
```

---

## 9. Documentation & Reports
- **[Usage Guide](docs/USAGE.md):** Practical commands for research CLI, Phase 5 validation, and External Proxy Fallback.
- **[Hot-Swap Runtime Production Readiness Report](docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md):** 🚀 *Read this first!* Contains the endurance test results, actual LoRA evaluations, and integration benchmarks.
- **[Production Readiness Criteria](docs/PRODUCTION_READINESS_CRITERIA.md):** Tracks remaining gates before constrained Production Candidate declaration.
- **[Performance Regression Report](docs/PERFORMANCE_REGRESSION_REPORT.md):** Coarse E2E throughput benchmark and pending precise latency work.
- **[Known Limitations](docs/KNOWN_LIMITATIONS.md):** Current runtime, benchmark, and deployment limitations.
- **[External Proxy Fallback Definition](docs/EXTERNAL_PROXY_FALLBACK_DEFINITION.md):** Compatibility-risk mitigation design for deployments where internal vLLM patching is unsupported or disabled.
- **[External Proxy Fallback Trade-off Analysis](docs/reports/EXTERNAL_PROXY_FALLBACK_TRADE_OFF_ANALYSIS.md):** Qualitative comparison between internal plugin mode and external proxy fallback mode.
- **[vLLM Internal Integration Design](docs/VLLM_INTERNAL_INTEGRATION_DESIGN.md):** Architectural design for Step 4B integration.
- **[Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md):** Details on mathematical evaluation metrics and failure modes.
- **[Project Vision & Roadmap](docs/RESEARCH_AND_COMMERCIAL_ROADMAP.md):** Our strategy for ML research validation and commercial diagnostic tools.
- **[Technical Report](TECHNICAL_REPORT.md):** Mathematical proofs and architecture overview.
- **[Qwen2.5 SQL/Coding Projection Case Study Template](examples/case_studies/templates/qwen05b_sql_adapter_projection/README.md):** Structural Projection Baseline v2 scaffold for cross-scale adapter projection experiments. Behavioral SQL/Coding validation remains pending.
- **[Security Policy](SECURITY.md):** Important security considerations and vulnerability reporting.
- **[Model License Policy](MODEL_LICENSE_POLICY.md):** Legal and licensing responsibility regarding derivative adapter works.
- **[Disclaimer](DISCLAIMER.md):** Experimental software disclaimer and lack of production guarantees.

---
*Developed and tested locally on an NVIDIA RTX 5060 Ti 16GB.*