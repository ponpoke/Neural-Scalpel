# Neural-Scalpel

**No-Retraining LoRA Migration & Diagnostic Toolkit**

[![Version](https://img.shields.io/badge/version-2.11.0-blue)](pyproject.toml)
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

## Status: Adapter Transfer Diagnostic v2.11.0 (Risk-Calibrated Safety Mapping)

Neural-Scalpel now provides a **comprehensive diagnostic-to-publishing workflow**:

1. `diagnose-adapter`: Multi-stage structural and behavioral feasibility check.
2. `project-adapter`: Experimental Structural Projection of weight deltas.
3. `evaluate-projected`: Target-side benchmarking and behavioral delta analysis.
4. `safe-project`: Unified orchestrator for the complete end-to-end pipeline.
5. **`generate-report`**: Automated creation of detailed scientific analysis reports.
6. **`generate-model-card`**: Automated generation of Hugging Face compatible Model Cards.

The framework classifies adapters as `PROJECTION_CANDIDATE`, runs structural projection, evaluates student-side behavior, and promotes successful runs to `RELEASE_READY` as a research artifact, subject to benchmark-specific validation.

> [!NOTE]
> Neural-Scalpel remains a research toolkit. `RELEASE_READY` means validated as a research artifact on the selected benchmark, not production deployment readiness.

In the latest real-model benchmark (Qwen2.5-Coder 7B → 0.5B SQL transplantation), we observed systematic performance changes in execution success, accuracy, and syntax validity as a function of the module-wise scaling factor ($\alpha$). This confirmed the effectiveness of **Interference-Aware Gating (IAPG)** in mitigating knowledge rejection.

### Phase 7 Success: Sentinel-Safe Positive Transfer (v2.10)

For the first time, Neural-Scalpel achieved **True Positive Transfer** involving both Attention and MLP components while maintaining zero regressions on sentinel cases.

| Setting | Accuracy | Fixed | Regressed | joins_007 | Status |
|---|---:|---:|---:|---|---|
| Baseline (0.5B Instruct) | 24.0% | 0 | 0 | **PASS** | Reference |
| v210_v0 (Attention-Only) | 24.0% | 0 | 0 | **PASS** | Validated |
| v210_v1c (Hybrid-Gated) | 26.0% | 1 | 0 | **PASS** | Best-Tested |

### v2.11 Evolution: Risk-Calibrated Safety Mapping (Diagnostic Infrastructure)

Neural-Scalpel v2.11 transitions from heuristic tuning to an **Empirical Diagnostic Framework**. Rather than searching for a universal alpha constant, v2.11 enables the construction of model-specific **Safety Maps**.

- **Non-Monotonic Safety Discovery**: Targeted sweeps revealed that safety does not follow a linear path. For example, in the Qwen2.5 7B→0.5B setup, **Alpha=3.0 was identified as a localized Failure Zone** (triggering Python hallucination), while higher (4.0) and lower (2.0) values remained safe.
- **Automated Risk Profiling**: The system now automatically generates `safety_map.json`, categorizing alpha ranges into `SAFE`, `UNSAFE`, and `AVOID BANDS`.
- **Reproducible Metrics**: Integrated `eval_metadata` (dtype/merge tracking) and true "Effective Delta" norm calculations for scientific traceability.

**Current Validated Safety Map (v2.11):**
- **Attention**: SAFE at [0.5, 1.0, 2.0, 4.0]. **AVOID BAND: [2.75, 3.25]**.
- **MLP**: High-sensitivity. Recommended Alpha: **0.0 (Gated)** or <0.02.

**Current Best Validated Configuration (v2.10):**
`--module-alpha-map q_proj=4,k_proj=4,v_proj=4,o_proj=4,gate_proj=0.125,up_proj=0.125,down_proj=0`
*(Note: Alpha values are relative to a global alpha of 16)*

> [!NOTE]
> The v2.10 hybrid-gated configuration is preserved as a successful historical run under its documented evaluation conditions. v2.11 re-evaluates the transfer landscape under stricter metadata-tracked conditions and reframes the result as part of a model-specific safety map rather than a universal recipe.

### Historical Scale Sensitivity: Alpha Sweep (Legacy Extractor)

The following results were obtained under an earlier SQL extraction/evaluation setup (Legacy Extractor) and are preserved for historical comparison. They should not be directly compared with the v2.10 fixed-extractor results above.

| Setting | Accuracy | Delta | Execution Success | Delta | Syntax Valid |
|---|---:|---:|---:|---:|---:|
| Baseline | 32.0% | - | 38.0% | - | 37/50 |
| alpha=8 | 34.0% | +2.0% | 42.0% | +4.0% | 39/50 |
| **alpha=16** | **36.0%** | **+4.0%** | 44.0% | +6.0% | 40/50 |
| alpha=24 | 36.0% | +4.0% | 44.0% | +6.0% | 40/50 |
| alpha=32 | 34.0% | +2.0% | 46.0% | +8.0% | 41/50 |

*Historical Interpretation:* The best balanced setting was observed at `alpha=16–24`. At `alpha=32`, execution success continued to improve, but exact accuracy declined, suggesting signal saturation.

### Historical Recommended Baseline Before Interference-Aware Gating (Legacy)

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

- **Released Adapter:** [qwen2.5-0.5b-instruct-sql-structural-projection-lora](https://huggingface.co/ponpoke/qwen2.5-0.5b-instruct-sql-structural-projection-lora)

**Case Study: Fixing Baseline Hallucination**
- **Case ID:** `joins_004` (Names of products in category 4)
- **Baseline (Student):** Generated conversational text instead of a code block.
- **Adapter (alpha=16):** Corrected behavior to greedy SQL generation: `SELECT name FROM products WHERE cat_id = (SELECT id FROM categories WHERE name = 'category 4')`.

---

### Recommended Workflow (v2.11.0)

1.  **Diagnose & Project**: Run `diagnose-adapter` and `safe-project` to establish source and target risk profiles.
2.  **Construct Safety Map**: Use targeted alpha sweeps (e.g., Attention-only) to construct a model-specific **Safety Map**.
3.  **Prescribe Surgery**: Use `project-adapter --module-alpha-map` with the validated safe region or avoid-band policy identified in the map.
4.  **Behavioral Validation**: Evaluate with `evaluate-projected` and inspect **Fixed / Regressed / Sentinel** cases.
5.  **Publish Artifacts**: Generate reports and model cards only after target-side validation confirms acceptable risk levels.

---

## Documentation & Reports
- **[Usage Guide](docs/USAGE.md):** Practical commands for research CLI, Phase 5 validation, and External Proxy Fallback.
- **[v2.11 Safety Map](reports/regression/v211_safety_map.json):** Model-specific safe/unsafe alpha regions and avoid bands for Qwen2.5-Coder 7B→0.5B.
- **[v2.11 Diagnostic Report](reports/regression/v211_diagnostic_report.md):** Full risk-calibrated safety mapping analysis.
- **[Chain of Evidence Report (Sample)](reports/sample_report.md):** Detailed automated analysis of a projection run.
- **[Technical Report](TECHNICAL_REPORT.md):** Mathematical proofs and architecture overview.
- **[Project Vision & Roadmap](docs/RESEARCH_AND_COMMERCIAL_ROADMAP.md):** Our strategy for ML research validation and commercial diagnostic tools.