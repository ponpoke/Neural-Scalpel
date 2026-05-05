# Neural-Scalpel

**No-Retraining LoRA Migration & Diagnostic Toolkit**

[![Version](https://img.shields.io/badge/version-1.0.0--alpha-orange)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-193%20non--live%20passed-brightgreen)](tests/TEST_REPORT.md)
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

## Status: Validated Prototype with Strong Controlled Runtime Evidence

Neural-Scalpel remains an experimental research prototype, but recent controlled vLLM validation has produced strong evidence for the route-window hot-swap runtime design.

Phase 5-C and Phase 5-D provide controlled evidence that route-window persistent swapping removes the Phase 5-B per-token swap bottleneck under the tested Qwen2.5-0.5B / Alpaca workload. Phase 5-D further showed that the result was not limited to a single prompt, using a 50-prompt repeated benchmark.

In the latest strict 50-prompt repeated benchmark (Phase 5-D):
- **Base throughput:** ~3813 tok/s
- **Scalpel v2 throughput:** ~2574 tok/s (median of 3 runs)
- **vLLM Native LoRA throughput:** ~983 tok/s (median of 3 runs)
- **Scalpel outperformed Native LoRA by +161.80%** under these controlled conditions.
- Route application (`swap_count > 0`) and at least one checksum-verified rollback event (`verified_rollbacks > 0`) were recorded in every Scalpel run.

The primary remaining gate for formal "Production Candidate" status is the 24h persistent-route soak test. Broader 3+ route and worst-case alternation stress tests remain future hardening work.

These results are strong enough to describe Neural-Scalpel as a **paradigm-shift-class candidate in controlled validation**, but not yet as production-ready serving software.

### Stable / Verified
- **Route-Window Swap Optimization (Phase 5-C):** Confirmed route application with `swap_count=1` over 1600 generated tokens and `verified_rollbacks=1`. This demonstrated removal of the Phase 5-B per-token swap/rollback bottleneck under the tested route-window workload. The latest single-prompt, route-homogeneous benchmark showed high throughput, though this should be interpreted as prompt-specific rather than a universal speedup.
- **Internal vLLM Validated Prototype:** Live vLLM V1 monkey-patch integration validated through route-window persistent swapping, real safetensors payload swap/rollback inside `_model_forward`, latest-branch 10K mixed-route endurance, and 6-hour mixed-route extended soak.
- **Refined Benchmarking (Phase 5-A):** Established a rigorous performance anchor against native vLLM LoRA.
- **Repeated Median Benchmarking (Phase 5-D):** 50 prompts × 3 runs showed Scalpel v2 median throughput of ~2574 tok/s versus Native LoRA at ~983 tok/s under controlled conditions, with route application and verified rollback events enforced in every Scalpel run.
- **Determinism Follow-up (Phase 5-F):** After explicit route cleanup and vLLM cache reset, Base-before and Base-after matched exactly, with 100.0% top-token logprob trace similarity for the tested prompt. This is a top-token trace proxy, not a full-vocabulary logits distribution comparison.

### Roadmap / Future Work

- Final 24h mixed-route soak validation with `--require-worker-health`
- Precise vLLM TTFT / TPOT regression measurement using real timing hooks
- Real swap / rollback / payload-load latency measurement
- Broader model coverage: Qwen/Llama-class fused attention variants
- Long-running multi-tenant production pilots
- GGUF/AWQ direct surgery

- **Production Hardening**:
    - [x] External Proxy Fallback for vLLM compatibility-risk mitigation.
    - [ ] 24-hour Soak Test (Final Production Candidate Gate).
- **24h Soak Pending:** 6-hour and 10K endurance tests passed, but final 24h persistent-route soak validation remains pending.
- **Two-route Mixed-Batch Validation Completed:** Phase 5-E-1 successfully demonstrated 0 route violations and verified safe isolation over 1000 dynamically routed requests across `__base__` and the Alpaca route. Broader 3+ route and worst-case alternation stress remain future hardening work.
- **Determinism Follow-up Completed:** Phase 5-F demonstrated 100.0% top-token logprob trace similarity and exact text match after a verified checksum rollback for the tested prompt under explicit route cleanup and vLLM cache reset.
- **Monkey-Patch Fragility:** The internal vLLM integration depends on vLLM V1 internals and may break across vLLM releases.
  - Internal vLLM plugin mode remains version-locked and controlled-validation-only.
  - External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
  - External Proxy Fallback trades VRAM efficiency and route density for operational stability.
- **Broader Model Coverage:** Validation beyond OPT-125M/Qwen2.5-class controlled tests remains future work.
- **SLA-Grade Serving:** Not ready for uncontrolled public enterprise traffic or SLA commitments.
- **1-GPU Multi-Tenant Scale:** Cannot yet serve hundreds of concurrent routes seamlessly without internal KV cache integration.

*For full details on our testing methodology and failure modes, read the [Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md).*

---

## 8. Quick Start

```bash
# 1. Run a capability transfer evaluation (e.g., Text-to-SQL)
python tests/bench_text_to_sql.py

# 2. Run the Hot-Swap Runtime external proxy with vLLM
VLLM_BACKEND_URL="http://localhost:8000/v1/completions" uvicorn neural_scalpel.serving.vllm_proxy:app --port 8080
```

---

## 9. Documentation & Reports
- **[Hot-Swap Runtime Production Readiness Report](docs/HOTSWAP_RUNTIME_PRODUCTION_READINESS_REPORT.md):** 🚀 *Read this first!* Contains the endurance test results, actual LoRA evaluations, and integration benchmarks.
- **[Production Readiness Criteria](docs/PRODUCTION_READINESS_CRITERIA.md):** Tracks remaining gates before constrained Production Candidate declaration.
- **[Performance Regression Report](docs/PERFORMANCE_REGRESSION_REPORT.md):** Coarse E2E throughput benchmark and pending precise latency work.
- **[Known Limitations](docs/KNOWN_LIMITATIONS.md):** Current runtime, benchmark, and deployment limitations.
- **[vLLM Internal Integration Design](docs/VLLM_INTERNAL_INTEGRATION_DESIGN.md):** Architectural design for Step 4B integration.
- **[Empirical Consistency Report](docs/LOGIC_CONSISTENCY_REPORT.md):** Details on mathematical evaluation metrics and failure modes.
- **[Project Vision & Roadmap](docs/RESEARCH_AND_COMMERCIAL_ROADMAP.md):** Our strategy for ML research validation and commercial diagnostic tools.
- **[Technical Report](TECHNICAL_REPORT.md):** Mathematical proofs and architecture overview.
- **[Security Policy](SECURITY.md):** Important security considerations and vulnerability reporting.
- **[Model License Policy](MODEL_LICENSE_POLICY.md):** Legal and licensing responsibility regarding derivative adapter works.
- **[Disclaimer](DISCLAIMER.md):** Experimental software disclaimer and lack of production guarantees.

---
*Developed and tested locally on an NVIDIA RTX 5060 Ti 16GB.*