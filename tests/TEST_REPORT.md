# Neural-Scalpel Ecosystem: Quality Assurance & Validation Report

**Verification Status:** `CORE MATH + STRUCTURAL PROJECTION VERIFIED; CONTROLLED RUNTIME TESTS PASSED (Research Preview)`  
**Environment:** Local CI (Windows/CPU + CUDA)  
**Version Target:** v1.0.0-alpha  
**Test Framework:** unittest discovery (`python -m unittest discover tests`)
**Total Non-Live Tests:** 200+ passed

> Live vLLM integration tests are excluded from the default suite and run separately in a controlled GPU environment.
> This report verifies mathematical, structural, and controlled-runtime behavior under test conditions. It does not claim arbitrary cross-model intelligence transfer or production SLA readiness.

This report details the test suite executed to validate the core mathematical components and structural projection helpers of the Neural-Scalpel Ecosystem.

## Transparency & Verification Scope

To ensure technical transparency, it is critical to define what this test suite proves:

1. **Mathematically Verified Core (Unit Tests):** The fundamental algorithms—such as physical dimension projection, AVPS, Head-wise Procrustes alignment, WDR, JTSA, and HAMA—execute actual PyTorch tensor operations and pass rigorous structural and mathematical checks. **This proves the mathematical components function as designed, but does not guarantee arbitrary cross-model intelligence transfer.**
2. **Structural Projection Baseline v2:** The Qwen2.5 cross-scale structural projection helpers are covered by focused unit tests for GQA-aware target-shape inference, interpolated layer mapping, SVD recompression statistics, fused Qwen tensor construction, and strict unexpected-tensor rejection. These tests verify structural correctness and regression safety, but do **not** prove SQL/Coding behavioral improvement.
3. **Framework Logic & Infrastructure:** I/O bridges (Safetensors, GGUF, AWQ), the CLI pipeline, Router signature verification, and VRAM Hot-Swap all execute real code paths with mock data. 
4. **E2E & Logic Tests:** Preliminary semantic logic testing (KL divergence checks) and end-to-end benchmarks are executed on heavily localized, small-scale mock environments to simulate behavior.

---

## Test Suite Overview

> The table below lists representative test files. The full non-live suite contains 200+ passing tests across structural projection, runtime, serving, route schema, failure-mode, and vLLM mock/integration modules.

### Test Files
| File | Tests | Scope |
| :--- | :---: | :--- |
| `test_structural_projection.py` | 7 | **Structural Projection Baseline v2**: GQA target-shape inference, interpolated layer mapping, SVD recompression, Qwen fused tensor shapes, strict unexpected-tensor failure |
| `test_comprehensive.py` | 76 | Full coverage of modules: math (WDR/Sinkhorn stability), adapters, I/O, router, hot-swap, CLI |
| `test_neural_scalpel.py` | 30 | Phase-structured validation (WDR soft-routing semantics verified) |
| `test_scalpel_kernel.py` | 2 | VRAM Hot-Swap CUDA synchronization and basic atomic pointer tests |
| `test_semantic_logic.py` | 2 | KL divergence bounds and HAMA vs JTSA logic stabilization |
| `test_evaluate_e2e_real.py`| 3 | Small-scale simulation of PPL evaluation and logic verification |
| `test_io_bridge.py` | 2 | GGUF/Safetensors read/write cycle |
| `test_calibration_manifold.py` | 2 | AWQ LMR error reduction, AWQ Bridge integration |

---

## Part 1: Core Math Engine (Unit Tests)

| Algorithm | Key Assertion | Result |
| :--- | :--- | :---: |
| **Head-wise Orthogonal Procrustes** | Cosine Similarity ≥ 0.99 for identity case | ✅ PASS |
| **Sparse Task Vector** | CSR format produced; boundary conditions caught | ✅ PASS |
| **Adaptive rSVD Bootstrap** | Rank-5 matrix recovered with < 5% relative error | ✅ PASS |
| **AVPS** | ≥ 98% L2 energy preserved after sparsification | ✅ PASS |
| **PCSI** | Correct output shape with graceful SVD component handling | ✅ PASS |
| **Soft-Routing Head Pooling** | Output head count matches target | ✅ PASS |
| **Wasserstein Discrete Routing / Log-domain Sinkhorn** | Stable square/rectangular Sinkhorn marginals; soft/hard WDR column sums; invalid epsilon/mode rejection; fp16 finite output | ✅ PASS |
| **JTSA / HAMA** | Jacobian/Hessian values in valid range; stable under small variance | ✅ PASS |
| **KL Divergence (Semantic)** | KL Div bounds maintained within strict mathematical thresholds | ✅ PASS |

## Part 1-B: Structural Projection Baseline v2

| Component | Key Assertion | Result |
| :--- | :--- | :---: |
| **GQA-aware Target Shape Inference** | Q/O, K/V, MLP projection shapes match Qwen2.5-0.5B-style configuration | ✅ PASS |
| **Interpolated Layer Mapping** | 28 source layers are mapped into 24 target layers with valid endpoints and alpha range | ✅ PASS |
| **SVD Recompression** | Resized tensors match target shape; energy retention remains bounded in [0, 1]; higher rank retains no less energy | ✅ PASS |
| **Qwen Fused Tensor Construction** | `q_proj/k_proj/v_proj` fuse into `qkv_proj`; `gate_proj/up_proj` fuse into `gate_up_proj` with expected GQA-aware shapes | ✅ PASS |
| **Strict Shape Verification Guard** | Unexpected payload tensors fail verification rather than being silently accepted | ✅ PASS |
| **Metadata Humility Guard** | Projection metadata keeps behavioral validation marked as `PENDING` | ✅ PASS |

These tests validate the structural projection helpers and metadata safety guarantees. They do **not** validate downstream SQL/Coding quality, long-form generation stability, or Neural-Scalpel runtime route application.

## Part 1-C: Behavioral Alignment Core Migration (Phase 5-G Hardened)

| Component | Key Assertion | Result |
| :--- | :--- | :---: |
| **PairedActivationDataset** | Matching sample counts accepted; mismatched counts rejected | ✅ PASS |
| **Hardened align API** | Prompts/Layers validation enforced; auto-correspondence integrated | ✅ PASS |
| **Mapping & Mapping-to-Delta** | Explicit module-to-delta mapping solves correctly | ✅ PASS |
| **PEFT Key Abstraction** | Custom prefixes and adapter names generated correctly | ✅ PASS |
| **validate_behavior Guards** | Status enum classification (BEHAVIORAL_SHIFT_DETECTED, etc.) verified | ✅ PASS |

These tests validate the Hardened Core Migration API (v1.1.0). They prove the system is robust against common failure modes (missing data, malformed shapes, numerical instability) but do not replace task-level evaluation.

## Part 2: Architecture Adapters (Unit Tests)

| Adapter | Key Assertion | Result |
| :--- | :--- | :---: |
| **LLaMA-3 → Qwen-2** | Dimensional projection applied successfully via SRHP or WDR | ✅ PASS |
| **SDXL → SDXL** | Tensor unchanged (passthrough identity) | ✅ PASS |
| **SDXL → FLUX** | Subspace injection produces correct shape mapping | ✅ PASS |
| **Adapter Factory** | Correct adapter class returned per architecture pair | ✅ PASS |

## Part 3: I/O Bridges & Hardware Hot-Swap (Unit Tests)

| Feature | Key Assertion | Result |
| :--- | :--- | :---: |
| **Safetensors / GGUF Bridge** | Round-trip data integrity; streaming yields all keys | ✅ PASS |
| **AWQ Bridge (Hybrid)** | Packed INT4 tensor produced; scales generated via Calibration data | ✅ PASS |
| **VRAM Hot-Swap (Sync)** | CUDA synchronization lock properly blocks asynchronous reads | ✅ PASS |
| **PPL Gateway** | Automatic rollback triggered if PPL ratio exceeds threshold | ✅ PASS |

Controlled soak validations were executed separately from the default non-live suite.

- 1h extended soak: ✅ PASS
- 6-hour mixed-route extended soak: ✅ PASS, 1,956,000 requests, 1,114,920 swaps/rollbacks, 0 violations, 0 errors, 0.0MB VRAM growth
- Final 24h mixed-route soak: ⏳ PENDING

---

## Phase 4-A Real-LoRA Route Smoke Check

| Validation | Status | Result |
|---|---:|---|
| Qwen2.5-0.5B load smoke | ✅ PASS | vLLM engine initialized |
| Real-LoRA payload conversion | ✅ PASS | 96 fused tensors generated for vLLM `gate_up_proj` / `qkv_proj` representation |
| Qualitative route smoke check | ✅ PASS | Observable base-vs-Alpaca route output differences |
| Native LoRA throughput baseline | ✅ COLLECTED | Base 3501.4 tok/s, Native LoRA 1968.25 tok/s, -43.79%; not Neural-Scalpel swap overhead |

## Phase 4-B Preliminary Quantitative Smoke Evaluation

| Item | Status | Result |
|---|---:|---|
| Evaluation harness | ✅ Implemented | Transformers track with un-fused vLLM payload |
| Rollback consistency | ✅ PASS | Base score 0.6447, rollback score 0.6447 (Perfect match) |
| Behavior change | ✅ Observed | Alpaca route output differed from base |
| Task improvement | ❌ Not proven | Alpaca route score 0.5805 < base 0.6447 under keyword-overlap metric |

This result confirms functional application and rollback consistency, but does not establish dataset-level task improvement.

## Conclusion

**200+ non-live tests passed.**

The Neural-Scalpel ecosystem has passed its mathematical and structural unit tests. The core linear and non-linear approximations (JTSA/HAMA) function according to their mathematical specifications under test conditions. 

The Structural Projection Baseline v2 is verified as a structural and format-compatibility baseline. The newer Paired Behavioral Alignment scaffold has demonstrated preliminary runtime behavioral shifts in the Qwen2.5 SQL case study, but task-level SQL capability transfer remains unverified and requires larger parse/execution evaluations.

*Reminder: While the components are structurally verified, the quality of real-world cross-architecture projection depends significantly on architectural homology and calibration data.*

---

## Live vLLM Validation

| Validation | Status | Result |
|---|---:|---|
| Latest-branch 10K endurance rerun | ✅ PASS | 10,000 requests, 896 swaps, 896 rollbacks, 0 violations, VRAM stable |
| 6-hour mixed-route extended soak | ✅ PASS | 1,956,000 requests, 1,114,920 swaps/rollbacks, 0 violations, 0 errors, VRAM growth 0.0MB |
| Phase 5-C Route-Window Benchmark | ✅ PASS | swap_count=1 / 1600 tokens; verified_rollbacks=1 |
| Checksum-level rollback | ✅ PASS | verified_rollbacks=1; bit-exact restoration proven |
| Text-level exact match | ⚠️ Environment-sensitive | Exact match may vary without explicit cache reset; checksum-level rollback remains the primary integrity evidence. |
| Final 24h persistent-route soak | ⏳ Pending | Required before constrained Production Candidate declaration |

---
*Verified via `PYTHONPATH=. python -m unittest discover tests`.*  
*Default suite excludes live vLLM GPU tests, which are executed separately in a controlled Linux/GPU environment.*  
*Latest live validation includes 10K endurance rerun and 6-hour mixed-route extended soak.*  
*Last updated: May 2026*