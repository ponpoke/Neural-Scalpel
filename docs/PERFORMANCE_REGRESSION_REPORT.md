# Neural-Scalpel Performance Regression Report

Generated: May 2026  
Status: Coarse E2E throughput benchmark + Phase 5-D repeated median validation

> This is not yet a precise TTFT/TPOT regression report. TTFT, TPOT, swap latency, rollback latency, and payload-load latency are approximate placeholders or pending real timing hooks in the current benchmark workflow.

## Phase 5-D: Repeated Median Benchmark (50 Prompts, 3 Runs)

To prove generalizability and stability, a repeated benchmark was executed using a 50-prompt subset of the Alpaca evaluation dataset. The results represent the median across 3 independent runs.

| Configuration | Throughput (tok/s) | StDev | Median VRAM (MB) |
| :--- | :--- | :--- | :--- |
| A: Vanilla vLLM Base | ~3813.84 | ± 158.67 | 13708 |
| B: Neural-Scalpel v2 | ~2574.31 | ± 176.09 | 14374 |
| C: vLLM Native LoRA | ~983.32 | ± 51.65 | 13688 |

**Analysis:**
- Scalpel outperformed vLLM Native LoRA by **+161.80%** under these controlled conditions.
- Scalpel throughput overhead vs. Base is **-32.50%**.
- Native LoRA throughput overhead vs. Base is **-74.22%**.
- Route application (`swap_count > 0`) and rollback verification (`verified_rollbacks > 0`) were strictly enforced and passed unconditionally across all Scalpel runs. At least one checksum-verified rollback event was recorded in every Scalpel run; full per-rollback verification coverage remains a future hardening metric.

## Coarse E2E Throughput Results

| Config | tok/s | Interpretation |
|---|---:|---|
| A_vanilla | 19,837 | Vanilla vLLM baseline |
| B_base_route | 19,757 | Scalpel patches enabled, base route only |
| C_simulated_routes | 9,064 | Simulated route swap path |
| D_safetensors_routes | 6,333 | Real safetensors payload route |
| E_mixed_route | 3,633 | Mixed-route route-homogeneous isolation |

## Initial Observations

- Base-route overhead was negligible in this run: approximately 0.4% throughput reduction from A to B.
- Simulated, safetensors, and mixed-route workloads show expected overhead from route control, payload validation, swap/rollback, and route-homogeneous scheduling.
- Mixed-route throughput reduction is an intentional safety trade-off for route isolation.

## Endurance Cross-Check

Latest-branch 10K endurance rerun passed:

- Requests: 10,000
- Forwards: 1,344
- Swaps / rollbacks: 896 / 896
- Violations: 0
- Throughput: 400.09 req/s / 12,802.72 tok/s
- VRAM after init / peak / end: 15166.0MB / 15166.0MB / 15166.0MB

6-hour extended soak passed:

- Requests: 1,956,000
- Batches: 19,560
- Swaps / rollbacks: 1,114,920 / 1,114,920
- Violations: 0
- Errors: 0
- VRAM reserved growth: 0.0MB
- VRAM allocated growth: 0.0MB

## Pending Precise Regression Work

- [ ] TTFT p50/p99 measurement using real timing hooks
- [ ] TPOT p50/p99 measurement using real timing hooks
- [ ] Real swap latency measurement
- [ ] Real rollback latency measurement
- [ ] Payload-load latency measurement
- [ ] GPU utilization and VRAM timeline integration

## Safety / Determinism Cross-Checks

- Phase 5-E-1 two-route mixed-batch safety validation passed with 1000 requests, 0 route violations, 0 quarantine events, and a healthy worker.
- Phase 5-F determinism follow-up passed under the tested cache-reset condition, with exact text match and 100.0% top-token logprob trace similarity.
