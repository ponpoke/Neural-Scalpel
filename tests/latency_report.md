# Hot-Swap Runtime Latency Report

## Configuration
- **gpu**: NVIDIA GeForce RTX 5060 Ti
- **torch_version**: 2.11.0+cu130
- **cuda_version**: 13.0
- **model**: Mock-Qwen2.5-0.5B-Layers
- **route_count**: 1
- **num_runs**: 500
- **precision**: fp16
- **vram_peak_mb**: 17.125

## Metrics (in ms, except tokens_per_sec)
| Metric | Mean | P50 | P90 | P95 | P99 | Max |
|---|---|---|---|---|---|---|
| end_to_end_latency | 4.08 | 3.77 | 4.09 | 4.29 | 4.97 | 138.88 |
| lock_wait_time | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| swap_latency | 1.83 | 1.75 | 1.97 | 2.04 | 2.30 | 25.47 |
| rollback_latency | 0.05 | 0.04 | 0.08 | 0.12 | 0.20 | 0.28 |
| swap_plus_rollback_latency | 1.88 | 1.80 | 2.04 | 2.10 | 2.35 | 25.60 |
| ttft | 0.31 | 0.08 | 0.13 | 0.16 | 0.25 | 110.86 |
| tokens_per_sec | 57428.86 | 60006.00 | 68080.47 | 70303.72 | 71737.81 | 74074.07 |
