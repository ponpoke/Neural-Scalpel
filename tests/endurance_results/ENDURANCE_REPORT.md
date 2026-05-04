# Real Model Endurance Benchmark Report

**Model:** Qwen/Qwen2.5-0.5B | **Device:** cuda

| Routes | Requests | Success | Leakage | RB Fail | E2E p99 | Swap p99 | PPL Delta | VRAM Peak | Leak |
|--------|----------|---------|---------|---------|---------|----------|-----------|-----------|------|
| 2 | 1000 | 1000 | 0 | 0 | 53.0ms | 4.3ms | 0.000000 | 1010MB | 36.2MB |
| 10 | 5000 | 5000 | 0 | 0 | 38.7ms | 4.4ms | 0.000000 | 1010MB | 3.8MB |
| 50 | 10000 | 10000 | 0 | 0 | 36.6ms | 4.3ms | 0.000000 | 1010MB | 3.8MB |

*Generated: 2026-05-05 05:42*
