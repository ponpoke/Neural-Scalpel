# Phase 5-C Benchmark Report

## Status

**Strong Positive Result — Production Candidate Gates Pending**

## Summary

Phase 5-C confirmed that route-window persistent swapping removes the per-token swap/rollback bottleneck observed in Phase 5-B. The latest run recorded one confirmed swap and one verified rollback across 1600 generated tokens.

## Results

| Metric | Value |
|---|---:|
| Base throughput | 2404.18 tok/s |
| Scalpel route throughput | 4086.68 tok/s |
| Throughput delta vs base | +69.98% |
| Generated tokens | 1600 |
| Swap count | 1 |
| Rollback count | 1 |
| Verified rollbacks | 1 |
| Swaps/token | 0.000625 |
| Swap latency p50/p99 | 79.96 / 79.96 ms |
| Rollback latency p50/p99 | 9.64 / 9.64 ms |
| Text exact match | false |

## Interpretation

The +69.98% throughput delta should not be interpreted as universal base-model outperformance. In this run, the routed output was highly repetitive, which likely reduced decoding complexity relative to the base output. The key validated result is that swap overhead was reduced to one confirmed route-window swap across 1600 generated tokens.

## Safety

Checksum-level rollback verification passed with `verified_rollbacks=1`. Text-level exact-match verification did not pass in this run, so output-level determinism remains pending.

## Conclusion

Phase 5-C was a strong positive performance result. At the time, repeated benchmark median, multi-route validation, and determinism follow-up were pending. These were later advanced in Phase 5-D, Phase 5-E-1, and Phase 5-F.

Current remaining Production Candidate gate: 24h persistent-route soak. Broader 3+ route and worst-case alternation stress remain future hardening work.

> Update after Phase 5-F:
> The Phase 5-C run reported `text exact match = false`. This was later followed up in Phase 5-F using explicit route cleanup and vLLM cache reset. Under that tested condition, Base-before and Base-after text matched exactly, and top-token logprob trace similarity reached 100.0%. Phase 5-C remains a performance benchmark; Phase 5-F is the determinism follow-up.
