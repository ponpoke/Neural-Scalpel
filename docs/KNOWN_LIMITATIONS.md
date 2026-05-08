# Neural-Scalpel Known Limitations

## Architecture

### Route Isolation Throughput Trade-off
Route-homogeneous batching intentionally fragments mixed-route workloads into
separate micro-batches. This has shown zero observed cross-route weight contamination in controlled validations, but remains subject to future hardening and reduces aggregate throughput compared to vanilla vLLM under mixed workloads.

**Impact**: Highly workload-dependent. Controlled coarse benchmark results showed negligible base-route overhead, but substantial mixed-route throughput reduction under aggressive route alternation. Mixed-route degradation can exceed 40% depending on route-switching frequency, payload type, and batching fragmentation.

**Mitigation**: Traffic shaping to maximize same-route batch coalescing.

### Serial Swap Execution
Weight swaps are serialized behind a mutex lock. Concurrent swap requests
are queued, not parallelized. This ensures atomicity but limits swap throughput.

**Impact**: Swap latency adds to request latency for route-switching requests.
**Mitigation**: Route-aware scheduling minimizes unnecessary swaps.

### KV Cache Invalidation
When VRAM weights change during a route swap, all KV cache blocks generated
under the previous route must be invalidated. This prevents stale cache reuse
but increases cold-start latency for the new route.

**Impact**: First request after route switch has higher TTFT.
**Mitigation**: Prefix caching with route-identity-aware hash keys.

## Operational

### vLLM Version Dependency & Fallback
- Internal vLLM plugin mode remains version-locked and controlled-validation-only.
- **Mitigation**: Neural-Scalpel uses strict version locking (`VERSION_LOCK.md`) and a "Fail-Closed" startup policy. In environments where internal patching is unsupported or risky, **External Proxy Fallback** is now available as a safer operational path.
- External Proxy Fallback trades VRAM efficiency and route density for operational stability.

### 24h Soak Pending

The runtime has passed 10K endurance and 6-hour extended soak tests. The final 24h persistent-route soak test remains pending. Until this test passes, Neural-Scalpel is best described as a validated prototype with strong controlled runtime evidence, and a paradigm-shift-class candidate under controlled validation.


### Coarse Benchmark Limitation

The current performance benchmark measures coarse E2E throughput. TTFT, TPOT, swap latency, rollback latency, and payload-load latency require precise timing hooks before being used as production-grade latency evidence.

### No Hot-Reload of Route Manifests
Routes must be registered via the API. File system changes to `.scalpel_route`
manifests are not automatically detected.

### No Graceful Degradation on Worker Quarantine
When a worker is quarantined (rollback failure), it cannot recover without
process restart. This is intentional: a corrupted weight state is unrecoverable.

### Audit Log Retention
The JSON-L audit log grows without bound. External log rotation (logrotate)
must be configured by the operator.

### No Built-in TLS
The API server does not implement TLS directly. A reverse proxy (nginx, envoy)
must terminate TLS in front of the Neural-Scalpel service.

## Model Support

### Fused QKV Architectures
Models with fused QKV projections (e.g., GPT-NeoX, Phi) require special
handling in the layer discovery module. Support is experimental.

### Quantized Models
INT8/INT4 quantized models are not validated. Weight delta arithmetic
on quantized tensors may produce incorrect results.

### Very Large Models
Models exceeding single-GPU VRAM capacity (tensor parallel, pipeline parallel)
are not supported. The hot-swap mechanism operates on a single weight copy.

## Security

### HMAC-SHA256 Signing
Route manifests use HMAC-SHA256 for signing. This is suitable for internal
deployments but should be upgraded to Ed25519/RSA for public-facing systems.

### JWT Algorithm
Only HS256 JWT tokens are validated. RS256/ES256 support requires
integration with a proper JWKS endpoint.

## Phase 5 Limitations

- **Single-Prompt Benchmark**: Early Phase 5-C results were from a single-prompt benchmark. Phase 5-D expanded this to 50 prompts, but throughput results should still not be generalized to all workloads.

### Text-level Determinism

Earlier Phase 5-C runs observed text-level mismatch after rollback, likely due to vLLM batching/cache behavior. Phase 5-F later passed for the tested prompt under an explicit route cleanup and vLLM cache-reset condition: Base-before and Base-after text matched exactly, and top-token logprob trace similarity reached 100.0%.

Remaining limitation:
This result is validated for the tested prompt/cache-reset condition. Broader prompt sets, batch shapes, and vLLM versions remain future hardening work.

### Multi-route Transitions

Phase 5-E-1 passed two-route mixed-batch validation (`__base__` ↔ Alpaca, 1000 requests, 0 violations). 3+ route mixed-batch and worst-case alternating-route stress remain pending.
