# Neural-Scalpel Production Readiness Criteria

## Status: Validated Prototype with Strong Controlled Runtime Evidence

Neural-Scalpel has achieved strong controlled runtime evidence and is best described as a validated prototype / paradigm-shift-class candidate under controlled validation. Most implementation, hardening, documentation, and endurance tests are complete. Phase 5-C and Phase 5-D provide controlled evidence that route-window persistent swapping removes the Phase 5-B per-token swap bottleneck under the tested Qwen2.5-0.5B / Alpaca workloads, and that the result is stable across the tested 50-prompt repeated benchmark. The primary remaining gate for formal "Production Candidate" status is 24h persistent-route soak validation. Phase 5-F has completed the text-level and top-token logprob trace determinism follow-up under the tested cache-reset condition.

## Terminology

| Term | Definition |
|------|-----------|
| **Production Candidate** | All known failure modes are handled with fail-close behavior. Monitoring, recovery, and performance baselines are documented. Limited pilot deployment is safe. |
| **Production Ready** | Production Candidate + multi-week real-traffic validation + SLA commitment. |

## Gate Criteria

### G1: Failure-Mode Hardening

- [x] Corrupted safetensors payloads are rejected before any swap occurs
- [x] SHA-256 mismatch triggers route quarantine
- [x] Missing/mismatched tensor keys, shapes, and dtypes are rejected pre-swap
- [x] NaN/Inf values in payload tensors are detected and rejected
- [x] Swap failures trigger immediate rollback
- [x] Rollback failures trigger worker quarantine
- [x] Worker quarantine blocks all future requests
- [x] healthz reports unhealthy after worker quarantine

### G2: Performance Regression

- [x] Coarse E2E throughput smoke benchmark exists
- [x] Precise E2E latency benchmark script exists
- [x] Base-route E2E overhead measured as negligible in latest run
- [x] Internal swap latency p50 measured
- [x] Internal rollback latency p50 measured
- [x] Payload validation latency p50 measured
- [ ] Precise TTFT p50/p90/p99 measured
- [ ] Precise TPOT p50/p90/p99 measured
- [ ] Payload-load latency measured
- [x] Existing-method comparison draft exists
- [x] vLLM Native LoRA direct benchmarks completed (earlier refined run: -43.79%; Phase 5-D median: ~983 tok/s under controlled 50-prompt rerun)
- [x] Model reload completed (p50: 7.55s)
- [x] Multi-instance VRAM scaling estimate completed from measured single-instance VRAM

### G3: Compatibility Matrix

- [ ] At least 2 model architectures fully live-tested for Production Candidate
- [x] Automatic layer discovery implemented and tested
- [x] Each model classified as supported/experimental/unsupported
- [x] Layer mapping templates documented

### G4: KV Cache Safety

- [x] Same prompt + different route: no KV cache reuse
- [x] Route identity included in prefix cache hash
- [ ] Request abort cleans route state
- [x] Finished request releases route state

### G5: Observability

- [x] Prometheus metrics documented for critical operations
- [x] JSON-L structured audit log documented for runtime events
- [x] Grafana dashboard template available

### G6: Security

- [x] JWT token validation documented
- [x] Admin endpoints protected by API key
- [x] Payload path traversal prevention documented
- [x] Payload size limits enforced/documented
- [x] Per-tenant rate limiting documented

### G7: Deployment

- [x] Dockerfile with reproducible builds
- [x] docker-compose for local development
- [x] Startup self-test documented
- [x] Version lock document specifying exact dependency versions

### G8: Operations

- [x] Runbook covering critical failure scenarios
- [x] Deployment guide with step-by-step instructions
- [x] Known limitations documented

### G9: Endurance

- [x] Latest-branch 10K request endurance test passes with real safetensors
- [x] 6-hour mixed-route extended soak passes with `--require-worker-health`
- [ ] 24h mixed-route soak test passes with `--require-worker-health`
- [x] VRAM growth threshold check implemented in soak script
- [x] VRAM growth <100MB confirmed in 6-hour soak
- [ ] VRAM growth <100MB confirmed in 24h soak

### G10: Task Evaluation

- [x] Minimal real-payload route E2E check implemented
- [x] Runtime swap/rollback counters verified in real-payload E2E check
- [x] Route violations confirmed as 0 in real-payload E2E check
- [x] Base before/after deterministic output match observed
- [x] Phase 4-A qualitative real-LoRA route smoke check passed (Alpaca)
- [x] Phase 4-B preliminary quantitative smoke evaluation passed
- [ ] Trained task-specific payload evaluated

### G11: Phase 5-C to 5-F Route-Window Runtime Validation

- [x] Route-window persistent swap implemented
- [x] Route application recorded in audit log (`swap_count > 0`)
- [x] Checksum-level rollback verification passed (`verified_rollbacks > 0`)
- [x] Phase 5-B per-token swap bottleneck removed under tested route-window workloads
- [x] Text-level and top-token logprob trace determinism verified under tested cache-reset condition
- [x] Repeated benchmark median across 3–5 runs collected
- [x] Two-route mixed-batch transition validation completed (`__base__` ↔ Alpaca, 1000 requests, 0 violations, 0 quarantine)
- [x] 3+ route mixed-batch transition validation completed
- [x] Worst-case alternating route transition stress completed
- [ ] 24h persistent-route soak completed

## Scope Limitations

This certification explicitly does NOT cover:
- Arbitrary vLLM versions (only validated version)
- Arbitrary model architectures (only compatibility matrix models)
- Multi-node/multi-GPU serving
- Streaming response support
- SLA guarantees without operator-controlled deployment

### vLLM Version Dependency & Fallback
- Internal vLLM plugin mode remains version-locked and controlled-validation-only.
- External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
- External Proxy Fallback trades VRAM efficiency and route density for operational stability.
