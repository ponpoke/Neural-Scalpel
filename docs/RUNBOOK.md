# Neural-Scalpel Operational Runbook

## Overview

This runbook covers incident response for all known failure modes in the
Neural-Scalpel hot-swap runtime. Each scenario includes symptoms, diagnostics,
immediate response, recovery, and prevention.

---

## Scenario 1: Rollback Failure

### Symptoms
- `scalpel_worker_health` gauge drops to 0
- `scalpel_rollback_failure_total` increments
- Audit log contains `worker_quarantined` event
- All subsequent requests return HTTP 503

### Diagnostics
```bash
# Check worker health
curl -s localhost:8000/healthz | jq .

# Check audit log for the failure event
grep "worker_quarantined" /var/log/neural-scalpel/audit.jsonl | tail -5

# Check metrics
curl -s -H "X-Admin-Key: $ADMIN_KEY" localhost:8000/admin/metrics | grep rollback_failure
```

### Immediate Response
1. **DO NOT attempt to send more requests** — the worker's weight state is corrupted
2. Note the route_id and request_id from the audit log
3. Alert the on-call engineer

### Recovery
1. Restart the Neural-Scalpel process to reload clean base model weights
2. Quarantine the offending route in the registry
3. Verify recovery: `curl localhost:8000/healthz` → `{"status": "ok", "quarantined": false}`

### Prevention
- Validate all payload files before registration
- Run endurance tests on new routes before production deployment
- Monitor `scalpel_rollback_latency_seconds` for anomalies

---

## Scenario 2: Route Quarantine

### Symptoms
- Requests to a specific route return HTTP 403
- `scalpel_quarantine_total{scope="ROUTE"}` increments
- Audit log contains `route_quarantined` event

### Diagnostics
```bash
# List route statuses
curl -s localhost:8000/v1/routes | jq '.[] | select(.status == "QUARANTINED")'

# Check quarantine reason in audit
grep "route_quarantined" /var/log/neural-scalpel/audit.jsonl | tail -5
```

### Immediate Response
1. Other routes continue operating normally — no action needed for them
2. Investigate the quarantine reason in the audit log

### Recovery
1. Fix the root cause (corrupted payload, SHA mismatch, etc.)
2. Re-register the route with a corrected manifest
3. Verify: send a test request to the re-registered route

---

## Scenario 3: Payload SHA-256 Mismatch

### Symptoms
- Route registration or first request fails with `payload_sha256_mismatch`
- Route is automatically quarantined

### Diagnostics
```bash
# Verify payload hash manually
sha256sum /path/to/payload.safetensors

# Compare with manifest
cat /path/to/route.scalpel_route | jq '.payload.sha256'
```

### Immediate Response
1. The route is already quarantined — no further action needed
2. Do not manually override the quarantine

### Recovery
1. Re-generate the payload or fix the manifest SHA-256
2. Re-register the route

---

## Scenario 4: VRAM Leak Suspected

### Symptoms
- GPU memory usage grows steadily over time
- `nvidia-smi` shows increasing memory allocation
- No corresponding increase in request rate

### Diagnostics
```bash
# Monitor GPU memory
watch -n 5 nvidia-smi

# Check swap/rollback balance
curl -s -H "X-Admin-Key: $ADMIN_KEY" localhost:8000/admin/metrics | \
  grep -E "scalpel_swap_total|scalpel_rollback_total"
```

### Immediate Response
1. If memory growth exceeds 100MB after warmup, prepare for restart
2. Check that swap_total == rollback_total (every swap must rollback)

### Recovery
1. Restart the process to release all GPU memory
2. File a bug report with the metrics snapshot

---

## Scenario 5: vLLM Patch Mismatch

### Symptoms
- Startup self-test fails with `vllm_version` check failure
- Runtime errors from vLLM internal APIs

### Diagnostics
```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Compare with version lock
cat docs/VERSION_LOCK.md
```

### Immediate Response
1. Do not deploy with a mismatched vLLM version
2. Pin vLLM to the validated version

### Recovery
1. Install the correct vLLM version: `pip install vllm==<locked_version>`
2. Re-run startup self-test
3. **Fallback**: If the version cannot be downgraded, switch to External Proxy Fallback mode.
   - Internal vLLM plugin mode remains version-locked and controlled-validation-only.
   - External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
   - External Proxy Fallback trades VRAM efficiency and route density for operational stability.

---

## Scenario 6: Throughput Degradation

### Symptoms
- `scalpel_swap_latency_seconds` p99 increasing
- Request latency climbing
- Throughput (req/s) dropping

### Diagnostics
```bash
# Check swap latency trend
curl -s -H "X-Admin-Key: $ADMIN_KEY" localhost:8000/admin/metrics | \
  grep swap_latency

# Check for excessive route switching
curl -s -H "X-Admin-Key: $ADMIN_KEY" localhost:8000/admin/metrics | \
  grep scalpel_swap_total
```

### Immediate Response
1. Check if traffic pattern has become highly mixed-route
2. Consider traffic shaping to reduce route switching frequency

### Recovery
1. Group same-route requests to maximize batch coalescing
2. Consider dedicated instances for high-traffic routes

---

## Scenario 7: Tenant Abuse

### Symptoms
- Single tenant consuming disproportionate resources
- Rate limiter triggering frequently for one tenant
- `scalpel_requests_total{tenant="..."}` skewed

### Diagnostics
```bash
# Check per-tenant request rates
curl -s -H "X-Admin-Key: $ADMIN_KEY" localhost:8000/admin/metrics | \
  grep scalpel_requests_total
```

### Immediate Response
1. Reduce rate limit for the offending tenant
2. Monitor for continued abuse

### Recovery
1. Adjust per-tenant rate limits in configuration
2. Contact the tenant about usage policies

---

## Scenario 8: Soak Test Failure

### Symptoms
- `soak_vllm_scalpel.py` exits with code 1
- `swap_count != rollback_count`
- `mixed_batch_violation_count > 0`
- Worker health unavailable or unhealthy
- VRAM growth exceeds 100MB after warmup
- `errors > 0`

### Immediate Response
1. Do not promote the build to Production Candidate.
2. Preserve `reports/soak_24h.json`.
3. Preserve audit logs and benchmark output.
4. Reproduce with a shorter 1h soak.
5. Investigate payload validation, rollback path, route scheduling, and worker health state.

### Recovery
1. Fix the failing path.
2. Re-run the 10K endurance test.
3. Re-run the 6h extended soak.
4. Re-run the final 24h soak before Production Candidate declaration.

---

## Escalation Criteria

| Severity | Condition | Escalation |
|----------|-----------|------------|
| P0 (Critical) | Worker quarantined | Immediate page to on-call |
| P0 (Critical) | Route violation detected | Immediate page to on-call |
| P1 (High) | Multiple route quarantines in 1h | Alert to team channel |
| P2 (Medium) | Throughput degradation >50% | Alert to team channel |
| P3 (Low) | Single route quarantine | Log and investigate next business day |
