# Neural-Scalpel Observability Guide

## Prometheus Metrics

All metrics are exported at the `/admin/metrics` endpoint (requires admin API key).

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `scalpel_requests_total` | Counter | tenant, route, status | Total inference requests |
| `scalpel_forward_total` | Counter | route | Total forward passes |

### Swap / Rollback Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `scalpel_swap_total` | Counter | route | Weight swaps performed |
| `scalpel_rollback_total` | Counter | route | Rollbacks performed |
| `scalpel_rollback_failure_total` | Counter | route | Rollback failures (CRITICAL) |
| `scalpel_swap_latency_seconds` | Histogram | — | Swap operation latency |
| `scalpel_rollback_latency_seconds` | Histogram | — | Rollback operation latency |
| `scalpel_payload_load_latency_seconds` | Histogram | — | Payload file load latency |

### Safety Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `scalpel_route_violation_total` | Counter | — | Route isolation violations |
| `scalpel_quarantine_total` | Counter | scope, reason | Quarantine events |
| `scalpel_scheduler_shelving_total` | Counter | — | Scheduler shelving events |

### System Gauges

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `scalpel_active_route` | Gauge | route | Currently active VRAM route |
| `scalpel_worker_health` | Gauge | — | 1=healthy, 0=quarantined |

## Structured Audit Log

All runtime events are written to a JSON-L audit log file.

### Event Types

```
request_received      → New inference request accepted
route_verified        → Route manifest validated against registry
payload_loaded        → Safetensors file loaded into memory
payload_hash_verified → SHA-256 integrity check passed
snapshot_captured     → Base model weights snapshotted
swap_started          → Weight delta application begun
swap_completed        → Weight delta application finished
forward_started       → Model forward pass begun
forward_completed     → Model forward pass finished
rollback_started      → Weight rollback begun
rollback_completed    → Weight rollback finished (checksum verified)
route_quarantined     → Route placed in quarantine
worker_quarantined    → Worker placed in quarantine (CRITICAL)
```

### Log Format

Each line is a valid JSON object:

```json
{
  "timestamp": "2026-05-05T01:00:00.000000+00:00",
  "request_id": "req-abc123def456",
  "tenant_id": "tenant_alpha",
  "route_id": "qwen_sql_v3",
  "event": "swap_completed",
  "status": "success",
  "latency_ms": 12.45,
  "route_version": "1.0.0",
  "route_sha256": "abc123..."
}
```

## Alerting Recommendations

### Critical Alerts (PagerDuty)

| Condition | Action |
|-----------|--------|
| `scalpel_worker_health == 0` | Worker quarantined — restart required |
| `scalpel_rollback_failure_total > 0` | Weight corruption risk |
| `scalpel_route_violation_total > 0` | Route isolation breach |

### Warning Alerts

| Condition | Action |
|-----------|--------|
| `scalpel_swap_latency_seconds{quantile="0.99"} > 1.0` | Swap performance degradation |
| `scalpel_quarantine_total increasing` | Route quality issue |
| `rate(scalpel_requests_total{status="rejected"}[5m]) > 10` | High rejection rate |

## Soak Test Validation Metrics

During long-duration soak validation, the following conditions must hold:

- `swap_count == rollback_count`
- `mixed_batch_violation_count == 0`
- `worker_health == healthy`
- `errors == 0`
- VRAM growth after warmup <= 100MB

The 24h soak test should be run with `--require-worker-health` so that unavailable worker health state is treated as a failure.

Latest controlled validation:
- 6-hour mixed-route extended soak passed
- 1,956,000 requests
- 1,114,920 swaps / 1,114,920 rollbacks (all rollbacks checksum-verified)
- 0 violations
- 0 errors
- 0.0MB VRAM growth

Additional controlled validation:
- Phase 5-D repeated median benchmark passed across 50 prompts × 3 runs
- Phase 5-E-1 two-route mixed-batch validation passed with 1000 requests, 0 route violations, 0 quarantine events, and healthy worker state
- Phase 5-F determinism follow-up passed under tested cache-reset condition, with exact text match and 100.0% top-token logprob trace similarity

## Grafana Dashboard

Import `dashboards/grafana_scalpel_runtime.json` into Grafana for pre-built panels:
- Request rate by route and status
- Swap/rollback latency percentiles
- Quarantine events timeline
- Worker health status
- VRAM utilization (requires node_exporter)
