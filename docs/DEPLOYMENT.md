# Neural-Scalpel Deployment Guide

## Prerequisites

- Docker 24+ with NVIDIA Container Toolkit
- NVIDIA GPU driver ≥525
- CUDA 11.8 or 12.1

## Quick Start (Docker Compose)

```bash
# Clone and enter the repository
git clone https://github.com/ponpoke/Neural-Scalpel.git
cd Neural-Scalpel

# Set environment variables
export JWT_SECRET="your-production-secret-key-32-bytes"
export ADMIN_API_KEY="your-admin-api-key"

# Start the service
docker compose -f docker-compose.vllm-scalpel.yml up -d

# Wait for startup self-test
docker compose -f docker-compose.vllm-scalpel.yml logs -f scalpel

# Verify health
curl http://localhost:8000/healthz
```

## Docker Image Build

```bash
docker build -f Dockerfile.vllm-scalpel -t neural-scalpel:latest .
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET` | Yes | — | Secret key for JWT token verification |
| `ADMIN_API_KEY` | Yes | — | API key for admin endpoints |
| `VLLM_BACKEND_URL` | No | Mock mode | URL of the vLLM backend |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-0.5B` | HuggingFace model identifier |
| `ROUTE_REGISTRY_DIR` | No | `/data/routes` | Route manifest storage |
| `PAYLOAD_DIR` | No | `/data/payloads` | Safetensors payload storage |
| `AUDIT_LOG_PATH` | No | `/var/log/neural-scalpel/audit.jsonl` | Audit log file path |
| `MAX_PAYLOAD_BYTES` | No | `2147483648` | Maximum payload file size |

### Volume Mounts

```yaml
volumes:
  - ./routes:/data/routes:ro       # Route manifests
  - ./payloads:/data/payloads:ro   # Safetensors payloads
  - ./logs:/var/log/neural-scalpel # Audit logs
```

## Startup Self-Test

The service performs the following checks before accepting traffic:

1. **GPU availability**: Verifies CUDA device is accessible
2. **vLLM version**: Checks compatibility with locked version
3. **Route registry**: Validates storage directory is readable
4. **Payload storage**: Validates payload directory is readable
5. **Dry-run swap**: Performs a test swap/rollback cycle

The `/healthz` endpoint returns `{"status": "ok"}` only after all checks pass.

Current status: Validated prototype with strong controlled runtime evidence. Formal Production Candidate declaration remains pending the final 24h persistent-route soak.

## Production Checklist

- [ ] Set strong, unique `JWT_SECRET` (≥32 bytes)
- [ ] Set unique `ADMIN_API_KEY`
- [ ] Configure TLS termination (nginx/envoy in front)
- [ ] Mount route and payload directories as read-only
- [ ] Configure log rotation for audit logs
- [ ] Set up Prometheus scraping for `/admin/metrics`
- [ ] Import Grafana dashboard
- [ ] Configure alerting rules (see OBSERVABILITY.md)
- [ ] Run startup self-test in staging before production
- [ ] Verify model compatibility with discovery tool
- [x] Run latest-branch 10K endurance test in controlled validation
- [x] Run 6-hour extended soak test in controlled validation
- [x] Run Phase 5-D repeated median benchmark in controlled validation
- [x] Run Phase 5-E-1 two-route mixed-batch validation in controlled validation
- [x] Run Phase 5-F determinism follow-up under tested cache-reset condition
- [ ] Run final 24h mixed-route soak test with `--require-worker-health`

## Hardening Checklist

- [ ] Run 3+ route mixed-batch validation
- [ ] Run worst-case alternating route stress validation
- [ ] Validate broader model/vLLM version coverage

## Monitoring

### Health Check

```bash
curl http://localhost:8000/healthz
# Expected: {"status": "ok", "quarantined": false}
```

### Prometheus Metrics

```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" http://localhost:8000/admin/metrics
```

## Scaling

Neural-Scalpel operates as a single-GPU, single-process service.
For horizontal scaling, deploy multiple instances behind a load balancer
with route-affinity (sticky sessions by route_id) to minimize swap overhead.
