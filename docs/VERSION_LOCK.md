# Neural-Scalpel Version Lock

## Validated Dependency Matrix

The following versions have been tested together and form the Production Candidate baseline.

### Core Runtime

| Package | Validated Version | Minimum | Notes |
|---------|-------------------|---------|-------|
| Python | 3.12.3 | 3.9 | Tested |
| PyTorch | 2.4.0+cu121 | 2.0.0 | CUDA 12.1 build |
| safetensors | 0.4.5 | 0.3.0 | Payload format |
| pydantic | 2.x | 1.10 | API schemas |

### Serving

| Package | Validated Version | Minimum | Notes |
|---------|-------------------|---------|-------|
| FastAPI | 0.115.x | 0.100.0 | HTTP API |
| uvicorn | 0.30.x | 0.20.0 | ASGI server |
| PyJWT | 2.9.x | 2.0.0 | JWT authentication |
| httpx | 0.27.x | 0.24.0 | Backend client |
| prometheus-client | 0.21.x | 0.17.0 | Metrics export |

### vLLM Integration

| Package | Validated Version | Notes |
|---------|-------------------|-------|
| vLLM | 0.20.1 / V1 engine | **Version-locked** |

> **Warning**: vLLM internal APIs change frequently between versions.
> The Neural-Scalpel scheduler patch and model runner hook are validated
> only against the version listed above. Using other versions may cause
> silent failures in route isolation.
>
> - Internal vLLM plugin mode remains version-locked and controlled-validation-only.
> - External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
> - External Proxy Fallback trades VRAM efficiency and route density for operational stability.

### CUDA / GPU

| Component | Validated | Notes |
|-----------|-----------|-------|
| CUDA Toolkit | 12.1 | Also tested with 11.8 |
| cuDNN | 8.9.x | Bundled with PyTorch |
| NVIDIA Driver | ≥525 | Ampere+ GPUs |

## Pinned Requirements

For reproducible environments, use the pinned requirements:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install safetensors==0.4.5 fastapi==0.115.0 uvicorn==0.30.0
pip install pyjwt==2.9.0 httpx==0.27.0 prometheus-client==0.21.0
pip install jsonschema==4.23.0
```

## Updating Versions

Before updating any dependency:
1. Run the default non-live suite: `PYTHONPATH=. python -m pytest -v --tb=short -m "not vllm_live"`
2. Run live vLLM smoke tests separately
3. Run the latest-branch 10K endurance test
4. Run the coarse E2E benchmark
5. Run Phase 5-D repeated median benchmark
6. Run Phase 5-E-1 two-route mixed-batch validation
7. Run Phase 5-E-2 3+ route mixed-batch validation
8. Run Phase 5-E-3 worst-case alternating route stress validation
9. Run Phase 5-F determinism follow-up
10. Run or schedule the 24h mixed-route soak test before Production Candidate declaration
11. Update this document with the new validated version
12. Tag the commit with the new version lock
