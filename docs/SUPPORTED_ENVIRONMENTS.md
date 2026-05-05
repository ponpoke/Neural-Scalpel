# Neural-Scalpel Supported Environments

## Production Candidate Preparation Scope

The following environments are validated or targeted for the initial production-candidate pilot. Final Production Candidate status remains pending until 24h mixed-route soak validation is complete.

### Hardware

| Component | Validated | Notes |
|-----------|-----------|-------|
| GPU | Single NVIDIA GPU (Ampere+) | A100, RTX 3090/4090, L40 |
| VRAM | ≥8 GB | Model-dependent; 0.5B models require ~2GB |
| CPU | x86_64 | ARM not yet tested |
| RAM | ≥16 GB | Includes payload staging buffer |

### Software

| Component | Validated Version | Notes |
|-----------|-------------------|-------|
| Python | 3.9 – 3.12 | 3.10+ recommended |
| PyTorch | 2.0+ | CUDA 11.8 or 12.1 |
| vLLM | V1 (version locked) | See VERSION_LOCK.md |
| CUDA | 11.8 / 12.1 | Driver ≥525 |
| OS | Linux (Ubuntu 22.04+) | Windows for development only |

### Models

| Model | Status | Architecture | Notes |
|-------|--------|-------------|-------|
| OPT-125M | Supported | OPT | Development reference |
| Qwen2.5-0.5B | Supported | Qwen2 | Primary validation target |
| Qwen2.5-Coder-0.5B | Supported | Qwen2 | Code generation variant |
| Qwen2.5-1.5B | Experimental | Qwen2 | Memory-constrained testing |
| Llama-3.2-1B | Experimental | Llama | Cross-architecture validation |
| Llama-3.1/3.2-8B | Unsupported | Llama | Requires >8GB VRAM headroom |

### Serving Modes

| Mode | Status |
|------|--------|
| Phase 5-D repeated median benchmark | Validated |
| Phase 5-E-1 two-route mixed-batch validation | Validated |
| Phase 5-F determinism follow-up | Validated under tested cache-reset condition |
| 3+ route mixed-batch validation | Pending |
| Worst-case alternating route stress | Pending |
| Route-homogeneous batching | Validated |
| Safetensors payload swap | Validated |
| Single-GPU serving | Validated |
| Streaming responses | Not supported |
| Multi-GPU tensor parallelism | Not supported |
| Multi-node serving | Not supported |

## Unsupported Configurations

The following are explicitly **out of scope** for Production Candidate:
- vLLM versions other than the locked version
- Models not in the compatibility matrix
- Multi-node deployments
- Custom CUDA kernels (fallback path only)
- SLA commitments without operator review
