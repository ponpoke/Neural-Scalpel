# Version Lock

To ensure reproducibility and safety when hooking into internal APIs, the vLLM Route Plugin is strictly locked to the following versions.

## Target Environment

| Component | Version | Notes |
| :--- | :--- | :--- |
| **vLLM** | `0.20.1` | Must match exact internal API surface |
| **Python** | `3.10` | |
| **PyTorch** | `>= 2.0.0` | Required for Hot-Swap synchronization |
| **CUDA** | `11.8` or `12.1` | Dependent on PyTorch wheel |

## Target Models

The integration is primarily tested against the following models:
- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`
