# vLLM Integration Validation (Step 4A)

This document describes how to run the external vLLM integration tests for Neural-Scalpel.

## Architecture

```text
[Neural-Scalpel vLLM Proxy (FastAPI)]
  ↑ HTTP (/v1/infer)
  | Enforces route-aware batching
  | Splits mixed-route batches
  | Triggers PyTorch native Swap & Cache Flush (Step 4B)
  ↓
[vLLM OpenAI-Compatible Server]
  ↑ HTTP (/v1/completions)
  |
(Qwen2.5-0.5B Model)
```

## Running the Backend (Docker)

To run the live integration test, you must first start the official vLLM Docker container as the backend engine.

### Prerequisites
- Docker installed
- NVIDIA Container Toolkit installed
- At least 8GB VRAM (for Qwen2.5-0.5B fp16)

### Start Command (Windows PowerShell / WSL2)

Run this command to start the backend vLLM server:

```bash
docker run --runtime nvidia --gpus all `
  -v ~/.cache/huggingface:/root/.cache/huggingface `
  -p 8000:8000 `
  --ipc=host `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-0.5B `
  --dtype float16 `
  --max-model-len 2048
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000` in the Docker logs.

## Running the Proxy & Tests

Neural-Scalpel provides a test suite (`tests/test_block_e_vllm.py`) that tests the proxy behavior.

### MOCK Mode (Fast CI)

By default, the tests run in MOCK mode to verify the proxy's internal temporal isolation logic without requiring the Docker backend.

```bash
python -m pytest tests/test_block_e_vllm.py -v
```

### LIVE Mode (Full Validation)

To test against the real vLLM backend running in Docker:

```bash
$env:VLLM_BACKEND_URL="http://localhost:8000/v1/completions"
python -m pytest tests/test_block_e_vllm.py -v
```

## What is Verified Here?

1. **Temporal Isolation:** The proxy guarantees that the backend vLLM engine only receives batches belonging to a single `route_id`. Mixed requests are queued and processed sequentially per route.
2. **Auditability:** Every request forwarded to vLLM is logged with its `route_id` and `tenant_id`.
3. **No Route Leakage:** Concurrent stress tests (e.g., 150 requests across 3 routes) complete with 0 leakage.

## Step 4B Status

Internal vLLM integration has since progressed beyond the original roadmap. A controlled vLLM V1 monkey-patch prototype has been validated through Phase 7H-2, including:

- route metadata injection
- active route-homogeneous scheduling
- real safetensors payload swap/rollback inside `_model_forward`
- 10K mixed-route endurance with 896 atomic swap/rollback cycles
- zero route violations in the tested environment

This remains a validated prototype, not production-ready serving software. Internal vLLM integration has since progressed beyond the original Step 4A/4B roadmap. Controlled validation now includes Phase 5-C route-window persistent swapping, Phase 5-D repeated median benchmarking across 50 prompts × 3 runs, Phase 5-E-1 two-route mixed-batch validation, and Phase 5-F determinism follow-up under explicit cache reset.

This remains controlled validation, not production-ready serving software. Formal Production Candidate status remains pending the final 24h persistent-route soak. Broader 3+ route validation, worst-case alternation stress, additional model families, and future vLLM version compatibility remain future hardening work.
