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

## Next Steps (Step 4B)

Once the external proxy validation is complete, the next phase is **Internal Integration**:
- Passing `route_id` into the vLLM `Scheduler`.
- Tagging KV Cache blocks in the `BlockAllocator` with the route ID.
- Hooking the PyTorch native `swap()` function directly into the vLLM `ModelRunner` forward pass.
