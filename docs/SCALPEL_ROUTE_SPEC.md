# Scalpel Route Specification

**Version:** 0.1.0
**Extension:** `.scalpel_route.json`

## Overview
A `.scalpel_route` file is a cryptographically verifiable manifest that defines a successful weight-delta projection (LoRA transplantation). It contains the necessary metadata, layer definitions, checksums, and signatures required for the Neural-Scalpel Hot-Swap Runtime to safely inject the adapter into a live model.

## Structure

### 1. Root Metadata
- `route_schema_version`: (String) Semantic version of the schema (e.g., "0.1.0").
- `route_id`: (String) Unique identifier for the route.
- `source_model`: (String) Original model ID (e.g., "meta-llama/Meta-Llama-3-8B").
- `target_model`: (String) Destination model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
- `source_adapter_sha256`: (String) SHA-256 hash of the original LoRA safetensors.
- `target_model_sha256`: (String) SHA-256 hash of the target base model safetensors.
- `projection_method`: (String) Mathematical projection method used (e.g., "JTSA_WDR_CALIBRATED").

### 2. Calibration Metadata
- `calibration`: (Object)
  - `forward_passes`: (Integer) Number of passes used for manifold estimation.
  - `dataset_hash`: (String, Optional) Identifier or hash of the calibration dataset.

### 3. Diagnostics & Safety Gates
- `diagnostics`: (Object)
  - `verdict`: (String) "PASS", "CAUTION", or "FAIL". The runtime rejects "FAIL".
  - `ppl_degradation`: (Float) Perplexity degradation measured during conversion.
  - `kl_divergence`: (Float) KL divergence from baseline.
  - `portability_score`: (Integer) Heuristic score out of 100.

### 4. Layer Definitions (The Payload)
- `layers`: (Array of Objects) Defines exactly which layers are targeted.
  - `name`: (String) Target layer name (e.g., "model.layers.0.self_attn.q_proj.weight").
  - `shape`: (Array of Integers) Expected tensor shape.
  - `dtype`: (String) PyTorch datatype string (e.g., "float16", "bfloat16").
  - `delta_sha256`: (String) Hash of the task vector for this specific layer.

### 5. Cryptographic Signature
- `signature`: (Object)
  - `algorithm`: (String) E.g., "ed25519", "hmac-sha256".
  - `key_id`: (String) Identifier of the signing key.
  - `value`: (String) The actual cryptographic signature.

## Validation Protocol
Before injection, the Hot-Swap Runtime MUST verify:
1. JSON Schema compliance.
2. `target_model_sha256` matches the loaded base model.
3. `verdict` is not "FAIL".
4. Layer shapes and dtypes match the live VRAM tensors.
5. `signature.value` is authentic using the trusted `key_id`.