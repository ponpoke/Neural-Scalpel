# vLLM Route Plugin (Internal Integration)

This directory contains the internal integration code for injecting Neural-Scalpel's `route_id` and Hot-Swap runtime directly into the core of vLLM.

## Purpose
The external proxy (`vllm_proxy.py`) guarantees 100% route isolation but severely limits continuous batching throughput. This plugin aims to implement **Route-Homogeneous Batching** directly within vLLM, allowing requests with the same `route_id` to be continuously batched together while maintaining strict isolation across different routes.

## Development Status

**Status: Monkey Patch Implementation Complete; Live Linux/vLLM Validation Pending.**

### Implemented:
- `route_id` metadata injection
- Route-homogeneous Scheduler patch
- Route-aware KV cache hash policy
- GPUModelRunner swap/rollback hook
- Route registry adapter skeleton

### Pending (Phase 7+):
- Live vLLM import/runtime tests
- End-to-end generation under patched vLLM
- KV cache collision validation in real engine
- Throughput / TTFT degradation measurement
- 1K / 10K mixed-route endurance

This plugin is currently under development (Phase 0-6 complete). It is completely isolated from the main Neural-Scalpel repository and acts as a monkey patch around a strictly locked version of vLLM.

## Tested vLLM Version
The patch points are highly sensitive to vLLM version changes. See [VERSION_LOCK.md](VERSION_LOCK.md) for the exact locked versions.

## Architecture
See [DESIGN.md](DESIGN.md) for the internal integration architecture.
See [VERSION_LOCK.md](VERSION_LOCK.md) for target versions.
