# vLLM Route Plugin (Internal Integration)

This directory contains the internal integration code for injecting Neural-Scalpel's `route_id` and Hot-Swap runtime directly into the core of vLLM.

## Purpose
The external proxy (`vllm_proxy.py`) guarantees 100% route isolation but severely limits continuous batching throughput. This plugin aims to implement **Route-Homogeneous Batching** directly within vLLM, allowing requests with the same `route_id` to be continuously batched together while maintaining strict isolation across different routes.

## Development Status
This plugin is currently under development (Phase 0). It is completely isolated from the main Neural-Scalpel repository and acts as a patch/wrapper around a strictly locked version of vLLM.

## Architecture
See [DESIGN.md](DESIGN.md) for the internal integration architecture.
See [VERSION_LOCK.md](VERSION_LOCK.md) for target versions.
