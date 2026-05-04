# Internal Integration Design

This document details the design of the vLLM internal plugin for Neural-Scalpel.

## Goal
To implement **Route-Homogeneous Batching** within the vLLM core.

- **Rule 1:** A single forward pass can only contain requests associated with exactly one `route_id`.
- **Rule 2:** Different `route_id`s must not be mixed within the same execution batch.
- **Rule 3:** The KV Cache must be strictly isolated by `route_identity` (combining `route_id`, payload hash, and target model hash) to prevent cross-contamination.
- **Rule 4:** `HotSwapRuntime` swap and rollback must be performed immediately before and after the forward pass (or after a route window) at the `ModelRunner` level.

## Phases

### Stage 1: Hook Discovery
- Inject `route_id` from the OpenAI-compatible API to the internal `Request` object.
- Persist `route_id` through the `SchedulerOutput`.

### Stage 2: Route-Aware Scheduling & KV Isolation
- Modify the `Scheduler` to group waiting requests into `route_id` buckets.
- Modify the prefix cache and block allocator to tag KV cache blocks with `route_identity`.

### Stage 3: ModelRunner Hook
- Intercept the forward pass in `ModelRunner`.
- Enforce the fail-close batch homogeneity check.
- Hook into `HotSwapRuntime` for atomic weight swapping and rollback.

## Target Execution Flow (Mode A: Per-Forward Rollback)

```
[API Server] -> Parses 'X-Scalpel-Route-ID' -> Adds to Request
    |
[Scheduler] -> Selects requests belonging to a SINGLE route_id
    |       -> Allocates KV Cache blocks tagged with route_identity
    |
[ModelRunner] -> Asserts batch is route-homogeneous
    |         -> HotSwapRuntime.swap(route_id)
    |         -> output = model.forward(batch)
    |         -> HotSwapRuntime.rollback()
    |
[Output]
```
