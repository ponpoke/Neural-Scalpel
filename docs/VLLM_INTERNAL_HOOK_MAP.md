# vLLM Internal Hook Map

This document outlines the strategic hook points within the vLLM (v0.20.1) core architecture required to implement Neural-Scalpel's Route-Homogeneous Batching and Hot-Swap Runtime.

## 1. Route ID Injection Point
**Goal:** Pass `route_id` from the OpenAI-compatible API to the internal `Request` object.
**Hook Points:**
- **`vllm/entrypoints/openai/serving_chat.py` / `serving_completion.py`:**
  - Extract `X-Scalpel-Route-ID` from HTTP headers or from a custom `metadata` field in the JSON body.
  - Pass this as an `extra_args` or a dedicated field to `SamplingParams` or directly to the engine's `add_request` method.
- **`vllm/v1/request.py` (`Request` class):**
  - Add `self.route_id: str | None = None` to the `Request` object.
  - Populate it during `Request.__init__` from the incoming API parameters.
  - Route ID defaults to `__base__` or `None` if missing.

## 2. Scheduler Modification Point
**Goal:** Enforce Route-Homogeneous Batching so that no mixed routes exist in a single forward pass.
**Hook Points:**
- **`vllm/v1/core/sched/scheduler.py` (`Scheduler.schedule`):**
  - **Bucketization:** Group the `waiting` queue into sub-queues partitioned by `route_id`.
  - **Selection:** Implement a round-robin or priority-based selection of an "active route" for the current scheduling cycle.
  - **Filtering:** Only pull requests from the active route's bucket into the `running` batch.
  - **Output:** Ensure `SchedulerOutput` contains requests strictly belonging to a single `route_id`.

## 3. KV Cache Tagging Point
**Goal:** Prevent KV Cache cross-contamination between different routes.
**Hook Points:**
- **`vllm/v1/core/kv_cache_manager.py` / `block_pool.py`:**
  - Add `route_identity` (e.g., `route_id + payload_hash + model_hash`) to the KV Cache block metadata.
- **`vllm/v1/core/sched/scheduler.py` (Prefix Cache Allocation):**
  - Modify the prefix caching hash function (`hash(tokens, route_identity)`) so identical prompts with different `route_id`s map to different cache blocks.
- **Verification:** Reject any attempt to append to or read from a block whose `route_identity` does not match the requesting request's `route_identity`.

## 4. Model Runner Swap/Rollback Point

**Final validated hook point:**
- `vllm/v1/worker/gpu_model_runner.py`
- `GPUModelRunner._model_forward`

Earlier prototypes targeted `GPUModelRunner.execute_model`, but live vLLM validation showed that `_model_forward` is the correct forward-boundary hook for Neural-Scalpel's per-forward swap/rollback lifecycle.

Validated behavior after Phase 5-C:
- read active route from runtime context
- call `runtime.ensure_route(route_id)`
- if the requested route is already active, skip redundant swap
- if the route changes, rollback the previous route and apply the new route
- execute original `_model_forward`
- keep the route active across the route window
- perform explicit cleanup rollback via `clear_active_route()`
- verify rollback through audit counters and checksum verification

## 5. Failure Handling Path
**Goal:** Guarantee fail-close behavior and quarantine upon critical errors.
**Hook Points:**
- **`vllm/v1/worker/gpu_model_runner.py`:**
  - Wrap the `execute_model` logic in a `try...except...finally` block.
  - On exception during `forward`, trigger an emergency `rollback()`.
  - If the checksum verification during `rollback()` fails, transition the worker node into a `QUARANTINE` state, preventing any further requests until a full model reload occurs.
- **`vllm/v1/core/sched/scheduler.py`:**
  - If `execute_model` fails, the scheduler must abort the current batch and emit failure states to the affected requests via `RequestStatus.FINISHED_ERROR`.
