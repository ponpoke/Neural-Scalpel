# Neural-Scalpel: vLLM Internal Integration Design (Step 4B)

## 1. Introduction: Why Internal Integration?

The **External Proxy (Step 4A)** successfully proved that we can achieve 100% strict temporal isolation and prevent route leakage by acting as a gateway in front of an unmodified vLLM server. However, external proxies have limitations:

1. **Throughput Bottleneck:** By forcing temporal isolation at the HTTP layer, the proxy limits continuous batching. It must drain the vLLM request queue entirely before swapping to a new route.
2. **Suboptimal Hardware Utilization:** If Route A has 1 request and Route B has 1 request, an external proxy must process them sequentially. A native integration could theoretically process them concurrently or at least pipeline them tightly.
3. **KV Cache Waste:** The proxy has no control over vLLM's `BlockAllocator`. When a route is swapped, previously computed KV caches for another route might be invalidated or, worse, cross-contaminated if we aren't careful with session IDs.

**Step 4B (Internal Integration)** aims to push route awareness directly into vLLM's core components (`Scheduler`, `BlockAllocator`, and `ModelRunner`).

---

## 2. Architecture Overview

To natively support Hot-Swap, vLLM must become "Route-Aware". This requires injecting the `route_id` into the lifecycle of every request.

### Core Components Modified

1. **`RouteAwareScheduler`**:
   - Understands `route_id` per request.
   - Enforces batching policies (e.g., only same-route requests in a single batch, or time-sliced route windows).
2. **`RouteTaggedKVBlock` / `RouteAwareBlockAllocator`**:
   - KV cache blocks must be tagged with `route_id`.
   - Reusing a KV cache block for a request with a *different* `route_id` must be strictly rejected to prevent memory contamination across tenants.
3. **`RouteAwareModelRunner`**:
   - Intercepts the execution right before `forward()`.
   - Checks the `route_id` of the scheduled batch.
   - Triggers `HotSwapRuntime.swap()` if the active route differs from the batch's route.
   - Triggers `HotSwapRuntime.rollback()` after the forward pass (or after a time-sliced window).
   - Enforces a fail-close quarantine if the forward pass crashes.

---

## 3. Continuous Batching Strategies

vLLM's continuous batching is designed to mix multiple requests together to maximize GPU utilization. Neural-Scalpel introduces a constraint: **A single forward pass can only execute under one active set of weights.**

## 8. Validation Update: Phase 7A-7H

The initial Strategy A design, Route-Homogeneous Batching, has been validated as a controlled vLLM V1 monkey-patch prototype.

Validated:
- route metadata injection
- active route-homogeneous scheduling via shelving
- mixed-route fail-close
- decode-step route lifecycle retention
- real safetensors payload swap/rollback inside `_model_forward`
- 10K mixed-route endurance with 896 atomic swap/rollback cycles
- zero route violations in the tested environment

Remaining:
- vanilla vLLM TTFT / throughput regression
- corrupted payload / I/O failure / rollback failure hardening
- broader model coverage
- version compatibility with future vLLM releases

### Strategy A: Route-Homogeneous Batching (Initial Target)
- **Concept:** The Scheduler is modified to group requests such that a single execution batch contains only requests sharing the same `route_id` (or Base/None).
- **Pros:** Safety is mathematically guaranteed. The `ModelRunner` just swaps to Route X, runs the batch, and rolls back.
- **Cons:** Lower throughput if requests are highly diverse, as batches will be smaller.

### Strategy B: Time-Sliced Route Windows (Intermediate Target)
- **Concept:** Similar to Strategy A, but the `ModelRunner` holds the swapped weights for a brief window (e.g., 50ms) to allow multiple generation steps for Route X to complete before swapping to Route Y.
- **Pros:** Reduces the overhead of swap/rollback operations.
- **Cons:** Requires slight modifications to how vLLM handles step iteration.

### Strategy C: Mixed Route Batching via Per-Token Adapters (Future / Out of Scope)
- **Concept:** The model allows multiple routes in the *same* forward pass, dynamically routing tokens to different LoRA weight matrices based on block metadata.
- **Pros:** Maximum theoretical throughput (true vLLM style).
- **Cons:** Requires custom CUDA kernels (e.g., Punica or S-LoRA) and breaks the architecture-agnostic promise of Neural-Scalpel. *This is explicitly NOT the goal of Neural-Scalpel.*

**Decision:** We will implement **Strategy A (Route-Homogeneous Batching)** for the mock and initial vLLM patch, as it aligns perfectly with Neural-Scalpel's fail-close security and architecture-agnostic goals.

---

## 4. Safety & Fail-Close Guarantees

1. **Mixed Batch Rejection:** If the `ModelRunner` receives a batch containing multiple different `route_id`s, it must raise an `UnsafeMixedRouteBatchError` and refuse to execute.
2. **KV Cache Isolation:** A cache hit is only valid if `hash(prompt) + route_id` matches. A block allocated under Route A cannot be appended to by Route B.
3. **Execution Quarantine:** If an exception occurs *during* the `forward()` pass while weights are swapped, the `ModelRunner` must attempt a rollback. If rollback fails checksum validation, the model enters a `QUARANTINE` state and requires a full weight reload.

---

## 5. Performance Tradeoffs

- **TTFT (Time To First Token):** Will increase by the duration of the `swap()` operation (~8-15ms for safetensors) for the first request of a new route.
- **Throughput:** Will be lower than vanilla vLLM due to Route-Homogeneous Batching, but significantly higher than the Step 4A External Proxy, because vLLM can still manage memory efficiently and batch requests of the *same* route natively.
