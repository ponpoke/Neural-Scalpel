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

## Validation Update: Phase 5-C to 5-F

Validated:
- Route-window persistent swapping implemented and benchmarked.
- Phase 5-D repeated median benchmark completed across 50 prompts × 3 runs.
- Phase 5-E-1 two-route mixed-batch validation passed across 1000 dynamically routed requests.
- Phase 5-F determinism follow-up passed under explicit route cleanup and vLLM cache reset.

Remaining:
- 24h persistent-route soak validation.
- 3+ route mixed-batch validation.
- Worst-case alternating route transition stress.
- Broader vLLM version compatibility.


### Strategy B: Route-Window Persistent Swapping (Implemented in Phase 5-C)
- **Concept:** The `ModelRunner` holds the swapped weights active until the route identity of the incoming batch changes or a cleanup/window timeout is reached.
- **Pros:** Dramatically reduces the overhead of swap/rollback operations (one-time cost per window).
- **Cons:** Requires explicit management of `active_route_id` and careful cleanup logic.

**Decision:** We have transitioned to **Strategy B (Route-Window Persistent Swapping)** as the primary implementation in Phase 5-C, as it resolves the performance-prohibitive overhead of per-token swapping while maintaining architectural agnosticism.


---

## 4. Safety & Fail-Close Guarantees

1. **Mixed Batch Rejection:** If the `ModelRunner` receives a batch containing multiple different `route_id`s, it must raise an `UnsafeMixedRouteBatchError` and refuse to execute.
2. **KV Cache Isolation:** A cache hit is only valid if `hash(prompt) + route_id` matches. A block allocated under Route A cannot be appended to by Route B.
3. **Execution Quarantine:** If an exception occurs *during* the `forward()` pass while weights are swapped, the `ModelRunner` must attempt a rollback. If rollback fails checksum validation, the model enters a `QUARANTINE` state and requires a full weight reload.

---

## 5. Performance Tradeoffs

- **TTFT (Time To First Token):** May increase by the route swap/validation path. Recent controlled measurements include internal swap p50 around 0.59ms and rollback p50 around 2.19ms, while safetensors-heavy real-payload scenarios have shown higher p99 costs. Precise TTFT/TPOT remains pending.
- **Throughput:** Will be lower than vanilla vLLM due to Route-Homogeneous Batching, but significantly higher than the Step 4A External Proxy, because vLLM can still manage memory efficiently and batch requests of the *same* route natively.

## Phase 5-E/F Validation Updates

- Phase 5-E-1 validated two-route mixed-batch execution across 1000 dynamically routed requests, with 0 route violations and 0 quarantine events.
- Phase 5-F validated deterministic return-to-base behavior under explicit route cleanup and vLLM cache reset, using exact text match and top-token logprob trace similarity as a proxy check.
- 3+ route mixed-batch and worst-case alternating route transitions remain future hardening work.
