はい。以下をそのまま
`docs/EXTERNAL_PROXY_FALLBACK_DEFINITION.md`
として保存できる定義書として使えます。

````md
# External Proxy Fallback Definition

## Status

**Planned / Compatibility-Risk Mitigation Track**

This document defines the External Proxy Fallback strategy for Neural-Scalpel.

The purpose of this track is to reduce operational risk caused by the current internal vLLM monkey-patch integration. The fallback mode provides a safer, version-resilient serving path when internal vLLM patching is unavailable, unsupported, or judged too risky for a deployment environment.

---

## 1. Background

Neural-Scalpel currently has two serving integration paths:

1. **Internal vLLM Plugin Path**
   - Uses vLLM V1 internal hooks / monkey-patching.
   - Enables in-process route-window weight swapping.
   - Provides strong controlled-validation performance evidence.
   - Carries compatibility risk when vLLM internals change.

2. **External Proxy Path**
   - Keeps vLLM instances outside the Neural-Scalpel runtime.
   - Routes requests at the API/process level.
   - Does not patch vLLM internals.
   - Trades performance and VRAM efficiency for operational stability.

The External Proxy Fallback is intended to become the primary safety fallback when internal patch compatibility checks fail.

---

## 2. Goal

The goal is to provide a production-safer fallback path that can continue serving requests without depending on vLLM internal monkey-patches.

The fallback should:

- avoid patching vLLM internals
- fail closed when internal plugin compatibility is uncertain
- route traffic through separate backend instances or route-isolated execution paths
- preserve tenant, route, audit, and policy enforcement at the Neural-Scalpel proxy layer
- provide a documented degradation path when internal hot-swap is unavailable

---

## 3. Non-Goals

External Proxy Fallback does **not** aim to preserve all benefits of the internal hot-swap path.

It does not guarantee:

- the same throughput as route-window persistent swapping
- the same VRAM efficiency as in-process swapping
- hundreds of routes on one GPU
- seamless high-density multi-tenant route serving
- elimination of all vLLM compatibility concerns
- SLA-grade production readiness by itself

This fallback is a safety and compatibility mechanism, not a replacement for full production hardening.

---

## 4. Deployment Modes

### Mode A: Internal Plugin Mode (Route-window swapping in a single vLLM process)

> [!IMPORTANT]
> **Mandatory Route Validation**: Even in External Proxy Fallback mode, Neural-Scalpel requires the route to be registered in the `RouteRegistry` before forwarding. The `BackendRegistry` only resolves `route_id` to `backend_url`; it does not bypass route policy, tenant authorization, revocation, or quarantine checks. This ensures uniform security and safety across all serving modes.

```text
Client
  ↓
Neural-Scalpel API
  ↓
Internal vLLM Plugin
  ↓
Single vLLM process with route-window swapping
````

Use when:

* vLLM version is pinned and supported
* startup compatibility check passes
* patch self-test passes
* route-window performance is required
* deployment accepts internal integration risk

Status:

```text
High performance / higher compatibility risk
```

---

### Mode B: External Proxy Fallback Mode

```text
Client
  ↓
Neural-Scalpel Proxy
  ↓
Route-aware dispatcher
  ↓
Separate vLLM backend(s)
```

Example:

```text
route: __base__  → vLLM backend A
route: alpaca    → vLLM backend B
route: sql       → vLLM backend C
```

Use when:

* internal vLLM compatibility check fails
* unsupported vLLM version is detected
* monkey-patch path is disabled by operator policy
* deployment prioritizes stability over memory efficiency
* route count is small enough for separate backend instances

Status:

```text
Lower integration risk / higher resource cost
```

---

### Mode C: Native LoRA Fallback Mode

```text
Client
  ↓
Neural-Scalpel Proxy
  ↓
vLLM Native LoRA backend
```

Use when:

* route can be represented as standard vLLM Native LoRA
* internal hot-swap is unavailable
* vLLM Native LoRA performance is acceptable
* Neural-Scalpel-specific payload surgery is not required

Status:

```text
Official-serving-compatible / reduced Neural-Scalpel functionality
```

---

### Mode D: Fail-Closed Mode

```text
Unsupported vLLM version
  ↓
Compatibility check fails
  ↓
Service refuses to start or disables internal plugin
```

Use when:

* no safe fallback backend is configured
* route isolation cannot be guaranteed
* payload compatibility is unknown
* startup self-test fails

Status:

```text
Safest failure mode
```

---

## 5. Trigger Conditions

External Proxy Fallback should activate when any of the following conditions are true:

```text
internal_vllm_compatibility_check_failed == true
unsupported_vllm_version_detected == true
patch_target_signature_mismatch == true
patch_self_test_failed == true
operator_forced_proxy_mode == true
internal_plugin_disabled_by_config == true
```

Recommended environment variable:

```bash
SCALPEL_SERVING_MODE=internal|external_proxy|native_lora|fail_closed
```

Recommended default:

```bash
SCALPEL_SERVING_MODE=fail_closed
```

Recommended controlled validation setting:

```bash
SCALPEL_SERVING_MODE=internal
```

Recommended safer deployment setting:

```bash
SCALPEL_SERVING_MODE=external_proxy
```

---

## 6. Compatibility Check Flow

Before starting the internal vLLM plugin, Neural-Scalpel should run:

```text
1. Check configured serving mode
2. If internal mode:
   - verify vLLM version
   - verify expected classes/methods exist
   - verify method signatures
   - apply patch
   - run patch self-test
3. If any check fails:
   - do not patch
   - emit audit event
   - select fallback mode if configured
   - otherwise fail closed
```

Suggested pseudo-flow:

```python
mode = os.environ.get("SCALPEL_SERVING_MODE", "fail_closed")

if mode == "internal":
    try:
        assert_vllm_compatible()
        apply_internal_patches()
        run_patch_self_test()
        start_internal_plugin()
    except Exception as exc:
        audit("internal_plugin_unavailable", reason=str(exc))
        if fallback_configured():
            start_external_proxy_fallback()
        else:
            fail_closed()

elif mode == "external_proxy":
    start_external_proxy_fallback()

elif mode == "native_lora":
    start_native_lora_fallback()

else:
    fail_closed()
```

---

## 7. External Proxy Architecture

### 7.1 Components

```text
Neural-Scalpel Proxy
├── Auth / tenant gate
├── Route registry
├── Route policy verifier
├── Route-to-backend resolver
├── Request dispatcher
├── Audit logger
├── Metrics collector
└── Health checker
```

### 7.2 Backend Registry

The proxy maintains a mapping from route IDs to backend endpoints.

Example:

```json
{
  "__base__": {
    "backend_url": "http://localhost:8001/v1/completions",
    "backend_type": "vllm_base",
    "model": "Qwen/Qwen2.5-0.5B",
    "health": "healthy"
  },
  "qwen2.5-0.5b-alpaca-lora-demo": {
    "backend_url": "http://localhost:8002/v1/completions",
    "backend_type": "vllm_native_lora_or_preloaded",
    "model": "Qwen/Qwen2.5-0.5B",
    "health": "healthy"
  },
  "qwen2.5-coder-0.5b-instruct_text_to_sql_lora_newdataset": {
    "backend_url": "http://localhost:8003/v1/completions",
    "backend_type": "vllm_native_lora_or_preloaded",
    "model": "Qwen/Qwen2.5-0.5B",
    "health": "healthy"
  }
}
```

---

## 8. Request Flow

```text
Client request
  ↓
Neural-Scalpel Proxy
  ↓
Authenticate tenant
  ↓
Validate route manifest / policy / license / revocation
  ↓
Resolve route_id to backend_url
  ↓
Forward request to selected vLLM backend
  ↓
Collect response
  ↓
Write audit event
  ↓
Return response
```

---

## 9. Route Isolation Model

In external proxy fallback mode, route isolation is achieved through **process/backend separation** rather than in-process weight swapping.

### Internal Plugin Isolation

```text
same vLLM process
different active route windows
runtime swap / rollback required
```

### External Proxy Isolation

```text
separate vLLM backend per route or route group
no in-process weight mutation required
```

This avoids:

* monkey-patch compatibility risk
* runtime weight mutation in vLLM internals
* route-window rollback dependency
* internal scheduler fragility

But introduces:

* higher VRAM usage
* more backend processes
* more operational complexity
* lower route density per GPU

---

## 10. Backend Allocation Strategies

### Strategy A: One Backend Per Route

```text
__base__ → backend A
alpaca   → backend B
sql      → backend C
```

Pros:

* simplest isolation model
* easiest to reason about
* no route mixing risk inside backend

Cons:

* high VRAM usage
* poor scaling to many routes

Recommended for:

```text
small number of high-value routes
controlled pilot deployments
fallback mode after internal plugin failure
```

---

### Strategy B: One Backend Per Route Group

```text
general routes → backend A
coding routes  → backend B
sql routes     → backend C
```

Pros:

* lower resource cost than one backend per route
* can group compatible LoRA / Native LoRA routes

Cons:

* more complex route grouping logic
* still requires backend-level isolation policy

Recommended for:

```text
medium route count
traffic-shaping experiments
```

---

### Strategy C: Native LoRA Backend Pool

```text
route group → vLLM Native LoRA backend
```

Pros:

* uses existing vLLM LoRA functionality
* avoids internal monkey-patch
* more route-dense than one backend per route

Cons:

* loses Neural-Scalpel route-window swapping benefits
* performance may be lower than internal route-window mode
* only works for routes expressible as Native LoRA

Recommended for:

```text
standard LoRA fallback
compatibility-first deployments
```

---

## 11. Required Metrics

External Proxy Fallback must expose:

```text
proxy_requests_total
proxy_request_latency_ms
proxy_backend_latency_ms
proxy_backend_errors_total
proxy_backend_health
proxy_route_resolution_failures_total
proxy_auth_failures_total
proxy_policy_rejections_total
proxy_fallback_mode_active
proxy_backend_selection_total{route_id, backend_url}
```

Recommended audit events:

```text
fallback_mode_selected
internal_plugin_disabled
internal_plugin_compatibility_failed
backend_resolved
backend_unhealthy
backend_request_failed
backend_response_returned
route_policy_rejected
```

---

## 12. Pass Criteria

External Proxy Fallback is considered implemented when all of the following pass in controlled validation:

```text
unsupported internal vLLM version fails closed or falls back safely
operator can force external_proxy mode
route_id resolves to expected backend
invalid route_id is rejected
revoked route is rejected
tenant mismatch is rejected
backend health check works
unhealthy backend is not selected
audit log records fallback mode and backend selection
metrics expose fallback status
1000 routed requests complete with 0 route-resolution errors
0 cross-route backend misroutes observed
- [x] Small live proxy forwarding smoke test passed using a local FastAPI backend.
```

---

## 13. Validation Plan

### Phase Fallback-A: Configuration & Mode Selection (PASS)

Objective:

```text
Verify serving mode selection and fail-closed behavior.
```

Tests:

```text
SCALPEL_SERVING_MODE=internal
SCALPEL_SERVING_MODE=external_proxy
SCALPEL_SERVING_MODE=native_lora
SCALPEL_SERVING_MODE=fail_closed
invalid mode rejected
```

Pass:

```text
All modes select expected startup path.
Invalid mode fails closed.
```

---

### Phase Fallback-B: Backend Registry (PASS)

Objective:

```text
Verify route-to-backend mapping.
```

Tests:

```text
registered route resolves to expected backend
unknown route rejected
revoked route rejected
tenant mismatch rejected
unhealthy backend skipped/rejected
```

Pass:

```text
0 incorrect backend selections
0 unauthorized routes accepted
```

---

### Phase Fallback-C: Request Forwarding (PASS)

Objective:

```text
Verify proxy can forward requests to route-specific vLLM backends.
```

Tests:

```text
1000 requests across __base__, alpaca, sql
all requests complete
route distribution recorded
backend selection recorded
0 backend misroutes
0 policy bypasses
```

Pass:

```text
all_requests_completed == true
backend_misroutes == 0
policy_bypass == 0
```

---

### Phase Fallback-D: Internal Plugin Failure Fallback (PASS)

Objective:

```text
Verify unsupported internal plugin path does not serve traffic unsafely.
```

Tests:

```text
simulate unsupported vLLM version
simulate missing patch target
simulate patch self-test failure
simulate route hook not firing
```

Pass:

```text
internal plugin disabled
fallback selected if configured
otherwise fail closed
audit event recorded
```

---

### Phase Fallback-E: Qualitative Trade-off Analysis (PASS)

Objective:

```text
Quantify trade-offs between internal plugin and external proxy fallback.
```

Metrics:

```text
throughput
latency p50/p90/p99
VRAM usage
number of backend processes
route capacity
failure isolation behavior
```

Output:

```text
reports/external_proxy_fallback_comparison.json
```

---

## 14. Documentation Requirements

The following documents must be updated:

```text
README.md
docs/KNOWN_LIMITATIONS.md
docs/PRODUCTION_READINESS_CRITERIA.md
docs/DEPLOYMENT.md
docs/VERSION_LOCK.md
docs/RUNBOOK.md
docs/OBSERVABILITY.md
```

Required wording:

```text
Internal vLLM plugin mode remains version-locked and controlled-validation-only.
External Proxy Fallback provides a safer compatibility fallback when internal patching is unsupported.
External Proxy Fallback trades VRAM efficiency and route density for operational stability.
```

---

## 15. Success Statement

After implementation and validation, use:

```text
External Proxy Fallback has been implemented as a compatibility-risk mitigation path. When internal vLLM patch compatibility checks fail or operator policy disables monkey-patching, Neural-Scalpel can route requests through isolated external vLLM backends or fail closed. This reduces dependency risk on vLLM internals, but does not provide the same route density or memory efficiency as the internal route-window hot-swap path.
```

Avoid:

```text
vLLM dependency risk is eliminated.
```

Use instead:

```text
vLLM internal monkey-patch dependency risk is mitigated by version locking, fail-closed compatibility checks, and an external proxy fallback path.
```

---

## 16. Production Candidate Impact

External Proxy Fallback should be considered a **risk mitigation requirement**, not a replacement for the 24h soak.

Recommended status mapping:

| Internal route-window hot-swap     | Controlled validation PASS                    |
| External proxy fallback            | Functional Validation PASS                    |
| Native LoRA fallback               | Optional / Placeholder                        |
| Unsupported internal vLLM versions | Refuse to start / Fail Closed PASS            |
| 24h soak                           | Required for constrained Production Candidate |

A constrained Production Candidate declaration should require either:

```text
A. internal plugin compatibility checks + 24h soak pass
```

or:

```text
B. external proxy fallback validation + 24h route-isolated proxy soak pass
```

depending on the target deployment mode.

````

この定義書では、**外部proxy fallbackは「性能最適化」ではなく「互換性リスク対策」** として位置づけています。

なので、主張としては：

```text
vLLM依存を完全に消した
````

ではなく、

```text
vLLM内部monkey-patch依存のリスクを、version lock / fail-close / external proxy fallbackで管理可能にする
```

が正しいです。
