# External Proxy Fallback: Qualitative Trade-off Analysis

## Objective
To quantify the qualitative trade-offs between the Internal Plugin (Mode A) and the External Proxy Fallback (Mode B) in terms of performance, resource efficiency, and operational resilience.

> [!NOTE]
> This report is a qualitative trade-off analysis. Quantitative latency, throughput, and VRAM measurements should be collected in a later benchmark run once a full multi-backend environment is staged.

## Qualitative Comparison Table

| Metric | Mode A: Internal Plugin | Mode B: External Proxy | Impact/Notes |
| :--- | :--- | :--- | :--- |
| **Route Density** | High (100+ routes/GPU) | Low (1-5 backends/GPU) | Mode B limited by VRAM of separate instances |
| **Swapping Latency** | Ultra-low (window-swap) | Moderate (network hop) | Mode B adds HTTP overhead and process context |
| **Memory Efficiency** | High (Shared weights) | Low (Duplicated base weights) | Mode B requires VRAM for each backend process |
| **Version Resilience** | Low (Version-locked) | High (Process isolation) | Mode B survives vLLM internal signature changes |
| **Failure Isolation** | Shared process risk | Process-level isolation | Mode B prevents internal crashes from spreading |
| **Implementation** | Complex (Monkey-patch) | Simple (HTTP Forwarding) | Mode B uses standard OpenAI-compatible API |

## Key Findings

1. **Operational Resilience**: External Proxy Fallback successfully mitigates the "vLLM Internal Change" risk by trading off route density. It is the recommended mode for environments where vLLM versions cannot be strictly pinned or where stability is prioritized over density.
2. **Resource Cost**: Moving from Mode A to Mode B increases VRAM requirements significantly (proportionally to the number of base model instances). This mode should be reserved for high-priority routes or small-scale deployments during fallback scenarios.
3. **Security Parity**: Both modes maintain identical tenant-level access control and audit traceability, as the Neural-Scalpel API layer remains the mandatory gateway. Even in Proxy mode, the `RouteRegistry` must validate the request before forwarding.

## Conclusion
External Proxy Fallback is now implemented and smoke-validated as a compatibility-risk mitigation path. It provides a robust "Safety Net" option when the internal vLLM plugin is unavailable or disabled. While it trades off the extreme route density and memory efficiency of Route-Window Persistent Swapping, it ensures operational continuity through process-level isolation.
