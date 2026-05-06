# Qwen2.5-0.5B SQL Adapter Projection Case Study Template
 
 > [!WARNING]
 > **Status: Phase 4 Negative Baseline Established / Phase 5-F-2 Runtime Validation Pending**  
 > This case study has transitioned from target-only conditioning to **Paired Source-Target Activation Alignment**. Behavioral transfer remains **NOT PROVEN**.

## Current Research Status

### Phase 4 Recap
Target-only activation conditioning produced 100% output identity. Simple signal amplification was insufficient.

### Phase 5-E / 5-F-1: Transported Delta and Activation-space Adapter Solve

Phase 5-E transports the source-side behavioral activation delta into the target representation space using learned source-to-target alignment maps. 

The output is a **desired target activation delta**, which serves as a teacher signal, not yet a deployable adapter.

Phase 5-F-1 solves an activation-space adapter to test whether the transported deltas are linearly recoverable from target hidden states.

### Phase 5-F-2: Runtime Activation Injection Smoke (Current Gate)

Phase 5-F-2 injects the solved activation-space adapter through runtime forward hooks to test whether the transported signal can produce measurable logit-level or text-level movement in the target model.

> [!IMPORTANT]
> Proceed to PEFT LoRA extraction only if runtime injection produces measurable logit movement without collapse.

Current status:
- Activation-space adapter solve: IMPLEMENTED
- Runtime injection smoke: IMPLEMENTED / READY FOR TEST
- PEFT LoRA extraction: PENDING
- Behavioral transfer: NOT YET PROVEN

## Strategic Research Roadmap

### **Phase 5: Paired Source-Target Alignment (Current Goal)**

1.  **Phase 5-A/B/C**: Capture paired activations, extract source deltas, and estimate initial layer correspondence. [COMPLETED]
2.  **Phase 5-D/E**: Learn alignment maps and transport behavioral deltas to target manifold. [COMPLETED]
3.  **Phase 5-F-1**: Solve activation-space adapter and verify linear recoverability. [COMPLETED]
4.  **Phase 5-F-2: Runtime Activation Injection Smoke**: (NEXT STEP) Inject solved activation-space deltas via forward hooks to observe logit/text divergence.
5.  **Phase 5-F-3**: Module-level activation collection (Prerequisite for PEFT).
6.  **Phase 5-F-4**: PEFT LoRA extraction (Weight Solve).

---

See [reports/](reports/) for generated validation logs.
