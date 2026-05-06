# Qwen2.5-0.5B SQL Adapter Projection Case Study Template
 
 > [!WARNING]
 > **Status: Phase 5-F-2 Positive Runtime Signal Observed / Proceeding to PEFT Solve**  
 > Runtime injection has demonstrated that the transported behavioral signal can influence the target model's decision boundary. However, task-level behavioral transfer is **NOT PROVEN**.

## Current Research Status

### Phase 5-F-2 Runtime Injection Result

The activation-space adapter was injected into the target model through forward hooks and evaluated with a gamma sweep.

Unlike the earlier target-only projection path, paired-alignment runtime injection produced measurable next-token distribution movement. At higher gamma values, all tested prompts showed Top-1 next-token changes.

This is a **positive runtime signal**: the transported activation-space adapter can affect the target model's decision boundary.

> [!IMPORTANT]
> This result does **NOT** yet prove SQL skill transfer, stable generation improvement, or deployable PEFT LoRA success. The observed token shifts demonstrate influence, not necessarily improvement. The next step is to collect module-level activations and attempt a low-rank PEFT-style adapter solve.

## Strategic Research Roadmap

### **Phase 5: Paired Source-Target Alignment (Current Goal)**

1.  **Phase 5-A/B/C**: Capture paired activations, extract source deltas, and estimate initial layer correspondence. [COMPLETED]
2.  **Phase 5-D/E**: Learn alignment maps and transport behavioral deltas to target manifold. [COMPLETED]
3.  **Phase 5-F-1**: Solve activation-space adapter and verify linear recoverability. [COMPLETED]
4.  **Phase 5-F-2: Runtime Activation Injection Smoke**: [COMPLETED] Positive runtime signal observed (top-1 shifts detected).
5.  **Phase 5-F-3: Module-level Activation Collection**: (NEXT STEP) Collect module-level target inputs required for PEFT LoRA solving.
6.  **Phase 5-F-4: PEFT LoRA Export**: Solve weight deltas and export as deployable `.safetensors`.

---

See [reports/](reports/) for generated validation logs.
