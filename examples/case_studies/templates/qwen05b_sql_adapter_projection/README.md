# Qwen2.5-0.5B SQL Adapter Projection Case Study Template
 
 > [!WARNING]
 > **Status: Phase 4 Negative Baseline Established / Phase 5 Paired Alignment Design Initiated**  
 > This case study has implemented a target-activation-conditioned projection scaffold. However, **behavioral transfer remains not proven**. Under the current setup, no measurable text-level or logit-level difference was detected up to $\gamma=32.0$.


## Overview

Can a tiny 0.5B model inherit useful SQL/Coding behavior without retraining? This template provides a structured scaffold for testing that question through mathematical adapter projection.

## Results Summary

### Gamma Sweep & Logit-level Delta Check

After the gamma sweep showed 100% output identity up to $\gamma=32.0$, a logit-level delta check was run to test whether any sub-threshold signal existed below the argmax decoding boundary.

**Result:** `NO_MEANINGFUL_LOGIT_DELTA`

- **Top-1 next-token agreement**: 100% (No preference shift)
- **Mean Symmetric KL Divergence**: Effectively Zero (No distribution shift)
- **SQL-related Token Logprobs**: No meaningful movement observed for keywords like `SELECT`.

**Interpretation:**  
Under the current 4-prompt greedy smoke setup, the target-only activation-conditioned projection did not produce a measurable output-level or distribution-level difference. This result establishes a **negative baseline**, suggesting that the current target-only self-alignment path is insufficient for this 7B → 0.5B case.

## Strategic Research Roadmap (Next Steps)

### **Phase 5: Paired Source-Target Activation Alignment (Current Goal)**

Paired Source-Target Alignment is the next most justified research step. The focus shifts from target-only statistics to learning explicit cross-model mappings:

1.  **Phase 5-A: Paired Activation Collection**: Collect hidden states from **Source Base**, **Source+LoRA**, and **Target Base** on identical prompts.
2.  **Phase 5-B: Source Behavioral Delta Extraction**: Calculate $\Delta H_{source} = H_{source+lora} - H_{source}$.
3.  **Phase 5-C: Alignment Map Learning**: Learn transformation matrices $P$ (e.g., via Procrustes) to translate the 7B behavioral signal into the 0.5B representation space.

## Current Project State (Summary)
- **Structural projection baseline**: IMPLEMENTED
- **Target-only activation conditioning**: NEGATIVE BASELINE (No signal detected)
- **Logit-level delta check**: IMPLEMENTED (Verdict: `NO_MEANINGFUL_LOGIT_DELTA`)
- **Behavioral transfer**: NOT PROVEN
- **Paired Source-Target Alignment**: IN DESIGN (NEXT STEP)

---

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.
