# Qwen2.5-0.5B SQL Adapter Projection Case Study Template
 
 > [!WARNING]
 > **Status: Phase 4 Initial Target-Activation-Conditioned Scaffold / Behavioral Transfer NOT PROVEN**  
 > This case study has implemented a **target-activation-conditioned projection scaffold** using JTSA-style self-alignment and target-side manifold statistics (mean/std/PCA). However, **behavioral improvement remains not proven**. Under greedy decoding, Base and Projected outputs up to $\gamma=4.0$ were identical across the smoke set.


## Overview

Can a tiny 0.5B model inherit useful SQL/Coding behavior without retraining? This template provides a structured scaffold for testing that question through mathematical adapter projection.

> [!NOTE]
> By default, the scaffold scripts can still run in simulation mode for reporting-flow tests. Real-validation outputs are explicitly labeled separately.

## Project Structure

- `scripts/`: Implementation scripts for each phase (00 to 08).
- `eval/`: Evaluation prompts and database schemas.
- `reports/`: Generated validation and evaluation reports.
- `hf_card/`: Assets for the Hugging Face model card.
- `docs/`: Technical methodology and deep dives.

## Results Summary

> [!WARNING]
> **Status: Phase 4 Target-Activation-Conditioned Scaffold / Behavioral Validation Pending**  
> This case study has completed structural projection and an initial target-conditioned self-alignment attempt. However, **behavioral improvement is NOT PROVEN**.  
> Real-inference smoke tests showed **100% identity** between base and projected outputs at low $\gamma$.

**Current Task Status:**
- [x] Structural projection baseline v2: **COMPLETED** (Interpolated Folding)
- [x] Target-shape verification: **COMPLETED** (**PASS**)
- [x] PEFT adapter smoke test: **COMPLETED** (**PASS**)
- [x] Phase 4 Initial Target-Activation-Conditioned Attempt: **SCAFFOLD COMPLETED**
- [x] Gamma Sweep Automation: **SCAFFOLD ADDED** (Validation Pending)

### Technical Findings (Phase 3 Preliminary)
1.  **Identity Check (100% Identity)**: Even with **Initial Target-Activation Conditioning**, the projected adapter produced outputs **bit-identical** to the base model across all smoke prompts at $\gamma=1.0$ and $\gamma=4.0$.
2.  **Behavioral Gravity**: The 0.5B instruct-tuned base model exhibits extreme "behavioral gravity." Target-only statistics are insufficient to override the base distribution without explicit cross-model mapping.
3.  **Signal vs. Compression**: The current result validates that **Signal Extraction** must precede **Rank Compression**. The 600MB adapter, while structurally correct, fails to emit a detectable signal at low $\gamma$.
4.  **Heuristic Limits**: Output length and SQL signal metrics are based on preliminary heuristics. `max_length_hit` is a length-based proxy and not a true token-limit detection.
5.  **Verdict**: The current target-conditioned scaffold is verified as structurally valid and PEFT-loadable under controlled smoke tests, but true behavioral transfer likely requires **Paired Source-Target Alignment**.

## Strategic Research Roadmap (Next Steps)

### **Phase 4-B: Gamma Sweep Validation (Current Focus)**
- **Breakthrough Point Search**: Execute `08_gamma_sweep.py` with $\gamma \in [0, 32]$ to find where `exact_same_rate_normalized` drops.
- **Status Monitoring**: Distinguish between `SIGNAL_CANDIDATE` (behavioral shift) and `COLLAPSE` (model failure).

### **Phase 5: Paired Activation Alignment (Future Work)**
- **Triple-Stream Collection**: Collect hidden states from Source Base, Source+LoRA, and Target Base using identical prompts.
- **Cross-Model Mapping**: Learn explicit transformation matrices $P$ using Procrustes or Ridge Regression.

## Current Project State (Summary)
- **Structural projection baseline**: IMPLEMENTED
- **Target-only activation conditioning**: IMPLEMENTED
- **Gamma sweep automation**: IMPLEMENTED (Validation Pending)
- **Behavioral transfer**: NOT PROVEN
- **Signal breakthrough point**: NOT YET OBSERVED
- **Paired source-target activation alignment**: FUTURE WORK

---

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.
