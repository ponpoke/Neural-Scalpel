# Research & Commercialization Roadmap

**Status:** Strategic Planning Document  
**Vision:** Transitioning Neural-Scalpel from an "Experimental Toolkit" to a robust "LoRA Portability Diagnostic & Evaluation Suite."

---

## 1. The Core Challenge

Initially, Neural-Scalpel demonstrated that cross-architecture weight projection (e.g., LLaMA to Qwen) is mathematically possible. However, a successful mathematical projection (low Procrustes error, low PPL degradation) does not automatically guarantee that the target model can execute the complex reasoning logic originally learned by the source model. 

To bridge the gap between **ML Research** and **Commercial Viability**, Neural-Scalpel has adopted a strict validation and diagnostic roadmap. We acknowledge that the true value of this technology lies not just in "attempting a conversion," but in **diagnosing the feasibility, safety, and exact quality retention** of that conversion before deployment.

---

## 2. ML Research Requirements

To establish Neural-Scalpel as a rigorous ML research project, we are implementing the following empirical validation frameworks:

### A. The 6-Way Empirical Comparison
Any downstream benchmark (e.g., HumanEval, GSM8K) must isolate the true impact of the Neural-Scalpel projection using a strict 6-way comparison:
1. **Source Base:** Establishes the baseline.
2. **Source Base + Source LoRA:** Proves the original LoRA actually possessed the capability.
3. **Target Base:** Establishes the target baseline.
4. **Target + Naive Projection:** (Zero-padded baseline) Proves that our algorithm is better than doing the bare minimum.
5. **Target + Random Projection:** (Control) Proves that structural rotation isn't just acting as beneficial noise.
6. **Target + Projected LoRA (Neural-Scalpel):** The final evaluation.

### B. Executable Ablation Framework
We must empirically prove the necessity of each mathematical component. The `diagnose` CLI tool now includes an `--ablation all` mode to evaluate the adapter across the following configurations:
- Naive Padding / Resize
- Random Orthogonal Projection
- Procrustes Only (Linear)
- Procrustes + AVPS (Sparsity check)
- Procrustes + WDR (Routing check)
- JTSA + WDR without calibration (Zero-dataset collapse proof)
- JTSA + WDR with empirical calibration (Full Pipeline)

### C. Multi-Adapter / Multi-Model Diversity
To prevent overfitting our claims to a single domain, future benchmarks will evaluate:

**LoRA Types:**
1. Creative-writing LoRA
2. Coding LoRA
3. Instruction/style LoRA
4. Reasoning LoRA
5. Vision-style LoRA

**Model Pairs:**
1. LLaMA-3 $\to$ Qwen-2.5
2. Qwen-2 $\to$ Qwen-2.5
3. Mistral $\to$ Qwen-2.5
4. SDXL $\to$ SDXL-compatible
5. SDXL $\to$ FLUX

---

## 3. Commercialization Strategy

Enterprises do not pay for experimental conversions with unpredictable failure rates. They pay for **risk mitigation, workflow automation, and asset preservation**.

Our commercial entry point is the **Diagnostic Suite**.

### Phase 1: OSS + Experimental Tooling (Current)
- Open-source the core projection algorithms.
- Establish the `diagnose` CLI to generate "LoRA Portability Feasibility Reports."
- **Phase 5-B: Per-Token Overhead Evaluation (Complete)** - Confirmed per-token swap is performance-prohibitive.
- **Phase 5-C: Route-Window Optimization (Complete)** - Confirmed 1 swap / 1600 tokens; verified checksum rollback; recovered throughput performance.
- [ ] **Phase 5-D: Performance Median** - Repeated benchmark median across multiple prompts/runs.
- [ ] **Phase 5-E: Multi-Route Transitions** - Validation of mixed-route batch transitions.
- [ ] **Phase 6: 24h Persistent-Route Soak** - Final endurance validation.


### Phase 2: Adapter Migration Diagnostics
- Provide automated reports detailing:
  - Architecture homology scores.
  - Required calibration dataset sizes.
  - Outlier preservation and adapter norm drift warnings.
  - Perplexity (PPL) and KL Divergence QA Gates.
- Instead of promising a perfect conversion, we deliver actionable intelligence: *"Is this LoRA safe to port, or should we retrain?"*

### Phase 3: Enterprise Adapter Migration Kit
- **Private/On-Premise Diagnostics:** Executable natively on air-gapped clusters to protect proprietary weights.
- **Enterprise Batch Diagnostics:** API and CLI tools to process hundreds of adapters automatically across `metrics.json` outputs.
- **License Risk Checking:** Auto-detection of underlying model and adapter licenses to flag potential copyleft contamination.
- Safe-mode fallback generation (low-intensity adapters).
- Continuous drift monitoring during inference (Hot-Swap API).

---

## 4. Conclusion

By shifting our focus from "We can convert anything" to "**We can diagnose portability and execute safe, verified projections**," Neural-Scalpel provides immediate value to developers managing large LoRA assets during model refresh cycles, while maintaining the rigorous skepticism required for tier-1 ML research.