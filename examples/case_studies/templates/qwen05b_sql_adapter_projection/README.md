# Qwen2.5-0.5B SQL Adapter Projection Case Study Template
 
> [!WARNING]
> **Status: Phase 4 Initial Target-Activation-Conditioned Attempt Completed / Behavioral Transfer NOT PROVEN**  
> This case study has implemented an initial **target-activation-conditioned projection attempt** using JTSA-style self-alignment. However, **behavioral improvement remains not proven**. Under greedy decoding, Base and Projected outputs (up to $\gamma=4.0$) were identical across the smoke set.


## Overview

Can a tiny 0.5B model inherit useful SQL/Coding behavior without retraining? This template provides a structured scaffold for testing that question through mathematical adapter projection.

> [!NOTE]
> By default, the scaffold scripts can still run in simulation mode for reporting-flow tests. Real-validation outputs are explicitly labeled separately.

## Project Structure

- `scripts/`: Implementation scripts for each phase (00 to 07).
- `eval/`: Evaluation prompts and database schemas.
- `reports/`: Generated validation and evaluation reports.
- `hf_card/`: Assets for the Hugging Face model card.
- `docs/`: Technical methodology and deep dives.

## How to Reproduce

### 1. Install Dependencies
```bash
pip install -e ../../../../  # Install Neural-Scalpel from parent
pip install -r requirements.txt
```

### 2. Run the Pipeline (Simulation Mode)
By default, scripts run in simulation mode to test the reporting flow.
```bash
python scripts/00_check_licenses.py
python scripts/01_inspect_source_adapter.py
python scripts/02_prepare_payload.py
python scripts/03_check_payload_integrity.py
python scripts/04_eval_before_after.py
python scripts/05_eval_sql_metrics.py
python scripts/06_runtime_validation.py
python scripts/07_make_model_card_assets.py
```

### 3. Run the Pipeline (Real Validation)
Real validation is staged. Payload generation, static integrity checks, PEFT smoke testing, and a small real-inference smoke evaluation are available. Benchmark-level SQL/Coding evaluation remains pending.

```bash
# 0. Prepare manual license verification report
python scripts/00_check_licenses.py --real \
  --source <SOURCE_MODEL_ID> \
  --target Qwen/Qwen2.5-0.5B-Instruct \
  --adapter <LORA_ID>

# 1. Inspect actual source adapter
python scripts/01_inspect_source_adapter.py --real \
  --adapter <ADAPTER_SAFETENSORS> \
  --target Qwen/Qwen2.5-0.5B

# 2. Generate projected payload
python scripts/02_prepare_payload.py --real --lora_id <LORA_ID>

# 3. Check integrity of generated payload
python scripts/03_check_payload_integrity.py --real --payload <PAYLOAD> --manifest <MANIFEST>

# 4. Run real before/after smoke inference (Greedy)
python scripts/04_eval_before_after.py \
  --adapter_path routes/qwen05b_sql_projection/peft_adapter \
  --base_model Qwen/Qwen2.5-0.5B-Instruct

# 5. Generate preliminary heuristic metrics (Identity check)
python scripts/05_real_metrics.py \
  --results_json reports/real_eval_results.json
```

## Results Summary

> [!WARNING]
> **Status: Structural Projection Baseline v2 / Behavioral Validation Inconclusive**  
> This case study has completed structural projection and target-shape verification. However, **behavioral improvement is NOT PROVEN**.  
> Real-inference smoke tests using greedy decoding showed **100% identity** between base and projected outputs on the initial 4-prompt test set.

**Current Task Status:**
- [x] Real source adapter license verification: **COMPLETED**
- [x] Real source adapter inspection: **COMPLETED**
- [x] Structural projection baseline v2: **COMPLETED** (Interpolated Folding)
- [x] Core logic unit testing: **COMPLETED** (Local regression tests established)
- [x] Target-shape verification: **COMPLETED** (**PASS** - RUNTIME_SHAPE_VERIFIED)
- [x] PEFT adapter smoke test: **COMPLETED** (**PASS** - Formally loadable)
- [x] Real qualitative inference: **COMPLETED** (Greedy smoke set)
- [x] Preliminary heuristic metrics: **COMPLETED** (Identity check)
- [x] Behavioral improvement: **NOT PROVEN** (Identical behavior observed)
- [ ] SQL parse / execution metrics: **PENDING**
- [x] Phase 4 Initial Activation-Conditioned Attempt: **SCAFFOLD COMPLETED** (Target-only JTSA Self-Alignment)

### Technical Findings (Phase 3 Preliminary)
1.  **Identity Check**: Even with **Phase 4 Initial Activation Conditioning**, the projected adapter produced outputs **bit-identical** to the base model across all smoke prompts at $\gamma=1.0$ and $\gamma=4.0$.
2.  **Behavioral Gravity**: The 0.5B instruct-tuned base model exhibits extreme "gravity." Target-only activation conditioning (currently using layer means for JTSA-style compensation) is insufficient to produce observable behavior changes in this setup.
3.  **Instruction Following**: The 0.5B model's base instruction-following capability is dominant; the projected adapter weights (transplanted from a 7B model) provided no observable delta in this minimal test.
4.  **Verdict**: The current Activation-Conditioned scaffold is verified as functionally stable, but true behavioral transfer likely requires **Paired Source-Target Alignment** and richer distribution-based projection.

## Research Roadmap (Next Steps)
- **Paired Activation Alignment**: Collect paired source (7B) and target (0.5B) activations using identical prompts to learn explicit cross-model alignment maps.
- **Manifold-Rich Projection**: Extend the projection engine to utilize the stored `std` and `samples` for PCA or Procrustes-based manifold alignment.
- **Target Scale Expansion**: Move testing to 1.5B or 3B target models where the base distribution might be more receptive to adapter signals.

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.

## License

This project is licensed under the Apache 2.0 License.
