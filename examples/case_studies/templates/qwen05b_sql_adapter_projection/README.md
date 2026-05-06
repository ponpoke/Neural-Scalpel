# Qwen2.5-0.5B SQL Adapter Projection Case Study Template

> [!WARNING]
> **Status: Structural Projection Baseline v2 / Behavioral Validation Inconclusive**  
> This case study has completed real source-adapter inspection, structural projection, target-shape verification, PEFT load smoke testing, and a 4-prompt real-inference smoke evaluation.  
> **Behavioral improvement is not proven.** Under greedy decoding, Base and Projected outputs were identical across the initial 4-prompt smoke set.

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
- [ ] Behavioral improvement: **NOT PROVEN**
- [ ] SQL parse / execution metrics: **PENDING**
- [ ] Activation-Calibrated Projection (Phase 4): **PLANNED**

### Technical Findings (Phase 3 Preliminary)
1.  **Identity Check**: Under greedy decoding, the projected adapter produced outputs **bit-identical** to the base model across all 4 initial smoke prompts.
2.  **Repetition/Truncation**: Both base and projected models exhibited similar truncation patterns at 128 tokens.
3.  **Instruction Following**: The 0.5B model's base instruction-following capability is dominant; the projected adapter weights (transplanted from a 7B model) provided no observable delta in this minimal test.
4.  **Inconclusive Evidence**: These results are inconclusive and suggest that simple structural projection may be insufficient for measurable behavior transfer without activation-based alignment or higher injection scales ($\gamma$).

## Research Roadmap (Phase 4: Activation Alignment)
- **Phase 4**: Implement **Activation-Calibrated Projection** to align representation subspaces using real task activation data.
- **Scale Sweep**: Test higher `scale_gamma` values to determine if the adapter signal can be amplified without causing divergence.

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.

## License

This project is licensed under the Apache 2.0 License.
