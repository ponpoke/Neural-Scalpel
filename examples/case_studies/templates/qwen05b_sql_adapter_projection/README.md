# Qwen2.5-0.5B SQL Adapter Projection Case Study Template

> [!WARNING]
> **Status: Case Study Template / Simulation Mode**  
> This directory is a reusable scaffold for building a real Qwen2.5-0.5B SQL/Coding adapter projection case study.  
> **It is not a completed evaluation.** Real-weight validation has not been completed yet. Scripts that generate inspection, metrics, runtime validation, and model-card reports use simulated values by default.

## Overview

Can a tiny 0.5B model inherit useful SQL/Coding behavior without retraining? This template provides a structured scaffold for testing that question through mathematical adapter projection.

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
Real validation is currently staged. Payload generation and static integrity checks can be run with `--real` once a redistributable source adapter is selected. Real inference, SQL metric evaluation, and runtime validation require additional output-generation scripts and are not yet fully automated in this scaffold.

```bash
# 0. Prepare manual license verification report
# This creates a manual-verification-ready report. It does not automatically grant redistribution rights.
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
```

## Results Summary

> [!IMPORTANT]
> **Real results are pending.** The current reports are scaffold outputs only.

> [!WARNING]
> **Status: Structural Projection Baseline v2 / Behavioral Validation Pending**  
> This case study has completed real source-adapter inspection, structural projection, artifact generation, and initial compatibility smoke tests.  
> It is **not yet a completed behavioral evaluation**. Real before/after inference and SQL task performance metrics remain pending.

**Current Task Status:**
- [x] Real source adapter license verification: **COMPLETED**
- [x] Real source adapter inspection: **COMPLETED**
- [x] Structural projection baseline v2: **COMPLETED** (Interpolated Folding)
- [x] Target-shape verification: **COMPLETED** (**PASS** - RUNTIME_SHAPE_VERIFIED)
- [x] PEFT adapter smoke test: **COMPLETED** (**PASS** - Loaded & Generated 1 token)
- [x] Calibrated scaling: **IMPLEMENTED** (Default: $\gamma=0.5$ for safety)
- [ ] Neural-Scalpel runtime route smoke test: **PENDING**
- [ ] Behavioral benchmark (Spider/BIRD): **STAGED**

### Technical Findings (Phase 2: Structural Baseline v2)
The architectural bridge has been established using the **Structural Projection Baseline v2** pipeline:
1.  **Interpolated Layer Folding**: 28 source layers were mapped to 24 target layers using weighted linear interpolation, ensuring continuity of weights across depths.
2.  **Runtime-Compatible Shape Alignment**: 100% of the 96 payload tensors (including vLLM-fused layers) were verified against the target model state_dict. Status: **RUNTIME_SHAPE_VERIFIED**.
3.  **PEFT Loadability**: The generated adapter was successfully loaded into a standard PEFT/Transformers environment and executed a single-token generation smoke test without failure.
4.  **High SVD Energy Retention**: SVD analysis shows a **Mean Energy Retention of 0.9580**, indicating that the structural resizing preserved ~96% of the singular-value energy of the source matrices.
5.  **Calibrated Intensity**: Implemented `--scale-gamma` to control the delta norm ratio, allowing for safer initial behavioral tests.

## Research Roadmap (Phase 3: Behavioral Validation)
- **Phase 3**: Execute comparative inference on SQL tasks to determine if the 0.5B model effectively inherited the 7B SQL ability.
- **Phase 4**: Move toward **Activation-Calibrated Projection** (Phase 3 Advanced) to align representation subspaces using real task activation data.

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.

## License

This project is licensed under the Apache 2.0 License.
