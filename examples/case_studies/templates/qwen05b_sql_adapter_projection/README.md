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

**Current Task Status:**
- [ ] Real source adapter license verification: **PENDING**
- [ ] Real source adapter inspection: **PENDING**
- [ ] Real payload generation: **PENDING**
- [ ] Static payload integrity validation: **PENDING**
- [ ] Real before/after inference: **STAGED**
- [ ] Real SQL/Coding metrics: **STAGED**
- [ ] Real runtime validation: **STAGED**

See [reports/](reports/) for generated scaffold logs.

## License

This project is licensed under the Apache 2.0 License.
