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
- [x] Real source adapter license verification: **COMPLETED**
- [x] Real source adapter inspection: **COMPLETED**
- [x] Baseline payload generation: **COMPLETED** (745.51 MB Dense Payload)
- [x] Config-based shape validation: **COMPLETED** (PASS - GQA-aware)
- [ ] Runtime state_dict validation: **STAGED**
- [ ] Real before/after inference: **STAGED**

### Technical Findings (Phase 2: Structural Projection Baseline)
We have implemented and executed a **Structural Bilinear Resize + SVD Recompression** pipeline to bridge the architectural gap between Qwen2.5-7B and Qwen2.5-0.5B:
1.  **GQA-Aware Mapping**: Successfully calculated target shapes for Q/K/V heads, ensuring that $d_{kv}$ (Grouped Query Attention dimension) is correctly handled for the 0.5B architecture.
2.  **Uniform Layer Sampling**: Mapped 28 source layers to 24 target layers using a uniform sampling strategy. The exact mapping is recorded in the `.scalpel_route` manifest.
3.  **Size Reduction**: Reduced the 12.45GB raw delta to a **745.51 MB rank-limited dense payload** (SVD Rank-16 constraint). While significantly smaller, this remains a dense matrix delta rather than a sparse LoRA adapter.
4.  **Scientific Status**: The current payload is a mathematical baseline. **Behavioral inheritance (SQL performance) and actual runtime loadability have not yet been verified.**

## Research Roadmap (Phase 3: Runtime & Behavioral Validation)
To move beyond the structural baseline, the following steps are required:
- **Phase 3**: Verify the payload against the actual vLLM runtime `state_dict` to ensure 100% naming and shape alignment.
- **Phase 4**: Execute "Before vs. After" inference to determine if the structural projection preserved the source SQL knowledge.

Detailed technical goals can be found in [docs/methodology.md](docs/methodology.md).

See [reports/](reports/) for generated validation logs.

## License

This project is licensed under the Apache 2.0 License.
