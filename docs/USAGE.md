# Neural-Scalpel Usage Guide

This guide provides practical examples for the multi-stage diagnostic and transplantation workflow.

## 1. Installation
Install the package in editable mode with CLI dependencies:
```bash
pip install -e .[cli]
```

---

## 2. Adapter Transfer Diagnostic CLI
Run the 7-stage diagnostic pipeline to evaluate a source adapter. This generates a `diagnostic_report.json`.

```bash
neural-scalpel diagnose-adapter \
    --source-base Qwen/Qwen2.5-Coder-7B-Instruct \
    --source-adapter jk200201/qwen2.5-coder-7b-sql-dpo \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --output-dir reports/diagnostics/qwen_coder_sql
```

---

## 3. LoRA Projection (Experimental)
Project source adapter weights into the target architecture. 

```bash
neural-scalpel project-adapter \
    --source-adapter jk200201/qwen2.5-coder-7b-sql-dpo \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --rank 16 \
    --alpha 16 \
    --output ./qwen25-05b-sql-projected
```

---

## 4. Target Evaluation Gate
Evaluate the projected adapter on the target model benchmarks. Use `--report` to integrate results and finalize the release decision.

```bash
neural-scalpel evaluate-projected \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --adapter ./qwen25-05b-sql-projected \
    --benchmark sql_50 \
    --report reports/diagnostics/qwen_coder_sql/diagnostic_report.json \
    --output reports/target_eval/eval_results.json
```

---

## 5. Safe-Project Orchestrator (v2.2)
**Recommended Workflow:** Run the entire pipeline (Diagnose -> Project -> Evaluate) in a single command.

```bash
neural-scalpel safe-project \
    --source-base Qwen/Qwen2.5-Coder-7B-Instruct \
    --source-adapter jk200201/qwen2.5-coder-7b-sql-dpo \
    --target Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --benchmark sql_50 \
    --rank 16 \
    --alpha 16 \
    --output-dir runs/qwen_sql_transfer
```

---

## 6. Automation & Publishing (v2.3)
Generate human-readable reports and model cards for the results.

### Generate Scientific Report
```bash
neural-scalpel generate-report \
    --run-dir runs/qwen_sql_transfer \
    --output reports/final_analysis.md
```

### Generate Model Card
```bash
neural-scalpel generate-model-card \
    --run-dir runs/qwen_sql_transfer \
    --output ./qwen25-05b-sql-projected/README.md
```

---

## 7. The Python API
For fine-grained control over the math engine:
```python
import torch
from neural_scalpel.core.math import adaptive_variance_preserving_sparsity

# Extract knowledge core
tau_sparse = adaptive_variance_preserving_sparsity(W_tuned, W_base, variance_preservation=0.99)
```

---

## 8. Semantic Routers (`.scalpel_route`)
Load and verify signed route manifests programmatically:
```python
from neural_scalpel.router.manager import ScalpelRouteManager
manager = ScalpelRouteManager(route_dir="./routes")
matrices = manager.verify_and_load_route(filepath="./routes/my_route.scalpel_route", ...)
```

---

## 9. Real Weights Verification
```bash
python examples/verify_real_safetensors.py
```

---

## 10. Release Packaging (v2.4)
Gather all weights, reports, and metadata into a single distribution folder for Hugging Face or Zenodo.

```bash
neural-scalpel package-release \
    --run-dir runs/qwen_sql_transfer \
    --adapter-dir ./qwen25-05b-sql-projected \
    --output-dir release_package/
```

---

## 11. Package Validation (v2.5)
Verify that a release package is authentic, intact, and consistent with its diagnostic reports.

```bash
neural-scalpel package-validate --package-dir ./release_package
```