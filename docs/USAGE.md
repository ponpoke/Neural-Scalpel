# Neural-Scalpel Usage Guide (v2.9)

Neural-Scalpel provides a robust CLI for cross-architecture intelligence transplantation.

---

## 1. Safe-Project Pipeline (Recommended)

The most secure way to perform transplantation. It runs diagnostics, applies adaptive scaling, and evaluates the result in one automated flow.

```bash
neural-scalpel safe-project \
  --source-base Qwen/Qwen2.5-Coder-7B \
  --source-adapter ./source_lora \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --benchmark sql_50 \
  --output-dir runs/my_release \
  --projection-mode piecewise
```

---

## 2. Advanced Structural Projection

Use this for fine-grained control over the weight projection process.

```bash
neural-scalpel project-adapter \
  --source-adapter ./path/to/source_lora \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --output ./projected_adapter \
  --rank 16 \
  --alpha 16 \
  --projection-mode piecewise
```

### Projection Modes (`--projection-mode`)
- `linear` (Default): Standard Procrustes alignment.
- `piecewise`: (v2.8) Energy-aware splitting. Best for preserving high-level reasoning.
- `kernel`: (v2.9) Kernel Orthogonal Procrustes (KOP) for non-linear manifold alignment.
- `jacobian`: (v2.9) Jacobian Tangent Space Alignment (JTSA) to compensate for activation distortion.

---

## 3. Delta Health Diagnostics (v2.6+)

Analyze the "health" and spectral properties of your adapter.

```bash
neural-scalpel diagnose-adapter \
  --source ./source_lora \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --output ./reports
```

### Key Metrics
- **Spectral Entropy**: Distribution of intelligence across singular components.
- **Effective Rank**: Real information density of the delta weights.
- **Concentration Score**: Detects if single layers dominate the adapter (Risk of instability).

---

## 4. Evaluation of Projected Adapters

Evaluate a projected adapter against a benchmark.

```bash
neural-scalpel evaluate-projected \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --adapter ./projected_adapter \
  --benchmark sql_50
```

---

## 5. Report & Model Card Automation (v2.3+)

Generate human-readable reports and standardized model cards for Hugging Face.

```bash
neural-scalpel generate-report --report-path ./runs/my_release/diagnostic_report.json
neural-scalpel generate-model-card --report-path ./runs/my_release/diagnostic_report.json
```

---

## 6. Release Packaging (v2.4+)

Bundle weights, reports, and metadata into a single, distribution-ready folder.

```bash
neural-scalpel package-release \
  --run-dir ./runs/my_release \
  --adapter-dir ./runs/my_release/projected_adapter \
  --output-dir ./release/v1.0.0
```

---

## 7. Integrity Validation (v2.5+)

Verify the authenticity and integrity of a release package before deployment.

```bash
neural-scalpel package-validate --package-dir ./release/v1.0.0
```

---

## 8. Adaptive Scaling (v2.7+)

Neural-Scalpel automatically modulates layer-wise alpha based on health data:
- **Dampening**: Auto-applied to outlier/unstable layers.
- **Boosting**: Applied to high-signal layers to maximize transfer efficiency.