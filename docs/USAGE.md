# Neural-Scalpel Usage Guide (v2.10)

Neural-Scalpel provides a robust CLI for cross-architecture intelligence transplantation.

---

## 1. safe-project Pipeline (Recommended)

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

> [!NOTE]
> `safe-project` automatically upgrades the projection mode to `piecewise` if diagnostics detect spectral instability or concentrated layers.

---

## 2. project-adapter (Advanced)

Use this for fine-grained control over the weight projection process.

```bash
neural-scalpel project-adapter \
  --source-adapter ./path/to/source_lora \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --output ./projected_adapter \
  --rank 16 \
  --alpha 16 \
  --projection-mode piecewise \
  --module-alpha-map q_proj=4,k_proj=4,v_proj=4,o_proj=4,gate_proj=0.125,up_proj=0.125,down_proj=0
```

### Interference-Aware Gating (v2.10)

Use `--module-alpha-map` to assign different effective alpha values to each module family. Modules with alpha=0 are physically excluded from the generated adapter.

- **`--alpha 16`**: This is the global PEFT alpha stored in `adapter_config.json`.
- **`--module-alpha-map`**: Rescales individual LoRA deltas to the requested effective alpha.
- **`alpha=0`**: Means physical exclusion from the weight file, not just zero-weight inclusion.

---

### Projection Modes (`--projection-mode`)
- `linear` (Default): Standard Procrustes alignment. Stable and validated.
- `piecewise`: (v2.8) Hardened energy-aware splitting with LoRA-pair reconstruction. Best for preserving high-level reasoning.
- `kernel`: (v2.9 Research Stub) Placeholder for future activation-based Kernel Orthogonal Procrustes.
- `jacobian`: (v2.9 Research Stub) Placeholder for Jacobian Tangent Space Alignment.

---

## 3. diagnose-adapter (v2.6+)

Analyze the "health" and spectral properties of your adapter.

```bash
neural-scalpel diagnose-adapter \
  --source-adapter ./source_lora \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --output ./reports
```

### Advanced Metrics (Hardened v2.6)
- **Normalized Spectral Entropy**: Intelligence distribution across singular components (0.0 to 1.0).
- **Normalized Effective Rank**: Real information density relative to total rank.
- **Concentration Score**: Detects if single layers dominate the adapter (Risk of "Robotomy").

---

## 4. evaluate-projected

Evaluate a projected adapter against a benchmark.

```bash
neural-scalpel evaluate-projected \
  --target Qwen/Qwen2.5-Coder-0.5B \
  --adapter ./projected_adapter \
  --benchmark sql_50
```

---

## 5. Report & Model Card Automation (v2.3+)

Generate human-readable scientific reports and standardized model cards for Hugging Face.

```bash
neural-scalpel generate-report --report-path ./runs/my_release/diagnostic_report.json
neural-scalpel generate-model-card --report-path ./runs/my_release/diagnostic_report.json
```

---

## 6. package-release & Traceability (v2.4+)

Bundle weights and scientific evidence into a single distribution-ready folder.

```bash
neural-scalpel package-release \
  --run-dir ./runs/my_release \
  --adapter-dir ./runs/my_release/projected_adapter \
  --output-dir ./release/v1.0.0
```

---

## 7. package-validate (v2.5+)

Verify the authenticity and integrity of a release package before deployment.

```bash
neural-scalpel package-validate --package-dir ./release/v1.0.0
```

---

## 8. Adaptive Scaling & Traceability (v2.7+)

Neural-Scalpel automatically modulates layer-wise strength based on health data.
- **Auto-Dampening**: Outlier or unstable layers are automatically scaled down to prevent performance collapse.
- **Auditing**: You can find exactly which scales were applied in the `applied_scales` section of the diagnostic report.
- **Configuration**: Use `--adaptive-scaling-config` to customize the dampening thresholds.