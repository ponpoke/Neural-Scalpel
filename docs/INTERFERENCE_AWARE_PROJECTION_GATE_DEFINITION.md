# v2.10 Architecture: Interference-Aware Projection Gate (IAPG)

## 1. Problem Statement
In v2.9.1 experiments, we confirmed that global weight projection (Linear Projection) often results in **Interference-Dominant** behavior, especially in small target models (0.5B). Even "safe" modules like Attention cause knowledge regression at moderate alpha levels. The current pipeline lacks a mechanism to protect high-confidence base knowledge from aggressive delta injection.

## 2. Core Concept: Diagnostic-Driven Gating
Instead of blind projection, v2.10 introduces a "Gate" that uses diagnostic evaluation to determine the **Risk Score** of each delta component (Module, Layer, or Case-type).

### Phase 1: Risk Assessment (IAPG v0.1)
- **Module Risk**: Identify which modules (`down_proj`, `o_proj`, etc.) consistently cause regression across a reference benchmark (SQL-50).
- **Layer Risk**: Detect if specific layers (e.g., final layers vs. early layers) are more susceptible to interference.
- **Sentinel Guard**: Automatically block/weaken any projection that causes a regression in predefined "Sentinel Cases" (e.g., `joins_007`).

## 3. Implementation Roadmap

### Stage 1: Diagnostic Registry
Create a standardized way to store and query the "Interference Map" generated from previous runs.
- **Input**: `Baseline` vs. `Projected` result JSONs.
- **Output**: `interference_map.json` (ID-level deltas).

### Stage 2: Module-level Alpha Scaling (Piecewise-Module)
Extend the CLI to support module-specific alpha values.
- **CLI Example**:
  ```bash
  --module-alpha-map "q_proj=8,v_proj=8,down_proj=0.5"
  ```
- **Goal**: Apply strong signals to "safe" modules and ultra-low signals to "high-risk" modules.

### Stage 3: Automated Gating Logic
A script that:
1. Performs a fast "Smoke Test" projection.
2. Evaluates regression on Sentinels.
3. If regression > 0, automatically reduces Alpha for the offending module and re-projects.
4. Iterates until a "Non-destructive improvement" is found.

## 4. Success Criteria
- **Baseline Integrity**: Zero regressions on identified Sentinel cases.
- **Net Accuracy**: > 24.0% (SQL-50) on Qwen2.5-Coder-0.5B-Instruct.
- **Stability**: Syntax Valid > 20/50 even at effective alpha > 8.

---
**Lead Architect**: Antigravity (Google DeepMind)
**Target Release**: v2.10-beta
