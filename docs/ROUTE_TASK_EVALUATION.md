# Route Task Evaluation Plan

## Status

Minimal E2E evaluation implemented. Full scale evaluation pending.

The `examples/route_task_demo.py` script performs a minimal E2E evaluation using a real safetensors payload. It verifies runtime route application and rollback cleanliness through counters and route-safety metrics. It does not claim task improvement or guaranteed output change for the current weak/test delta.

Full-scale empirical validation across a complete dataset is still pending.

## Minimal E2E Real Payload Check

Status: **PASS**

A minimal base → sql-route → base E2E check was executed using the real safetensors payload.

### Results:

- **Runtime swaps:** 32
- **Runtime rollbacks:** 32
- **Route violations:** 0
- **Base before/after output under deterministic sampling:** exact match
- **SQL exact match:** not achieved
- **Route output change:** not observed

### Interpretation:

This validates the E2E harness for real payload application and rollback cleanliness. It does **not** prove Text-to-SQL task improvement, because the current payload behaves like a weak/test delta and did not change the generated output for the prompt.

P2-B, real task-improvement evaluation using a trained task-specific payload and dataset, remains pending.

## Example / Expected Report Format

| Task | Route | Payload Type | Metric | Score | Improvement |
| ---- | ----- | ------------ | ------ | ----- | ----------- |
| Text-to-SQL | `__base__` | None | Exact Match | TBD | - |
| Text-to-SQL | `sql-route`| Safetensors | Exact Match | TBD | TBD |
| Math (GSM8K)| `__base__` | None | Accuracy | TBD | - |
| Math (GSM8K)| `math-route`| Safetensors | Accuracy | TBD | TBD |
