# Reproducibility Guide

To reproduce the Neural-Scalpel runtime results locally:

## Requirements
- OS: Linux (Ubuntu 22.04+)
- GPU: NVIDIA (Compute Capability 8.0+)
- Python: 3.10+
- vLLM: 0.20.1 / V1 engine

## Reproduce Phase 5-D/E/F Controlled Validation

```bash
PYTHONPATH=. python scripts/run_phase_5d_median.py \
  --runs 3 \
  --prompts 50 \
  --output reports/phase_5d_repeated_median.json

PYTHONPATH=. python scripts/run_phase_5e_mixed_batch.py \
  --requests 1000 \
  --max-tokens 16 \
  --output reports/phase_5e_mixed_batch.json

PYTHONPATH=. python scripts/run_phase_5f_determinism.py \
  --max-tokens 32 \
  --output reports/phase_5f_determinism.json
```

Expected high-level results:

* Phase 5-D: Scalpel median throughput above Native LoRA under controlled conditions
* Phase 5-E-1: 0 route violations, 0 quarantine events
* Phase 5-F: checksum rollback verified, exact text match, 100.0% top-token logprob trace similarity under tested cache-reset condition

## Dynamic Routing Evaluation

To evaluate the dynamic routing capabilities and run the 1000-request mixed-batch endurance test:
```bash
git clone https://github.com/ponpoke/Neural-Scalpel.git
cd Neural-Scalpel
pip install -e .
python examples/run_dynamic_route_demo.py
```

## Expected Output
You should observe exactly 0 violations, and `swap_count` must equal `rollback_count`.
VRAM growth should be <=100MB after warmup for soak validation.
