# Route Injection Quality Evaluation Report

**Model:** Qwen/Qwen2.5-0.5B | **Device:** cuda

| Mode | PPL | KL Div | Code Pass | Rep Rate | Entropy |
|------|-----|--------|-----------|----------|--------|
| Target Base | 14.9346 | -0.003887 | 16/25 | 0.0619 | 5.55 |
| Target + Naive | 124.7942 | 127.437500 | 2/25 | 0.5475 | 2.78 |
| Target + Random LR | 14.8842 | 0.030838 | 16/25 | 0.0710 | 5.61 |
| Target + Projected | 14.8849 | 0.016205 | 16/25 | 0.0681 | 5.64 |
| Target + Actual LoRA | 17.5804 | 7.320312 | 16/25 | 0.1363 | 4.82 |
| After Rollback | 14.9346 | -0.003887 | 16/25 | 0.0619 | 5.55 |

**Rollback Integrity:** PPL=PASS, KL=PASS, Code=PASS

*Generated: 2026-05-05 06:34*
