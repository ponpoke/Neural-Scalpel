# Source Adapter Quality Gate Report

## Summary

| Field | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Adapter | jk200201/qwen2.5-coder-7b-sql-dpo |
| Benchmark | sql_50 |
| **Verdict** | **POSITIVE_TEACHER** |
| **Status** | **PASS** |
| Recommendation | PROCEED_TO_PROJECTION |

## Metrics

| Metric | Base | Adapter | Delta |
|---|---:|---:|---:|
| execution_accuracy | 62.0% | 78.0% | +16.0% |
| execution_success | 84.0% | 100.0% | +16.0% |
| syntax_validity | 88.0% | 100.0% | +12.0% |

## Diagnostic Analysis

| Analysis | Value |
|---|---:|
| Total Cases | 50 |
| Regression Rate | 2.0% |
| Collapse detected | 0.0% |
| Empty output rate | 0.0% |
| Repetition rate | 0.0% |
| Regression rate | 2.0% |

## Failure Classification

| Type | Count |
|---|---:|
| Fixed | 9 |
| Regressed | 1 |
| Both succeeded | 30 |
| Both failed | 10 |

## Metadata

```json
{
  "base_eval_mode": "peft_disable_adapter",
  "metadata_validation_mode": "skipped_for_hub",
  "warnings": [],
  "torch_dtype": "torch.float16"
}
```
