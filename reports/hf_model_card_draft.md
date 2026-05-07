# Qwen2.5-0.5B SQL Structural Projection LoRA

This adapter is an experimental no-retraining structural projection of a Qwen2.5-7B SQL LoRA into Qwen2.5-0.5B-Instruct.

It is not a fully trained SQL model and does not guarantee general SQL improvement.

## Benchmarks (SQL-50)

On the Neural-Scalpel SQL-50 benchmark, this adapter improved:
- **Execution Accuracy:** 32.0% → 36.0%
- **Execution Success:** 38.0% → 44.0%
- **Syntax Validity:** 37/50 → 40/50

**Best tested alpha:** 16.

## Qualitative Improvements
- **Fixed Hallucinations:** Corrects cases where the base model produces conversational text or "Explanation" blocks instead of pure SQL.
- **Join Logic:** Improved handling of multi-table joins and subquery constraints.

## Technical Details
This adapter was generated using the **Neural-Scalpel** framework via Structural Projection (RSVD-based weight delta transport). It approximates and compresses the source adapter's weight-delta structure into a PEFT-compatible LoRA for Qwen2.5-0.5B-Instruct.

## Limitations
- Evaluated only on the project-specific SQL-50 benchmark.
- Not validated on Spider, BIRD, or production text-to-SQL workloads.
- Improvements are modest and task-dependent.
- This adapter may still fail on complex schemas, ambiguous natural language, or multi-hop SQL queries.
- This is a structural projection baseline, not a distilled or fine-tuned SQL model.

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "ponpoke/qwen2.5-0.5b-sql-structural-projection-lora")
```
