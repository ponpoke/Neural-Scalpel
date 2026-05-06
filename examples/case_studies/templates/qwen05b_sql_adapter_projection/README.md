# Qwen2.5-0.5B SQL Adapter Projection Case Study

> [!WARNING]
> **Status: Phase 5-F-4 Preliminary Runtime Success / SQL Capability Transfer NOT YET VERIFIED**  
> A PEFT-loadable LoRA artifact was produced from paired-alignment-derived module-level solves and was observed to change the Qwen2.5-0.5B target model's generation behavior. However, task-level SQL capability transfer is **NOT PROVEN** and requires Phase 6 evaluation.

## Current Result: Phase 5-F PEFT LoRA Export

Phase 5-F-4 produced a PEFT-loadable LoRA artifact from paired-alignment-derived module-level solves. Unlike the earlier target-only projection path, the exported LoRA produced observable generation changes in the Qwen2.5-0.5B target model.

### Key Observations
- **Technical Validation**: The exported LoRA passed PEFT load validation and was confirmed to have non-zero LoRA parameters in model memory.
- **Runtime Behavior**: At high scale (`lora_alpha=32`), the adapter produced strong generation changes but triggered repetition / mode-collapse. At a lower scale (`lora_alpha=16`), generation became more stable while maintaining observable behavioral influence.
- **Qualitative Shift**: In preliminary examples, the adapted model produced more advanced SQL structures such as CTEs and window functions compared with the base model's simpler queries.

### Interpretation
This is a **positive runtime result**: under the tested setup, the paired-alignment pipeline can produce a PEFT-style adapter signal that measurably influences the target model's generation behavior.

However, this does **not** yet prove successful SQL skill transfer or production-quality task improvement. The current result should be interpreted as:
> **Successful signal delivery and preliminary qualitative behavior change, not yet verified SQL capability transfer.**

## Phase 6: Initial SQL Capability Evaluation

Phase 6 evaluated whether the qualitative SQL shifts observed in Phase 5 translate into parseable and schema-consistent SQL outputs.

### Initial 4-sample smoke result
- **Overall parse success**: 75.0% → 75.0%  
  This includes one non-SQL Python prompt. On the SQL-only subset, all 3 SQL prompts were parseable for both Base and LoRA outputs.
- **Advanced SQL structure rate**: 0.0% → 25.0%  
  One projected output introduced a more advanced SQL structure such as a CTE or window-function pattern.
- **Schema hallucination rate**: 0.0% → 0.0%  
  No new table hallucination was detected in this small smoke set.
- **Repetition / collapse rate**: no increase observed under the tested `lora_alpha=16` setting.
- **Verdict**: `BEHAVIORAL_SHIFT_ONLY`

### Interpretation
This is an encouraging initial result. The projected LoRA changed SQL-generation behavior without degrading parse success or causing schema hallucination in the current 4-sample smoke set.

However, this does **not** yet prove SQL capability transfer. The observed improvement is structural and qualitative, not yet validated by execution accuracy or larger benchmark coverage. The current result should be interpreted as:
> **preliminary evidence of SQL-style behavioral shift, not yet verified SQL task improvement.**

## Next Steps: Phase 6 Full Evaluation
- **Large-scale Evaluation**: Expand to the full 50-prompt controlled SQL set.
- **Metric Refinement**: Report SQL-only metrics separately from non-SQL prompts.
- **Execution Validation**: Compare Base vs LoRA on execution results against controlled SQLite databases.
- **Failure Case Analysis**: Track overcomplication: cases where the LoRA uses CTE/window functions unnecessarily or incorrectly.

See [reports/](reports/) for generated validation logs.
