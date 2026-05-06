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

## Next Steps: Phase 6 SQL Capability Evaluation
- **Parse Validation**: Measure SQL parse success rate using `sqlglot`.
- **Schema Validation**: Check whether generated queries use valid tables and columns defined in the prompt.
- **Execution Benchmarking**: Run generated queries against a controlled database and compare execution results.
- **Robustness Analysis**: Measure repetition rate, max-length hit rate, and failure cases across a larger prompt set.
- **Refinement**: Evaluate lower/higher `lora_alpha`, rank variants, and multi-module signal distribution (e.g., adding `o_proj`).

See [reports/](reports/) for generated validation logs.
