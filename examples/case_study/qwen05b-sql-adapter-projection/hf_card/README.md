# Qwen2.5-0.5B SQL Adapter Projection via Neural-Scalpel

> [!CAUTION]
> **SIMULATED CASE-STUDY SCAFFOLD**: Real-weight evaluation has NOT been completed yet.
> The metrics and examples below are placeholders for demonstration purposes.

## What is this?

This is an experimental projected SQL/Coding adapter for Qwen2.5-0.5B, produced with Neural-Scalpel.

It tests whether task behavior from a larger Qwen-family SQL/Coding adapter can survive projection into a much smaller 0.5B target model without gradient-based retraining.

## Key Result

# Quantitative Evaluation Summary (SIMULATED)

> [!CAUTION]
> **SIMULATED METRICS**: Do not use as real evaluation results.
> These numbers are placeholders from the case-study scaffold.

Evaluated on 50 curated SQL/Coding prompts.

| Metric | Base Qwen2.5-0.5B | Projected SQL Route | Delta |
| :--- | :---: | :---: | :---: |
| Syntax Validity | 82% | 94% | +12% |
| Execution Accuracy | 64% | 78% | +14% |
| Exact Match | 58% | 72% | +14% |
| Judge Preference | 40% | 60% | +20% |


## Before / After

# Qualitative Before/After Evaluation (SIMULATED)

> [!CAUTION]
> **SIMULATED EXAMPLES**: These are placeholders for the model card template.
> Real inference has not been executed for this report.

### Example 1

**Prompt:**
```
Given a table orders(order_id, customer_id, total, created_at), write SQL to find the top 5 customers by total revenue in 2024.
```

**Base Qwen2.5-0.5B:**
```
SELECT customer_id, SUM(total) FROM orders WHERE created_at LIKE '2024%' GROUP BY customer_id ORDER BY SUM(total) DESC LIMIT 5;
```

**Projected SQL Route:**
```
SELECT customer_id, SUM(total) AS total_revenue FROM orders WHERE created_at >= '2024-01-01' AND created_at <= '2024-12-31' GROUP BY customer_id ORDER BY total_revenue DESC LIMIT 5;
```

**Observation:** Projected route used more robust date filtering and aliased the sum column, matching SQL-centric adapter behavior.

---

### Example 2

**Prompt:**
```
Write a Python function to check if a string is a palindrome.
```

**Base Qwen2.5-0.5B:**
```
def is_palindrome(s):
    return s == s[::-1]
```

**Projected SQL Route:**
```
def is_palindrome(text: str) -> bool:
    """Checks if a string is a palindrome."""
    processed_text = ''.join(e for e in text.lower() if e.isalnum())
    return processed_text == processed_text[::-1]
```

**Observation:** Projected route produced a more robust and type-hinted implementation.

---



## Failure Cases

# Observed Failure Cases

> [!NOTE]
> These failure cases are illustrative examples for the scaffold.

- **Schema Hallucination:** In 2 cases, the route referenced table names not present in the prompt.
- **Redundant Aliasing:** Occasional over-aliasing in simple SELECT queries.
- **Logic Drift:** One case where a JOIN was preferred over a subquery, resulting in a correct but less efficient plan.


## Limitations

- Downstream task improvement is not guaranteed.
- This is a projected adapter experiment, not model distillation.
- Use only after downstream validation.
