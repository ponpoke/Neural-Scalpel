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

