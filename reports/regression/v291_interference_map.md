# v2.9.1 Interference Map (Case-Level Analysis)

## 1. Case-Level Matrix

| Case ID | Category | Baseline | attention_a4 | attention_a6 | attention_a8 | attention_a12 | attention_a16 | mlp_a4 | down_proj_a4 |
|---| ---| ---| ---| ---| ---| ---| ---| ---| ---| |
| `aggregation_001` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_002` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_003` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_004` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_005` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_006` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_007` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_008` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_009` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `aggregation_010` | aggregation | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `basic_select_001` | basic | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS |
| `basic_select_002` | basic | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `basic_select_003` | basic | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS |
| `basic_select_004` | basic | PASS | PASS | PASS | FAIL | FAIL | FAIL | PASS | FAIL |
| `basic_select_005` | basic | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `basic_select_006` | basic | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS |
| `basic_select_007` | basic | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `basic_select_008` | basic | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `basic_select_009` | basic | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `basic_select_010` | basic | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_001` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_002` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_003` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_004` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_005` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_006` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_007` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_008` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_009` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `complex_logic_010` | complex | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `joins_001` | joins | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `joins_002` | joins | FAIL | FAIL | PASS | FAIL | PASS | PASS | FAIL | FAIL |
| `joins_003` | joins | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `joins_004` | joins | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `joins_005` | joins | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| `joins_006` | joins | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS |
| `joins_007` | joins | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `joins_008` | joins | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `joins_009` | joins | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `joins_010` | joins | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_001` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_002` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_003` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_004` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_005` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_006` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_007` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_008` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_009` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `subqueries_010` | subqueries | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## 2. Transition Summary

| Setting | Fixed | Regressed | Net Delta | PASS/PASS | FAIL/FAIL |
|---|---:|---:|---:|---:|---:|
| attention_a4 | 0 | 0 | +0 | 12 | 38 |
| attention_a6 | 1 | 1 | +0 | 11 | 37 |
| attention_a8 | 0 | 2 | -2 | 10 | 38 |
| attention_a12 | 1 | 5 | -4 | 7 | 37 |
| attention_a16 | 2 | 6 | -4 | 6 | 36 |
| mlp_a4 | 0 | 1 | -1 | 11 | 38 |
| down_proj_a4 | 0 | 2 | -2 | 10 | 38 |

## 3. Deep Analysis: attention_a6 Trade-off (Trade-off Candidate)

### 3.1 Fixed Cases (Knowledge Injection Success)

#### Case: `joins_002`
- **Prompt Snippet**: `SQL: Names of products in category 2.
Table: products(id, name, cat_id), categories(id, name)
Output...`
- **Baseline Error**: Invalid expression / Unexpected token. Line 2, Col: 12.
  ['Product1', 'Product2']
Explanation[4m:[0m 
- Product1 belongs to category 1.
- Product2 belongs to category 2.

```python
import pandas as pd
- **Adapter Generated SQL**:
```sql
-- No SQL
```

### 3.2 Regressed Cases (Interference Victim)

#### Case: `joins_007`
- **Baseline SQL (Correct)**:
```sql

```
- **Adapter SQL (Regressed)**:
```sql

```
- **Adapter Error**: Invalid expression / Unexpected token. Line 2, Col: 12.
  ['Product 1', 'Product 2', 'Product 3']
Explanation[4m:[0m 
- Product 1 is in category 7.
- Product 2 is in category 7.
- Product 3 is in category 7.

```pyth

