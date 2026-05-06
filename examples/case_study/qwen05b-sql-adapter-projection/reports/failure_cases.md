# Observed Failure Cases

> [!NOTE]
> These failure cases are illustrative examples for the scaffold.

- **Schema Hallucination:** In 2 cases, the route referenced table names not present in the prompt.
- **Redundant Aliasing:** Occasional over-aliasing in simple SELECT queries.
- **Logic Drift:** One case where a JOIN was preferred over a subquery, resulting in a correct but less efficient plan.
