# Qualitative Before/After Evaluation (REAL)

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct
**Adapter:** routes/qwen05b_sql_projection/peft_adapter

### Example 1

**Prompt:**
```
Table: sales(id, product, amount, date). SQL to find total sales for 'Laptop'.
```

**Base Output:**
```
SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id;

SELECT SUM(amount) FROM sales WHERE product = 'Laptop'; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop';

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop'; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY
```

**Projected Output:**
```
SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id;

SELECT SUM(amount) FROM sales WHERE product = 'Laptop'; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop';

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY id; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop'; 

SELECT SUM(amount) FROM sales WHERE product = 'Laptop' GROUP BY
```

---

### Example 2

**Prompt:**
```
Table: employees(id, name, department, salary). SQL for average salary in 'Engineering'.
```

**Base Output:**
```
SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id;

SELECT AVG(salary) FROM employees WHERE department = 'Engineering'; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering';

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering'; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY
```

**Projected Output:**
```
SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering'; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary) FROM employees WHERE department = 'Engineering' GROUP BY id; 

SELECT AVG(salary)
```

---

### Example 3

**Prompt:**
```
Write a SQL query to list all students who scored above 90 in 'Mathematics'.
```

**Base Output:**
```
Additionally, include the following additional requirements:

1. Exclude any student whose name starts with "A" or ends with "Z".
2. Only consider students who have taken at least one course.
3. Sort the results by their score in descending order and then by their name in ascending order.

Assume that there is a table named `students` with columns `name`, `math_score`, and `course_taken`. The table has a column `is_passed` indicating whether the student passed the course (1 for passed, 0 for failed).

```sql
SELECT *
FROM students
WHERE math_score > 90 AND name NOT
```

**Projected Output:**
```
Additionally, include the following additional requirements:

1. Exclude any student whose name starts with "A" or ends with "Z".
2. Only consider students who have taken at least one course.
3. Sort the results by their score in descending order and then by their name in ascending order.

Assume that there is a table named `students` with columns `name`, `math_score`, and `course_taken`. The table has a column `is_passed` indicating whether the student passed the course (1 for passed, 0 for failed).

```sql
SELECT *
FROM students
WHERE math_score > 90 AND name NOT
```

---

### Example 4

**Prompt:**
```
Write a Python function for palindrome check.
```

**Base Output:**
```
The function should take in two strings as input and return True if the first string is a palindrome, False otherwise.

Example:
Input: "racecar"
Output: True

Input: "hello"
Output: False

Input: ""
Output: True

Note:
- The function should ignore spaces and punctuation.
- The function should not use any built-in functions or libraries for checking palindromes.
- The function should be case-insensitive.
- The function should handle both uppercase and lowercase letters.
def is_palindrome(s):
    # Convert the string to lowercase
    s = s.lower()
    
    # Remove all non-al
```

**Projected Output:**
```
The function should take in two strings as input and return True if the first string is a palindrome of the second string, and False otherwise.

Example:
Input: "racecar"
Output: True

Input: "hello"
Output: False

Input: "madam"
Output: True

Input: "python"
Output: False
def is_palindrome(s):
    # Remove all non-alphanumeric characters from the string
    s = ''.join(char.lower() for char in s if char.isalnum())
    
    # Check if the cleaned string is equal to its reverse
    return s == s[::-1]
```

---

