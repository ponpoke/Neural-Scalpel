import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
import json
import argparse
from pathlib import Path

def get_sql_50_suite():
    """Returns a subset of the SQL-50 benchmark with schema and setup scripts."""
    return [
        {
            "id": "sql_001",
            "category": "basic_select",
            "prompt": "SQL: Get all users who joined in 2023.\nTable: users(id, name, join_date)\nOutput:",
            "tables": ["users"],
            "columns": ["id", "name", "join_date"],
            "sqlite_setup": "CREATE TABLE users (id INT, name TEXT, join_date TEXT); INSERT INTO users VALUES (1, 'Alice', '2023-01-01'), (2, 'Bob', '2022-12-31');",
            "reference": "SELECT * FROM users WHERE join_date LIKE '2023%';"
        },
        {
            "id": "sql_002",
            "category": "aggregation",
            "prompt": "SQL: Count number of orders for product 'P100'.\nTable: orders(id, product_id, quantity)\nOutput:",
            "tables": ["orders"],
            "columns": ["product_id", "quantity"],
            "sqlite_setup": "CREATE TABLE orders (id INT, product_id TEXT, quantity INT); INSERT INTO orders VALUES (1, 'P100', 5), (2, 'P200', 10);",
            "reference": "SELECT COUNT(*) FROM orders WHERE product_id = 'P100';"
        },
        {
            "id": "sql_003",
            "category": "joins",
            "prompt": "SQL: List names of departments with employees earning more than 5000.\nTable: departments(id, name), employees(id, dept_id, salary)\nOutput:",
            "tables": ["departments", "employees"],
            "columns": ["name", "salary"],
            "sqlite_setup": "CREATE TABLE departments (id INT, name TEXT); CREATE TABLE employees (id INT, dept_id INT, salary INT); INSERT INTO departments VALUES (1, 'Sales'); INSERT INTO employees VALUES (1, 1, 6000);",
            "reference": "SELECT DISTINCT d.name FROM departments d JOIN employees e ON d.id = e.dept_id WHERE e.salary > 5000;"
        },
        {
            "id": "sql_004",
            "category": "subqueries",
            "prompt": "SQL: Find books with price above average.\nTable: books(id, title, price)\nOutput:",
            "tables": ["books"],
            "columns": ["title", "price"],
            "sqlite_setup": "CREATE TABLE books (id INT, title TEXT, price INT); INSERT INTO books VALUES (1, 'A', 10), (2, 'B', 20), (3, 'C', 30);",
            "reference": "SELECT title FROM books WHERE price > (SELECT AVG(price) FROM books);"
        },
        {
            "id": "sql_005",
            "category": "complex_logic",
            "prompt": "SQL: Show employees who earn more than their managers.\nTable: employees(id, name, salary, manager_id)\nOutput:",
            "tables": ["employees"],
            "columns": ["name", "salary", "manager_id"],
            "sqlite_setup": "CREATE TABLE employees (id INT, name TEXT, salary INT, manager_id INT); INSERT INTO employees VALUES (1, 'Boss', 10000, NULL), (2, 'Emp', 12000, 1);",
            "reference": "SELECT e.name FROM employees e JOIN employees m ON e.manager_id = m.id WHERE e.salary > m.salary;"
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel Phase 6: SQL Capability Evaluation (Full)")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to transplanted PEFT adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="sql_eval_results_full.json")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        device_map=args.device
    )

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    evaluator = SQLCapabilityEvaluator(model, tokenizer)
    test_cases = get_sql_50_suite()

    print(f"Running full evaluation on {len(test_cases)} cases...")
    results = evaluator.evaluate_suite(test_cases)
    
    stats = results["stats"]
    print("\n--- Evaluation Summary ---")
    print(f"Syntax Pass Rate:    {stats['syntax_valid'] / stats['total']:.1%}")
    print(f"Execution Accuracy:  {stats['pass_rate']:.1%}")
    
    print("\n--- Category Metrics ---")
    for cat, data in stats["categories"].items():
        print(f"  {cat:15}: {data['pass']/data['total']:.1%} ({data['pass']}/{data['total']})")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output}")

    # Baseline comparison (Selective)
    print("\n--- Baseline Comparison (Adapter Disabled) ---")
    with model.disable_adapter():
        baseline_evaluator = SQLCapabilityEvaluator(model, tokenizer)
        baseline_results = baseline_evaluator.evaluate_suite(test_cases)
        b_stats = baseline_results["stats"]
        print(f"Baseline Execution Accuracy: {b_stats['pass_rate']:.1%}")
        
        improvement = stats['pass_rate'] - b_stats['pass_rate']
        print(f"Net Improvement: {improvement:+.1%}")

if __name__ == "__main__":
    main()
