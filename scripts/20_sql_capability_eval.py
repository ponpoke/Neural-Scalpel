import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
import json
import argparse
from pathlib import Path

def get_sql_eval_subset():
    """Returns a representative subset of SQL evaluation cases with expected results."""
    return [
        {
            "id": "sql_001",
            "category": "basic_select",
            "prompt": "SQL: Get all users who joined in 2023.\nTable: users(id, name, join_date)\nOutput:",
            "tables": ["users"],
            "columns": ["id", "name", "join_date"],
            "sqlite_setup": "CREATE TABLE users (id INT, name TEXT, join_date TEXT); INSERT INTO users VALUES (1, 'Alice', '2023-01-01'), (2, 'Bob', '2022-12-31');",
            "expected_result": [(1, 'Alice', '2023-01-01')],
            "order_sensitive": False,
            "reference": "SELECT * FROM users WHERE join_date LIKE '2023%';"
        },
        {
            "id": "sql_002",
            "category": "aggregation",
            "prompt": "SQL: Count number of orders for product 'P100'.\nTable: orders(id, product_id, quantity)\nOutput:",
            "tables": ["orders"],
            "columns": ["product_id", "quantity"],
            "sqlite_setup": "CREATE TABLE orders (id INT, product_id TEXT, quantity INT); INSERT INTO orders VALUES (1, 'P100', 5), (2, 'P100', 3), (3, 'P200', 10);",
            "expected_result": [(2,)],
            "order_sensitive": False,
            "reference": "SELECT COUNT(*) FROM orders WHERE product_id = 'P100';"
        },
        {
            "id": "sql_003",
            "category": "joins",
            "prompt": "SQL: List names of departments with employees earning more than 5000.\nTable: departments(id, name), employees(id, dept_id, salary)\nOutput:",
            "tables": ["departments", "employees"],
            "columns": ["name", "salary"],
            "sqlite_setup": "CREATE TABLE departments (id INT, name TEXT); CREATE TABLE employees (id INT, dept_id INT, salary INT); INSERT INTO departments VALUES (1, 'Sales'), (2, 'HR'); INSERT INTO employees VALUES (1, 1, 6000), (2, 2, 4000);",
            "expected_result": [('Sales',)],
            "order_sensitive": False,
            "reference": "SELECT DISTINCT d.name FROM departments d JOIN employees e ON d.id = e.dept_id WHERE e.salary > 5000;"
        },
        {
            "id": "sql_004",
            "category": "subqueries",
            "prompt": "SQL: Find books with price above average.\nTable: books(id, title, price)\nOutput:",
            "tables": ["books"],
            "columns": ["title", "price"],
            "sqlite_setup": "CREATE TABLE books (id INT, title TEXT, price INT); INSERT INTO books VALUES (1, 'A', 10), (2, 'B', 20), (3, 'C', 30);",
            "expected_result": [('C',)],
            "order_sensitive": False,
            "reference": "SELECT title FROM books WHERE price > (SELECT AVG(price) FROM books);"
        },
        {
            "id": "sql_005",
            "category": "complex_logic",
            "prompt": "SQL: Show employees who earn more than their managers.\nTable: employees(id, name, salary, manager_id)\nOutput:",
            "tables": ["employees"],
            "columns": ["name", "salary", "manager_id"],
            "sqlite_setup": "CREATE TABLE employees (id INT, name TEXT, salary INT, manager_id INT); INSERT INTO employees VALUES (1, 'Boss', 10000, NULL), (2, 'Emp', 12000, 1);",
            "expected_result": [('Emp',)],
            "order_sensitive": False,
            "reference": "SELECT e.name FROM employees e JOIN employees m ON e.manager_id = m.id WHERE e.salary > m.salary;"
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel Phase 6: SQL Capability Evaluation Pipeline")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to transplanted PEFT adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="sql_eval_results_consolidated.json")
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

    test_cases = get_sql_eval_subset()
    final_output = {
        "metadata": {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "num_cases": len(test_cases)
        },
        "adapter_results": {},
        "baseline_results": {},
        "comparison": {}
    }

    # 1. Evaluate Adapter
    print(f"\nEvaluating Adapter on {len(test_cases)} cases...")
    adapter_evaluator = SQLCapabilityEvaluator(model, tokenizer)
    adapter_res = adapter_evaluator.evaluate_suite(test_cases)
    final_output["adapter_results"] = adapter_res
    
    a_stats = adapter_res["stats"]
    print("\n--- Adapter Summary ---")
    print(f"Syntax Pass Rate:           {a_stats['syntax_valid'] / a_stats['total']:.1%}")
    print(f"SQLite Success Rate:        {a_stats['execution_success_rate']:.1%}")
    print(f"Execution Accuracy (Match): {a_stats['execution_accuracy']:.1%}")

    # 2. Evaluate Baseline
    print("\nEvaluating Baseline (Adapter Disabled)...")
    with model.disable_adapter():
        baseline_evaluator = SQLCapabilityEvaluator(model, tokenizer)
        baseline_res = baseline_evaluator.evaluate_suite(test_cases)
        final_output["baseline_results"] = baseline_res
        
        b_stats = baseline_res["stats"]
        print("\n--- Baseline Summary ---")
        print(f"Syntax Pass Rate:           {b_stats['syntax_valid'] / b_stats['total']:.1%}")
        print(f"SQLite Success Rate:        {b_stats['execution_success_rate']:.1%}")
        print(f"Execution Accuracy (Match): {b_stats['execution_accuracy']:.1%}")

    # 3. Consolidate Comparison
    final_output["comparison"] = {
        "execution_success_delta": a_stats["execution_success_rate"] - b_stats["execution_success_rate"],
        "execution_accuracy_delta": a_stats["execution_accuracy"] - b_stats["execution_accuracy"],
        "syntax_pass_delta": (a_stats['syntax_valid'] - b_stats['syntax_valid']) / a_stats['total']
    }
    
    print("\n--- Comparison ---")
    print(f"Success Delta:  {final_output['comparison']['execution_success_delta']:+.1%}")
    print(f"Accuracy Delta: {final_output['comparison']['execution_accuracy_delta']:+.1%}")

    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\nConsolidated results saved to {args.output}")

if __name__ == "__main__":
    main()
