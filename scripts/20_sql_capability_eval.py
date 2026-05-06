import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel Phase 6: SQL Capability Evaluation")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to transplanted PEFT adapter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="sql_eval_results.json")
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

    # Controlled SQL-50 Mock/Selection
    test_cases = [
        {
            "prompt": "SQL: Get all users who joined in 2023.\nTable: users(id, name, join_date)\nOutput:",
            "reference": "SELECT * FROM users WHERE join_date LIKE '2023%';"
        },
        {
            "prompt": "SQL: Count number of orders for product 'P100'.\nTable: orders(id, product_id, quantity)\nOutput:",
            "reference": "SELECT COUNT(*) FROM orders WHERE product_id = 'P100';"
        },
        {
            "prompt": "SQL: List names of departments with more than 10 employees.\nTable: departments(id, name), employees(id, dept_id)\nOutput:",
            "reference": "SELECT name FROM departments JOIN employees ON departments.id = employees.dept_id GROUP BY departments.id HAVING COUNT(*) > 10;"
        },
        {
            "prompt": "SQL: Find the average price of books in the 'Sci-Fi' category.\nTable: books(id, title, category, price)\nOutput:",
            "reference": "SELECT AVG(price) FROM books WHERE category = 'Sci-Fi';"
        },
        {
            "prompt": "SQL: Show the names of employees who earn more than their managers.\nTable: employees(id, name, salary, manager_id)\nOutput:",
            "reference": "SELECT e.name FROM employees e JOIN employees m ON e.manager_id = m.id WHERE e.salary > m.salary;"
        }
    ]

    print("Running evaluation...")
    results = evaluator.evaluate_suite(test_cases)
    
    print(f"Pass Rate (Syntax): {results['pass_rate']:.1%}")
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

    # Baseline comparison
    print("\n--- Baseline Comparison (Adapter Disabled) ---")
    with model.disable_adapter():
        baseline_evaluator = SQLCapabilityEvaluator(model, tokenizer)
        baseline_results = baseline_evaluator.evaluate_suite(test_cases)
        print(f"Baseline Pass Rate (Syntax): {baseline_results['pass_rate']:.1%}")

if __name__ == "__main__":
    main()
