import os
import argparse
from pathlib import Path

def eval_before_after(real_inference=False):
    mode = "REAL" if real_inference else "SIMULATED"
    print(f"[Phase 4] Performing qualitative evaluation ({mode})")
    
    if real_inference:
        raise NotImplementedError(
            "Real before/after inference is not implemented in this scaffold. "
            "Generate real outputs with a dedicated inference script, then pass them to this report generator."
        )
        
    examples = [
        {
            "prompt": "Given a table orders(order_id, customer_id, total, created_at), write SQL to find the top 5 customers by total revenue in 2024.",
            "base_output": "SELECT customer_id, SUM(total) FROM orders WHERE created_at LIKE '2024%' GROUP BY customer_id ORDER BY SUM(total) DESC LIMIT 5;",
            "projected_output": "SELECT customer_id, SUM(total) AS total_revenue FROM orders WHERE created_at >= '2024-01-01' AND created_at <= '2024-12-31' GROUP BY customer_id ORDER BY total_revenue DESC LIMIT 5;",
            "observation": "Projected route used more robust date filtering and aliased the sum column, matching SQL-centric adapter behavior."
        },
        {
            "prompt": "Write a Python function to check if a string is a palindrome.",
            "base_output": "def is_palindrome(s):\n    return s == s[::-1]",
            "projected_output": "def is_palindrome(text: str) -> bool:\n    \"\"\"Checks if a string is a palindrome.\"\"\"\n    processed_text = ''.join(e for e in text.lower() if e.isalnum())\n    return processed_text == processed_text[::-1]",
            "observation": "Projected route produced a more robust and type-hinted implementation."
        }
    ]
    
    report_path = Path("reports/before_after.md")
    os.makedirs(report_path.parent, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative Before/After Evaluation ({mode})\n\n")
        if not real_inference:
            f.write(f"> [!CAUTION]\n")
            f.write(f"> **SIMULATED EXAMPLES**: These are placeholders for the model card template.\n")
            f.write(f"> Real inference has not been executed for this report.\n\n")
            
        for i, ex in enumerate(examples):
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Prompt:**\n```\n{ex['prompt']}\n```\n\n")
            f.write(f"**Base Qwen2.5-0.5B:**\n```\n{ex['base_output']}\n```\n\n")
            f.write(f"**Projected SQL Route:**\n```\n{ex['projected_output']}\n```\n\n")
            f.write(f"**Observation:** {ex['observation']}\n\n")
            f.write("---\n\n")
            
    print(f"Qualitative evaluation saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Perform real inference evaluation")
    args = parser.parse_args()

    eval_before_after(real_inference=args.real)
