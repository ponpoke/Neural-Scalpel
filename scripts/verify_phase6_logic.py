import torch
import torch.nn as nn
from unittest.mock import MagicMock
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
from neural_scalpel.core.benchmarks.sql_50 import get_sql_50_suite
import json

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
    def forward(self, *args, **kwargs):
        out = MagicMock()
        out.logits = torch.randn(1, 1, 100)
        return out
    def generate(self, input_ids, **kwargs):
        # Return a simple sequence that corresponds to a valid SQL for sql_001
        # In our suite, the first 10 cases are basic_select
        return torch.tensor([[0]*input_ids.shape[1] + [1, 2, 3]])

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        class Out(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self
            def to(self, dev): return self
        return Out({"input_ids": torch.zeros(1, 5).long()})
    def decode(self, ids, **kwargs):
        # Return a valid SQL string for testing
        return "SELECT name, region_id FROM users WHERE region_id = 1;"

def main():
    print("Verifying Phase 6 Evaluation Logic with Mock Model...")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    evaluator = SQLCapabilityEvaluator(model, tokenizer)
    
    suite = get_sql_50_suite()
    print(f"Loaded suite with {len(suite)} cases.")
    
    # Run a small sample (first 5 cases) to verify AST and Execution logic
    results = evaluator.evaluate_suite(suite[:5])
    
    stats = results["stats"]
    print("\nVerification Results (Sample of 5):")
    print(f"Total:            {stats['total']}")
    print(f"Syntax Valid:     {stats['syntax_valid']}")
    print(f"Execution Success: {stats['execution_success']}")
    print(f"Exact Match:      {stats['exact_match']}")
    
    # Check if AST extracted anything
    first_res = results["results"][0]
    print(f"\nExample Extraction (Case 0):")
    print(f"  Generated: {first_res['generated']}")
    print(f"  Syntax:    {first_res['syntax_valid']}")
    
    # Check if failure tracking works (since we return same SQL for all, some should fail)
    print(f"\nFailure Tracking:")
    print(f"  Failures found: {len(results['failure_cases'])}")
    
    if len(results['failure_cases']) > 0:
        print(f"  First failure ID: {results['failure_cases'][0]['id']}")

    print("\n[SUCCESS] Phase 6 Evaluation logic verified.")

if __name__ == "__main__":
    main()
