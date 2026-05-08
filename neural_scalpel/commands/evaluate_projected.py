import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator

def run_evaluate(args):
    """Modularized evaluation entry point, synchronized with main.py logic."""
    print(f"\n[Layer 6: Evaluation] Running benchmark: {args.benchmark}")
    
    torch_dtype = torch.float16 if getattr(args, "dtype", "float16") == "float16" else torch.float32
    
    print(f"Loading base model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    
    if hasattr(args, "adapter") and args.adapter:
        print(f"Applying adapter: {args.adapter} (Merge: {getattr(args, 'merge_adapter', False)})")
        model = PeftModel.from_pretrained(model, args.adapter)
        if getattr(args, "merge_adapter", False):
            model = model.merge_and_unload()
    
    evaluator = SQLCapabilityEvaluator(model=model, tokenizer=tokenizer)
    
    # Load dataset based on benchmark name
    if args.benchmark == "sql_50":
        import sys
        # Ensure the evaluation suite directory is in path (common setup in this repo)
        benchmark_path = os.path.abspath("qwen2.5-0.5b-sql-structural-projection")
        if benchmark_path not in sys.path:
            sys.path.append(benchmark_path)
        try:
            from eval.sql_50_suite_definition import get_sql_50_suite
            suite = get_sql_50_suite()
            eval_results = evaluator.evaluate_suite(suite)
        except ImportError as e:
            raise RuntimeError(f"Failed to load SQL-50 benchmark from {benchmark_path}. Error: {e}")
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # Prepare results in standard format (matching main.py/scratch expectations)
    output_results = {
        "stats": eval_results["stats"],
        "results": eval_results["results"],
        "eval_metadata": {
            "eval_dtype": getattr(args, "dtype", "float16"),
            "adapter_merge": bool(getattr(args, "merge_adapter", False)),
            "target_model": args.target_model,
            "adapter": getattr(args, "adapter", None)
        }
    }
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2, default=lambda x: str(x))
        print(f"[SUCCESS] Evaluation results saved to {args.output}")
    
    # Print summary metrics
    stats = output_results["stats"]
    print(f"Accuracy: {stats['execution_accuracy']:.2%}")
    print(f"Syntax Valid: {stats['syntax_valid']}/{stats['total']}")

def add_evaluate_projected_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-projected",
        help="Evaluate a projected adapter or baseline on target model benchmarks."
    )

    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model path or ID")
    parser.add_argument("--adapter", required=False, default=None,
                        help="Path to projected adapter (optional for baseline evaluation)")
    parser.add_argument("--benchmark", default="sql_50",
                        help="Benchmark to run (default: sql_50)")
    parser.add_argument("--output", help="Path to save evaluation JSON results")
    parser.add_argument("--dtype", default="float16", help="Precision for evaluation (default: float16)")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge adapter weights before evaluation")

    parser.set_defaults(func=run_evaluate)
