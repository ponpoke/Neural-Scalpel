import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from neural_scalpel.core.benchmarks.registry import BenchmarkRegistry
from pathlib import Path

def run_evaluate(args):
    print(f"[Eval] Evaluating projected adapter: {args.adapter}")
    print(f"[Eval] Target Base: {args.target_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    
    evaluator = BenchmarkRegistry.get_evaluator(args.benchmark, model, tokenizer)
    suite = BenchmarkRegistry.get_suite(args.benchmark)
    
    print(f"[Eval] Running benchmark '{args.benchmark}'...")
    results = evaluator.evaluate_suite(suite)
    
    stats = results["stats"]
    print(f"\n--- Evaluation Results ({args.benchmark}) ---")
    print(f"Accuracy: {stats['execution_accuracy']*100:.2f}%")
    print(f"Success Rate: {stats['execution_success_rate']*100:.2f}%")
    print(f"Syntax Valid: {stats['syntax_valid']}/{stats['total']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Detailed results saved to {output_path}")

def add_evaluate_projected_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-projected",
        help="Evaluate a projected adapter on target model benchmarks."
    )

    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model")
    parser.add_argument("--adapter", required=True,
                        help="Path to projected adapter")
    parser.add_argument("--benchmark", default="sql_50",
                        help="Benchmark to run")
    parser.add_argument("--output", default="reports/target_eval/eval_results.json",
                        help="Path to save evaluation results")

    parser.set_defaults(func=run_evaluate)
