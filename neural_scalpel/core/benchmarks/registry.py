from typing import Dict, Type, Any
from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
from neural_scalpel.core.benchmarks.sql_50 import get_sql_50_suite

class BenchmarkRegistry:
    """Registry for managing benchmarks and their respective evaluators."""
    _evaluators = {}
    _suites = {}

    @classmethod
    def register(cls, name: str, evaluator_cls: Type, suite_func: Any):
        cls._evaluators[name] = evaluator_cls
        cls._suites[name] = suite_func

    @classmethod
    def get_evaluator(cls, name: str, model: Any, tokenizer: Any):
        if name not in cls._evaluators:
            raise ValueError(f"Benchmark '{name}' is not registered.")
        return cls._evaluators[name](model, tokenizer)

    @classmethod
    def get_suite(cls, name: str):
        if name not in cls._suites:
            raise ValueError(f"Benchmark '{name}' suite not found.")
        return cls._suites[name]()

# Default registration
BenchmarkRegistry.register("sql_50", SQLCapabilityEvaluator, get_sql_50_suite)

# Task Profiles (Pre-defined configurations for different tasks)
TASK_PROFILES = {
    "sql": {
        "primary_metric": "execution_accuracy",
        "secondary_metrics": ["execution_success", "syntax_validity"],
        "stability_metrics": ["empty_output_rate", "repetition_rate"]
    },
    "code": {
        "primary_metric": "pass_at_1",
        "secondary_metrics": ["syntax_success"],
        "stability_metrics": ["runtime_error_rate"]
    }
}
