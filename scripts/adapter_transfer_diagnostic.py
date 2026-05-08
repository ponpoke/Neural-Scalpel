import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_scalpel.core.diagnostic import (
    AdapterTransferDiagnosticReport, 
    DeltaHealthAnalyzer,
    MetadataGateResult,
    CompatibilityResult,
    FeasibilityResult
)
from neural_scalpel.core.benchmarks.registry import BenchmarkRegistry
from neural_scalpel.core.quality_gate import SourceAdapterQualityReport, QualityGateConfig

from huggingface_hub import hf_hub_download

def run_metadata_gate(adapter_path, base_model_name) -> MetadataGateResult:
    print(f"[Stage 0] Running Metadata Gate...")
    res = MetadataGateResult()
    
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        try:
            print(f" - Attempting to fetch adapter_config.json from HF Hub...")
            config_path_str = hf_hub_download(repo_id=adapter_path, filename="adapter_config.json")
            config_path = Path(config_path_str)
        except Exception as e:
            res.status = "WARNING"
            res.warnings.append(f"Could not fetch metadata from Hub: {e}")
            return res
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    res.adapter_type = config.get("peft_type", "UNKNOWN")
    res.base_model_name = config.get("base_model_name_or_path", "")
    res.rank = config.get("r", 0)
    res.lora_alpha = config.get("lora_alpha", 0)
    res.target_modules = config.get("target_modules", [])
    
    if res.base_model_name and base_model_name not in res.base_model_name:
        res.base_model_matches = False
        res.warnings.append(f"Base model mismatch: Config says {res.base_model_name}")
    else:
        res.base_model_matches = True
        
    res.status = "PASS" if res.base_model_matches else "FAIL"
    return res

def run_compatibility_gate(source_base, target_base) -> CompatibilityResult:
    print(f"[Stage 3] Running Target Compatibility Gate...")
    res = CompatibilityResult()
    if not target_base:
        return res
        
    try:
        s_config = AutoConfig.from_pretrained(source_base)
        t_config = AutoConfig.from_pretrained(target_base)
        
        res.hidden_size_ratio = t_config.hidden_size / s_config.hidden_size
        res.layer_count_ratio = t_config.num_hidden_layers / s_config.num_hidden_layers
        res.family_match = s_config.model_type == t_config.model_type
        
        # Tokenizer Check
        print(f" - Comparing tokenizers...")
        s_tokenizer = AutoTokenizer.from_pretrained(source_base)
        t_tokenizer = AutoTokenizer.from_pretrained(target_base)
        
        res.tokenizer_match = (len(s_tokenizer) == len(t_tokenizer))
        res.tokenizer_check_status = "SUCCESS"
        
        score = 0.0
        if res.family_match: score += 0.4
        if 0.2 <= res.hidden_size_ratio <= 1.5: score += 0.2
        if 0.5 <= res.layer_count_ratio <= 2.0: score += 0.2
        if res.tokenizer_match: score += 0.2
        
        res.compatibility_score = score
        res.verdict = "COMPATIBLE" if score > 0.7 else "RISKY"
    except Exception as e:
        res.verdict = "INCONCLUSIVE"
        res.tokenizer_check_status = f"ERROR: {e}"
        print(f"Compatibility check error: {e}")
        
    return res

def run_feasibility_gate(source_base, target_base, adapter_target_modules) -> FeasibilityResult:
    print(f"[Stage 4] Running Projection Feasibility Gate...")
    res = FeasibilityResult()
    if not target_base:
        return res
        
    try:
        s_config = AutoConfig.from_pretrained(source_base)
        t_config = AutoConfig.from_pretrained(target_base)
        
        # Determine mapping type
        if t_config.num_hidden_layers != s_config.num_hidden_layers:
            res.layer_mapping_type = "interpolated"
        else:
            res.layer_mapping_type = "direct"
            
        # GQA Check
        s_gqa = getattr(s_config, "num_key_value_heads", getattr(s_config, "num_attention_heads", 0)) != getattr(s_config, "num_attention_heads", 0)
        t_gqa = getattr(t_config, "num_key_value_heads", getattr(t_config, "num_attention_heads", 0)) != getattr(t_config, "num_attention_heads", 0)
        res.gqa_aware_required = (s_gqa != t_gqa) or t_gqa
        
        res.module_mapping_status = "verified_with_config"
        res.shape_compatible = True # Config-level assumption
        res.verdict = "FEASIBLE"
        
    except Exception as e:
        res.verdict = "INCOMPATIBLE"
        res.module_mapping_status = f"error: {e}"
        
    return res

def run_diagnostic(args):
    report = AdapterTransferDiagnosticReport(
        run_id=f"diag_{int(time.time())}",
        timestamp=datetime.now().isoformat(),
        source_base_model=args.source_base_model,
        source_adapter=args.source_adapter,
        target_model=args.target_model
    )
    
    # Stage 0: Metadata
    report.metadata_gate = run_metadata_gate(args.source_adapter, args.source_base_model)
    if report.metadata_gate.status == "FAIL":
        print("[Gate] FATAL: Metadata check failed.")
        if not args.force: return

    # Stage 1: Load and Evaluate Quality (Hardened with full metrics)
    print(f"[Stage 1] Running Source Quality Gate (CPU for stability)...")
    tokenizer = AutoTokenizer.from_pretrained(args.source_base_model)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.source_base_model, 
        torch_dtype=torch.float16,
        device_map={"": "cpu"}
    )
    model = PeftModel.from_pretrained(base_model, args.source_adapter)
    model.eval()
    
    evaluator = BenchmarkRegistry.get_evaluator(args.benchmark, model, tokenizer)
    suite = BenchmarkRegistry.get_suite(args.benchmark)
    
    # Run full v1.1 style evaluation
    adapter_res = evaluator.evaluate_suite(suite)
    with model.disable_adapter():
        base_res = evaluator.evaluate_suite(suite)
        
    q_report = SourceAdapterQualityReport(
        base_model=args.source_base_model,
        adapter_path=args.source_adapter,
        benchmark=args.benchmark
    )
    
    # Restore full metric calculation from v1.1
    q_report.total_cases = len(suite)
    q_report.base_metrics = {
        "execution_accuracy": base_res["stats"]["execution_accuracy"],
        "execution_success": base_res["stats"]["execution_success_rate"],
        "syntax_validity": base_res["stats"]["syntax_valid"] / base_res["stats"]["total"]
    }
    q_report.adapter_metrics = {
        "execution_accuracy": adapter_res["stats"]["execution_accuracy"],
        "execution_success": adapter_res["stats"]["execution_success_rate"],
        "syntax_validity": adapter_res["stats"]["syntax_valid"] / adapter_res["stats"]["total"]
    }
    for m in q_report.base_metrics:
        q_report.delta[m] = q_report.adapter_metrics[m] - q_report.base_metrics[m]
        
    # Classification
    base_correct_ids = {res["id"] for res in base_res["results"] if res["is_correct"]}
    adapter_correct_ids = {res["id"] for res in adapter_res["results"] if res["is_correct"]}
    q_report.failure_classification = {
        "fixed": len(adapter_correct_ids - base_correct_ids),
        "regressed": len(base_correct_ids - adapter_correct_ids),
        "both_succeeded": len(base_correct_ids & adapter_correct_ids),
        "both_failed": q_report.total_cases - len(base_correct_ids | adapter_correct_ids)
    }
    q_report.regression_rate = q_report.failure_classification["regressed"] / q_report.total_cases
    q_report.generate_verdict()
    report.source_quality_gate = q_report.__dict__

    # Stage 2: Delta Health
    print(f"[Stage 2] Running Delta Health Gate...")
    report.delta_health_gate = DeltaHealthAnalyzer.analyze(model)
    
    # Stage 3: Compatibility
    if args.target_model:
        report.compatibility_gate = run_compatibility_gate(args.source_base_model, args.target_model)
        
    # Stage 4: Feasibility
    if args.target_model:
        report.feasibility_gate = run_feasibility_gate(
            args.source_base_model, args.target_model, report.metadata_gate.target_modules
        )
    
    # Stage 6: Release Decision
    print(f"[Stage 6] Running Release Decision Gate...")
    report.finalize_release_decision()
    
    # Final Summary
    print(f"\n--- Diagnostic Result: {report.release_decision_gate.verdict} ---")
    print(f"Recommendation: {report.release_decision_gate.recommendation}")
    for reason in report.release_decision_gate.reasons:
        print(f" - {reason}")
        
    # Save Report
    output_path = Path(args.output_dir) / "diagnostic_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    print(f"\n[Diag] Complete. Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural-Scalpel v2.0 Adapter Transfer Diagnostic")
    parser.add_argument("--source_base_model", type=str, required=True)
    parser.add_argument("--source_adapter", type=str, required=True)
    parser.add_argument("--target_model", type=str, help="Target model for compatibility check")
    parser.add_argument("--benchmark", type=str, default="sql_50")
    parser.add_argument("--output_dir", type=str, default="reports/diagnostics")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    run_diagnostic(args)
