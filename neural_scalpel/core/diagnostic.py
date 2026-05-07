from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import torch
import os
from pathlib import Path

@dataclass
class MetadataGateResult:
    status: str = "PENDING"
    adapter_type: str = ""
    base_model_name: str = ""
    base_model_matches: bool = False
    rank: int = 0
    lora_alpha: int = 0
    target_modules: List[str] = field(default_factory=list)
    license: str = "UNKNOWN"
    warnings: List[str] = field(default_factory=list)

@dataclass
class DeltaHealthResult:
    verdict: str = "PENDING"
    global_frobenius_norm: float = 0.0
    module_norms: Dict[str, float] = field(default_factory=dict)
    layer_distribution_entropy: float = 0.0
    effective_rank_mean: float = 0.0
    outliers: List[str] = field(default_factory=list)

@dataclass
class CompatibilityResult:
    verdict: str = "PENDING"
    compatibility_score: float = 0.0
    hidden_size_ratio: float = 0.0
    layer_count_ratio: float = 0.0
    tokenizer_match: Optional[bool] = None
    tokenizer_check_status: str = "NOT_CHECKED"
    family_match: bool = False

@dataclass
class FeasibilityResult:
    verdict: str = "PENDING"
    layer_mapping_type: str = "PENDING"
    module_mapping_status: str = "PENDING"
    shape_compatible: bool = False
    gqa_aware_required: bool = False
    unsupported_modules: List[str] = field(default_factory=list)

@dataclass
class ReleaseDecision:
    verdict: str = "PENDING"
    recommendation: str = ""
    reasons: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)

@dataclass
class AdapterTransferDiagnosticReport:
    """The ultimate multi-stage diagnostic report for Neural-Scalpel v2.0."""
    schema_version: str = "adapter_transfer_diagnostic.v2.0.1"
    run_id: str = ""
    timestamp: str = ""
    
    # Target Info (Optional for source-only diagnostic)
    source_base_model: str = ""
    source_adapter: str = ""
    target_model: Optional[str] = None
    
    # Stages
    metadata_gate: MetadataGateResult = field(default_factory=MetadataGateResult)
    source_quality_gate: Dict[str, Any] = field(default_factory=dict) # v1.1 structure
    delta_health_gate: DeltaHealthResult = field(default_factory=DeltaHealthResult)
    compatibility_gate: CompatibilityResult = field(default_factory=CompatibilityResult)
    feasibility_gate: FeasibilityResult = field(default_factory=FeasibilityResult)
    target_evaluation_gate: Dict[str, Any] = field(default_factory=dict)
    release_decision_gate: ReleaseDecision = field(default_factory=ReleaseDecision)

    def to_json(self) -> str:
        def dfilter(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        return json.dumps(self, default=dfilter, indent=2, ensure_ascii=False)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    def finalize_release_decision(self):
        """Unified multi-stage logic for release determination."""
        reasons = []
        
        # 1. Source Quality Check
        source_verdict = self.source_quality_gate.get("verdict", "INCONCLUSIVE")
        source_status = self.source_quality_gate.get("gate_status", "FAIL")
        
        # 2. Health and Feasibility
        health_verdict = self.delta_health_gate.verdict
        feasibility_status = self.feasibility_gate.verdict
        
        # 3. Target Eval
        target_eval_verdict = self.target_evaluation_gate.get("verdict", "PENDING")
        
        # Decision Logic: Renamed to reflect strict pipeline
        if source_verdict == "POSITIVE_TEACHER" and source_status == "PASS":
            if health_verdict == "HEALTHY_DELTA":
                if feasibility_status == "FEASIBLE":
                    if target_eval_verdict == "POSITIVE_TARGET_TRANSFER":
                        self.release_decision_gate.verdict = "RELEASE_READY"
                        self.release_decision_gate.recommendation = "PUBLISH_WITH_FULL_BENCHMARKS"
                        reasons.append("End-to-end success: Improved performance on target model.")
                    elif target_eval_verdict == "PENDING":
                        self.release_decision_gate.verdict = "PROJECTION_CANDIDATE"
                        self.release_decision_gate.recommendation = "PROCEED_TO_TARGET_EVALUATION"
                        reasons.append("Source quality and feasibility confirmed. Awaiting target evaluation.")
                    else:
                        self.release_decision_gate.verdict = "RESEARCH_ONLY"
                        self.release_decision_gate.recommendation = "ANALYZE_TARGET_INTERFERENCE"
                        reasons.append("Source good, but failed to improve target model.")
                else:
                    self.release_decision_gate.verdict = "SOURCE_READY"
                    self.release_decision_gate.recommendation = "CHECK_ARCHITECTURE_COMPATIBILITY"
                    reasons.append("High-quality source adapter but structural projection is technically pending/failed.")
            else:
                self.release_decision_gate.verdict = "RESEARCH_ONLY"
                reasons.append(f"Source good, but delta health is {health_verdict}.")
        else:
            self.release_decision_gate.verdict = "INCONCLUSIVE"
            reasons.append("Primary source quality gate not satisfied.")

        self.release_decision_gate.reasons = reasons
        self.release_decision_gate.required_artifacts = ["diagnostic_report.json"]

class DeltaHealthAnalyzer:
    """Analyzes LoRA weights for Stage 2 (Delta Health Gate)."""
    @staticmethod
    def analyze(model) -> DeltaHealthResult:
        result = DeltaHealthResult()
        total_norm = 0.0
        norms = {}
        
        # Analysis logic (simplified for implementation)
        for name, param in model.named_parameters():
            if "lora_" in name:
                n = torch.norm(param.data.float()).item()
                norms[name] = n
                total_norm += n**2
        
        result.global_frobenius_norm = total_norm**0.5
        result.module_norms = norms
        
        if result.global_frobenius_norm > 500.0: # Arbitrary heuristic
            result.verdict = "OVERPOWERED_DELTA"
        elif result.global_frobenius_norm < 0.01:
            result.verdict = "LOW_SIGNAL_DELTA"
        else:
            result.verdict = "HEALTHY_DELTA"
            
        return result
