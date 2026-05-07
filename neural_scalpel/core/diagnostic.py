from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import torch
import math
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
    # Advanced Metrics (v2.6)
    spectral_entropy: float = 0.0
    effective_rank: float = 0.0
    concentration_score: float = 0.0 # 0.0 to 1.0
    layer_wise_entropy: float = 0.0
    outliers: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

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
class TargetEvaluationResult:
    verdict: str = "PENDING"
    total_cases: int = 0
    base_metrics: Dict[str, float] = field(default_factory=dict)
    adapter_metrics: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    failure_classification: Dict[str, int] = field(default_factory=dict)
    regression_rate: float = 0.0

@dataclass
class ReleaseDecision:
    verdict: str = "PENDING"
    recommendation: str = ""
    reasons: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)

@dataclass
class AdapterTransferDiagnosticReport:
    """The ultimate multi-stage diagnostic report for Neural-Scalpel v2.6."""
    schema_version: str = "adapter_transfer_diagnostic.v2.6"
    run_id: str = ""
    timestamp: str = ""
    
    source_base_model: str = ""
    source_adapter: str = ""
    target_model: Optional[str] = None
    
    # Stages
    metadata_gate: MetadataGateResult = field(default_factory=MetadataGateResult)
    source_quality_gate: Dict[str, Any] = field(default_factory=dict)
    delta_health_gate: DeltaHealthResult = field(default_factory=DeltaHealthResult)
    compatibility_gate: CompatibilityResult = field(default_factory=CompatibilityResult)
    feasibility_gate: FeasibilityResult = field(default_factory=FeasibilityResult)
    target_evaluation_gate: TargetEvaluationResult = field(default_factory=TargetEvaluationResult)
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

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        report = cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            source_base_model=data.get("source_base_model", ""),
            source_adapter=data.get("source_adapter", ""),
            target_model=data.get("target_model")
        )
        
        if "metadata_gate" in data: report.metadata_gate = MetadataGateResult(**data["metadata_gate"])
        if "source_quality_gate" in data: report.source_quality_gate = data["source_quality_gate"]
        if "delta_health_gate" in data: report.delta_health_gate = DeltaHealthResult(**data["delta_health_gate"])
        if "compatibility_gate" in data: report.compatibility_gate = CompatibilityResult(**data["compatibility_gate"])
        if "feasibility_gate" in data: report.feasibility_gate = FeasibilityResult(**data["feasibility_gate"])
        if "target_evaluation_gate" in data: report.target_evaluation_gate = TargetEvaluationResult(**data["target_evaluation_gate"])
        if "release_decision_gate" in data: report.release_decision_gate = ReleaseDecision(**data["release_decision_gate"])
        
        return report

    def finalize_release_decision(self):
        """Unified multi-stage logic for release determination."""
        reasons = []
        source_verdict = self.source_quality_gate.get("verdict", "INCONCLUSIVE")
        source_status = self.source_quality_gate.get("gate_status", "FAIL")
        health_verdict = self.delta_health_gate.verdict
        feasibility_status = self.feasibility_gate.verdict
        target_eval_verdict = self.target_evaluation_gate.verdict
        
        artifacts = ["diagnostic_report.json"]
        
        if source_verdict == "POSITIVE_TEACHER" and source_status == "PASS":
            if health_verdict in ["HEALTHY_DELTA", "MODERATELY_CONCENTRATED"]:
                if feasibility_status == "FEASIBLE":
                    if target_eval_verdict == "POSITIVE_TARGET_TRANSFER":
                        self.release_decision_gate.verdict = "RELEASE_READY"
                        self.release_decision_gate.recommendation = "PUBLISH_WITH_FULL_BENCHMARKS"
                        reasons.append("End-to-end success: Improved performance on target model.")
                        artifacts += ["target_evaluation_results.json", "projected_adapter/", "README.md"]
                    elif target_eval_verdict == "PENDING":
                        self.release_decision_gate.verdict = "PROJECTION_CANDIDATE"
                        self.release_decision_gate.recommendation = "PROCEED_TO_TARGET_EVALUATION"
                        reasons.append("Source quality and health confirmed. Spectral entropy suggests stable transfer.")
                        artifacts += ["projection_metadata.json"]
                    else:
                        self.release_decision_gate.verdict = "RESEARCH_ONLY"
                        reasons.append(f"Source good, but target evaluation result is {target_eval_verdict}.")
                else:
                    self.release_decision_gate.verdict = "SOURCE_READY"
                    self.release_decision_gate.recommendation = "CHECK_ARCHITECTURE_COMPATIBILITY"
                    reasons.append("High-quality source but structural projection is infeasible.")
            else:
                self.release_decision_gate.verdict = "RESEARCH_ONLY"
                reasons.append(f"Unhealthy delta distribution detected: {health_verdict}.")
                for r in self.delta_health_gate.reasons:
                    reasons.append(f"Health Detail: {r}")
        else:
            self.release_decision_gate.verdict = "INCONCLUSIVE"
            reasons.append("Primary source quality gate not satisfied.")

        self.release_decision_gate.reasons = reasons
        self.release_decision_gate.required_artifacts = sorted(list(set(artifacts)))

class DeltaHealthAnalyzer:
    """Advanced Spectral Delta Health Analysis (v2.6)."""
    @staticmethod
    def analyze(model) -> DeltaHealthResult:
        result = DeltaHealthResult()
        total_sq_norm = 0.0
        module_norms = {}
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if "lora_" in name:
                n = torch.norm(param.data.float()).item()
                module_norms[name] = n
                total_sq_norm += n**2
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p.isdigit():
                        layer_idx = int(p)
                        layer_norms[layer_idx] = layer_norms.get(layer_idx, 0.0) + n**2
                        break

        result.global_frobenius_norm = total_sq_norm**0.5
        result.module_norms = module_norms
        
        if result.global_frobenius_norm == 0:
            result.verdict = "LOW_SIGNAL_DELTA"
            result.reasons.append("No adapter weights found.")
            return result

        # Spectral Analysis
        all_singular_values = []
        params = dict(model.named_parameters())
        for name, param in params.items():
            if "lora_B" in name:
                b_weight = param.data.float()
                a_name = name.replace("lora_B", "lora_A")
                if a_name in params:
                    a_weight = params[a_name].data.float()
                    delta_w = b_weight @ a_weight
                    s = torch.linalg.svdvals(delta_w)
                    all_singular_values.extend(s.tolist())

        if all_singular_values:
            s_tensor = torch.tensor(all_singular_values)
            s_norm = s_tensor / s_tensor.sum()
            entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10)).item()
            result.spectral_entropy = entropy
            result.effective_rank = torch.exp(torch.tensor(entropy)).item()

        if layer_norms:
            layer_vals = torch.tensor(list(layer_norms.values()))
            layer_probs = layer_vals / layer_vals.sum()
            result.layer_wise_entropy = -torch.sum(layer_probs * torch.log(layer_probs + 1e-10)).item()
            max_layer_contribution = (layer_vals.max() / layer_vals.sum()).item()
            result.concentration_score = max_layer_contribution
            if max_layer_contribution > 0.5:
                result.outliers.append(f"Layer {torch.argmax(layer_vals).item()} (>{max_layer_contribution:.1%})")

        reasons = []
        if result.global_frobenius_norm > 800.0:
            result.verdict = "OVERPOWERED_DELTA"
            reasons.append("Global norm exceeds safety threshold.")
        elif result.global_frobenius_norm < 0.05:
            result.verdict = "LOW_SIGNAL_DELTA"
            reasons.append("Delta magnitude is too low.")
        elif result.concentration_score > 0.7:
            result.verdict = "CRITICALLY_CONCENTRATED"
            reasons.append(f"Single layer dominates {result.concentration_score:.1%} of delta.")
        elif result.spectral_entropy < 0.5:
            result.verdict = "LOW_SPECTRAL_ENTROPY"
            reasons.append("Delta energy is concentrated in too few singular components.")
        else:
            result.verdict = "HEALTHY_DELTA" if result.concentration_score <= 0.3 else "MODERATELY_CONCENTRATED"
            reasons.append("Delta energy is well-distributed.")

        result.reasons = reasons
        return result
