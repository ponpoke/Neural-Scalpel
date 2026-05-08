import torch
import numpy as np
from safetensors.torch import load_file
import os
import json
import re
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Dict
from datetime import datetime

# --- Legacy / Stable Diagnostic Dataclasses (v2.1 - v2.9) ---

@dataclass
class MetadataGateResult:
    status: str = "UNKNOWN"
    base_model_name: str = ""
    base_model_matches: bool = False
    adapter_type: str = "UNKNOWN"
    rank: int = 0
    lora_alpha: int = 0
    target_modules: List[str] = field(default_factory=list)
    license: str = "unknown"
    warnings: List[str] = field(default_factory=list)

@dataclass
class CompatibilityResult:
    verdict: str = "UNKNOWN"
    score: float = 0.0
    compatibility_score: float = 0.0  # Alias for v2.3+
    hidden_size_ratio: float = 1.0
    layer_count_ratio: float = 1.0
    family_match: bool = False
    tokenizer_match: bool = False
    tokenizer_check_status: str = "PENDING"
    notes: List[str] = field(default_factory=list)

@dataclass
class FeasibilityResult:
    verdict: str = "UNKNOWN"
    feasible: bool = False
    layer_mapping_type: str = "direct"
    gqa_aware_required: bool = False
    module_mapping_status: str = "PENDING"
    shape_compatible: bool = False
    notes: List[str] = field(default_factory=list)

@dataclass
class TargetEvaluationResult:
    verdict: str = "PENDING"
    regression_rate: float = 0.0
    accuracy_delta: float = 0.0
    sentinel_pass: bool = False
    notes: List[str] = field(default_factory=list)

@dataclass
class ReleaseDecisionGateResult:
    verdict: str = "INCONCLUSIVE"
    recommendation: str = ""
    reasons: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    suggested_projection_mode: Optional[str] = None

@dataclass
class DeltaHealthResult:
    verdict: str = "UNKNOWN"
    global_frobenius_norm: float = 0.0
    concentration_score: float = 0.0
    outliers: List[str] = field(default_factory=list)
    normalized_spectral_entropy: float = 1.0
    applied_scales: Dict[str, float] = field(default_factory=dict)
    layer_health: Dict[str, Any] = field(default_factory=dict)

# Compatibility Aliases for test suites
DeltaHealthGateResult = DeltaHealthResult
ReleaseDecision = ReleaseDecisionGateResult

# --- Core Diagnostic Report Class ---

@dataclass
class AdapterTransferDiagnosticReport:
    schema_version: str = "adapter_transfer_diagnostic.v2.6"
    run_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_base_model: str = ""
    source_adapter: str = ""
    target_model: str = ""

    metadata_gate: MetadataGateResult = field(default_factory=MetadataGateResult)
    source_quality_gate: Dict[str, Any] = field(default_factory=dict)
    delta_health_gate: DeltaHealthResult = field(default_factory=DeltaHealthResult)
    compatibility_gate: CompatibilityResult = field(default_factory=CompatibilityResult)
    feasibility_gate: FeasibilityResult = field(default_factory=FeasibilityResult)
    target_evaluation_gate: TargetEvaluationResult = field(default_factory=TargetEvaluationResult)
    release_decision_gate: ReleaseDecisionGateResult = field(default_factory=ReleaseDecisionGateResult)

    def finalize_release_decision(self):
        source_verdict = self.source_quality_gate.get("verdict")
        source_status = self.source_quality_gate.get("gate_status", self.source_quality_gate.get("status"))

        if self.target_evaluation_gate.verdict == "POSITIVE_TARGET_TRANSFER":
            self.release_decision_gate.verdict = "RELEASE_READY"
            self.release_decision_gate.recommendation = "PUBLISH_WITH_FULL_BENCHMARKS"
            self.release_decision_gate.required_artifacts = [
                "diagnostic_report.json",
                "target_evaluation_results.json",
                "projected_adapter/",
                "README.md",
            ]
            return

        if self.target_evaluation_gate.verdict == "TARGET_INTERFERENCE":
            self.release_decision_gate.verdict = "RESEARCH_ONLY"
            self.release_decision_gate.recommendation = "DO_NOT_RELEASE"
            return

        if source_verdict == "POSITIVE_TEACHER" and source_status == "PASS":
            if self.feasibility_gate.verdict == "FEASIBLE" or self.feasibility_gate.feasible:
                if self.delta_health_gate.verdict == "HEALTHY_DELTA":
                    self.release_decision_gate.verdict = "PROJECTION_CANDIDATE"
                    self.release_decision_gate.recommendation = "PROCEED_TO_TARGET_EVALUATION"
                    self.release_decision_gate.required_artifacts = ["diagnostic_report.json"]
                elif self.delta_health_gate.verdict in ["LOW_SPECTRAL_ENTROPY", "MODERATELY_CONCENTRATED"]:
                    self.release_decision_gate.verdict = "PROJECTION_CANDIDATE"
                    self.release_decision_gate.recommendation = "PROCEED_WITH_PIECEWISE_OR_ADAPTIVE_PROJECTION"
                    self.release_decision_gate.required_artifacts = ["diagnostic_report.json"]
                    self.release_decision_gate.suggested_projection_mode = "piecewise"
                else:
                    self.release_decision_gate.verdict = "SOURCE_READY"
            else:
                self.release_decision_gate.verdict = "SOURCE_READY"
        else:
            self.release_decision_gate.verdict = "INCONCLUSIVE"
        
        # Populate reasons if applicable
        if self.release_decision_gate.verdict == "INCONCLUSIVE":
            self.release_decision_gate.reasons = ["Source quality or metadata not verified"]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=lambda x: str(x))

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        report = cls(
            schema_version=data.get("schema_version", "adapter_transfer_diagnostic.v2.6"),
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            source_base_model=data.get("source_base_model", ""),
            source_adapter=data.get("source_adapter", ""),
            target_model=data.get("target_model", ""),
        )

        report.metadata_gate = MetadataGateResult(**data.get("metadata_gate", {}))
        report.source_quality_gate = data.get("source_quality_gate", {})
        report.delta_health_gate = DeltaHealthResult(**data.get("delta_health_gate", {}))
        report.compatibility_gate = CompatibilityResult(**data.get("compatibility_gate", {}))
        report.feasibility_gate = FeasibilityResult(**data.get("feasibility_gate", {}))
        if "target_evaluation_gate" in data:
            report.target_evaluation_gate = TargetEvaluationResult(**data.get("target_evaluation_gate", {}))
        report.release_decision_gate = ReleaseDecisionGateResult(**data.get("release_decision_gate", {}))
        return report

# --- Legacy Analyzers ---

class DeltaHealthAnalyzer:
    @staticmethod
    def _extract_layer(name: str) -> Optional[int]:
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    @staticmethod
    def analyze(model: torch.nn.Module) -> DeltaHealthResult:
        layer_energy = {}
        total_frob_norm_sq = 0.0
        
        # Group by module to compute W = B @ A
        lora_pairs = {}
        for name, param in model.named_parameters():
            if "lora_" not in name: continue
            
            base_name = name.replace(".lora_A.weight", "").replace(".lora_B.weight", "").replace(".weight", "")
            if base_name not in lora_pairs: lora_pairs[base_name] = {}
            
            if "lora_A" in name: lora_pairs[base_name]["A"] = param.data.detach().float()
            if "lora_B" in name: lora_pairs[base_name]["B"] = param.data.detach().float()

        entropies = []
        outliers = []
        concentration_score = 0.0

        for base_name, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair: continue
            
            A, B = pair["A"], pair["B"]
            # W = B @ A
            W = B @ A
            energy = float(torch.sum(W * W).item())
            total_frob_norm_sq += energy
            
            layer = DeltaHealthAnalyzer._extract_layer(base_name)
            if layer is not None:
                layer_energy[layer] = layer_energy.get(layer, 0.0) + energy
            
            # Spectral Entropy
            if W.ndim >= 2:
                U, S, V = torch.svd(W)
                s_norm = S / (torch.sum(S) + 1e-10)
                entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10)).item()
                # Normalize by log(rank)
                max_entropy = math.log(len(S) + 1e-10)
                entropies.append(entropy / max_entropy)

        normalized_entropy = np.mean(entropies) if entropies else 1.0
        total_energy = sum(layer_energy.values()) + 1e-10
        
        for layer, energy in layer_energy.items():
            ratio = energy / total_energy
            if ratio > 0.5:
                outliers.append(f"Layer {layer} ({ratio*100:.1f}%)")
                concentration_score = max(concentration_score, ratio)

        if concentration_score > 0.8:
            verdict = "CRITICALLY_CONCENTRATED"
        elif normalized_entropy < 0.4:
            verdict = "LOW_SPECTRAL_ENTROPY"
        else:
            verdict = "HEALTHY_DELTA"

        return DeltaHealthResult(
            verdict=verdict,
            global_frobenius_norm=math.sqrt(total_frob_norm_sq),
            concentration_score=concentration_score,
            outliers=outliers,
            normalized_spectral_entropy=normalized_entropy,
        )

# --- v2.11 Diagnostic Engine (Manifold/Norm-based Safety Mapping) ---

class DiagnosticEngine:
    """
    Diagnostic Engine for Neural-Scalpel v2.11.
    Calculates weights-based risk metrics to recommend optimal alpha values.
    """
    def __init__(self, source_adapter_path: str, target_model_path: str):
        self.source_adapter_path = source_adapter_path
        self.target_model_path = target_model_path
        self.metrics = {}

    def calculate_spectral_entropy(self, tensor: torch.Tensor) -> float:
        """Calculates the entropy of the singular value distribution."""
        if tensor.ndim < 2: return 0.0
        # Use float32 for SVD stability
        U, S, V = torch.svd(tensor.to(torch.float32))
        s_norm = S / (torch.sum(S) + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10)).item()
        return entropy

    def calculate_effective_rank(self, tensor: torch.Tensor) -> float:
        """Calculates the effective rank (exp of spectral entropy)."""
        entropy = self.calculate_spectral_entropy(tensor)
        return np.exp(entropy)

    def analyze_projected_module(self, module_name: str, lora_a: torch.Tensor, lora_b: torch.Tensor, target_weight: torch.Tensor, alpha: float, rank: int):
        """Analyze a projected module's interference risk."""
        # Calculate Effective Delta (W = B @ A * (alpha/rank))
        scaling = alpha / rank
        delta_w = (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scaling
        
        target_norm = torch.norm(target_weight.to(torch.float32)).item()
        delta_norm = torch.norm(delta_w).item()
        
        # Risk Metrics
        norm_ratio = delta_norm / (target_norm + 1e-10)
        entropy = self.calculate_spectral_entropy(delta_w)
        eff_rank = np.exp(entropy)
        
        return {
            "module": module_name,
            "delta_norm": round(delta_norm, 6),
            "target_norm": round(target_norm, 6),
            "norm_ratio": norm_ratio,
            "spectral_entropy": round(entropy, 4),
            "effective_rank": round(eff_rank, 2)
        }

    def aggregate_layer_stats(self, layer_metrics: list):
        """Aggregates metrics across all layers for a module."""
        if not layer_metrics: return {}
        ratios = [m["norm_ratio"] for m in layer_metrics]
        ranks = [m["effective_rank"] for m in layer_metrics]
        
        return {
            "mean_norm_ratio": round(np.mean(ratios), 6),
            "max_norm_ratio": round(np.max(ratios), 6),
            "p90_norm_ratio": round(np.percentile(ratios, 90), 6),
            "std_norm_ratio": round(np.std(ratios), 6),
            "mean_effective_rank": round(np.mean(ranks), 2),
            "max_effective_rank": round(np.max(ranks), 2)
        }

    def run_diagnostics(self, module_map: dict):
        """Placeholder for future automated end-to-end diagnostics."""
        pass

class AlphaRecommender:
    """
    Automated Alpha Selection for v2.11.
    Uses 'Safety Budgets' calibrated from empirical data.
    """
    def __init__(self, target_model_size: str = "0.5B"):
        # Calibrated Budgets based on v2.11 analysis
        # Budget = Max allowed (Projected Delta Norm / Target Weight Norm)
        if "0.5B" in target_model_size:
            self.budgets = {
                "attention": 0.002,    # 0.2% ratio is safe for Attn
                "mlp": 0.00002,        # 0.002% ratio for MLP (Extremely sensitive)
                "default": 0.001
            }
        else:
            # Safer defaults for unknown sizes
            self.budgets = {
                "attention": 0.001,
                "mlp": 0.00001,
                "default": 0.0005
            }

    def recommend_alpha(self, module_name: str, raw_delta_norm_ratio: float) -> float:
        """
        Calculates recommended alpha for a module.
        raw_delta_norm_ratio: The ratio calculated at alpha=1.0.
        """
        if any(x in module_name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            budget = self.budgets["attention"]
        elif any(x in module_name for x in ["gate_proj", "up_proj"]):
            budget = self.budgets["mlp"]
        elif "down_proj" in module_name:
            return 0.0 # Strict Policy: Keep down_proj at 0
        else:
            budget = self.budgets["default"]

        # Rec Alpha = Budget / Raw Ratio
        rec_alpha = budget / (raw_delta_norm_ratio + 1e-10)
        
        # Clamp to reasonable ranges
        if budget == self.budgets["attention"]:
            return min(max(rec_alpha, 0.125), 16.0) # Attn range
        else:
            return min(max(rec_alpha, 0.01), 1.0)   # MLP range (Stricter)

def generate_recommendation_report(metrics_summary: dict, current_alpha_map: dict, target_model_size: str = "0.5B"):
    """
    Generates a module-alpha-map based on summary metrics and current scaling.
    current_alpha_map: { 'module_name': actual_alpha_used_in_experiment }
    """
    recommender = AlphaRecommender(target_model_size)
    recommended_map = {}
    
    for m_type, stats in metrics_summary.items():
        if m_type == "_metadata": continue
        
        p90_ratio = stats["p90_norm_ratio"]
        current_alpha = current_alpha_map.get(m_type, 16.0)
        
        # New_Alpha = Current_Alpha * (Target_Budget / Current_Ratio)
        budget = recommender.budgets.get("mlp" if "gate" in m_type or "up" in m_type else "attention")
        if "down_proj" in m_type:
            recommended_map[m_type] = 0.0
            continue

        rec_alpha = current_alpha * (budget / (p90_ratio + 1e-10))
        
        # Clamp to reasonable ranges
        if budget == recommender.budgets["attention"]:
            rec_alpha = min(max(rec_alpha, 0.125), 16.0)
        else:
            rec_alpha = min(max(rec_alpha, 0.01), 1.0)
            
        recommended_map[m_type] = round(rec_alpha, 4)

    # Format for CLI
    alpha_map_str = ",".join([f"{k}={v}" for k, v in recommended_map.items()])
    if "down_proj" not in recommended_map:
        alpha_map_str += ",down_proj=0.0"

    return recommended_map, alpha_map_str

if __name__ == "__main__":
    # Example usage for manual check
    print("Neural-Scalpel v2.11 Diagnostic Engine Initialized (Backward Compatible Layer Enabled).")
