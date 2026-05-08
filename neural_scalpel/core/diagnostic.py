import torch
import numpy as np
from safetensors.torch import load_file
import os
import json

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
        s_norm = S / torch.sum(S)
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
    print("Neural-Scalpel v2.11 Diagnostic Engine Initialized.")
