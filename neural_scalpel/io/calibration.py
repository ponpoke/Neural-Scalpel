import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ManifoldProfiler:
    """
    Utility to collect activation statistics (the manifold) from a small dataset.
    Essential for preserving LLM emergent outliers during data-dependent operations.
    """
    def __init__(self, samples: Optional[torch.Tensor] = None):
        self.samples = samples # (N_samples, hidden_dim)

    @torch.no_grad()
    def get_activation_magnitudes(self) -> torch.Tensor:
        """Returns the mean absolute magnitude per input feature."""
        if self.samples is None:
            return torch.ones(1)
        # Mean absolute value along the sample dimension
        return torch.abs(self.samples).mean(dim=0)

class SyntheticManifoldGenerator:
    """
    Zero-Dataset Fallback Implementation.
    Generates synthetic activation manifolds mathematically derived from weight
    variances/norms. 
    WARNING: Does not capture emergent massive outliers in LLMs. Use only as a last resort.
    """
    @staticmethod
    @torch.no_grad()
    def get_synthetic_activation_magnitudes(weight: torch.Tensor) -> torch.Tensor:
        """
        Calculates synthetic activation statistics mathematically from the weight variance.
        Assumes normally distributed inputs scaled by weight norms.
        """
        w_mag = torch.abs(weight).mean(dim=0)
        # Approximate input activation magnitude as inversely proportional to weight norms
        synthetic_mag = 1.0 / (w_mag + 1e-6)
        # Normalize
        synthetic_mag = synthetic_mag / synthetic_mag.mean()
        return synthetic_mag

def search_optimal_awq_scales(
    weight: torch.Tensor,
    activations: Optional[torch.Tensor] = None,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    max_expand: float = 2.0
) -> torch.Tensor:
    """
    Lightweight Manifold Re-calibration (LMR) for AWQ-style scaling.
    Searches for scaling factors that minimize quantization error.
    Uses real activations if provided, otherwise falls back to synthetic generation (Warning: Risk of quality loss).
    """
    device = weight.device
    if activations is not None:
        x_mag = torch.abs(activations).mean(dim=0)
    else:
        print("[WARNING] No calibration data provided. Falling back to Synthetic Manifold Generation. This may destroy outlier semantics in large models.")
        x_mag = SyntheticManifoldGenerator.get_synthetic_activation_magnitudes(weight).to(device)
    
    # w_mag: magnitude of weights (out_features, in_features)
    w_mag = torch.abs(weight).mean(dim=0)
    
    # Initial scale guess based on AWQ paper: s = x_mag^alpha
    # We search for the best alpha in [0, 1]
    best_error = float('inf')
    best_s = torch.ones_like(x_mag)
    
    # For a few grid points of alpha
    for alpha in torch.linspace(0, 1, n_grid):
        s = torch.pow(x_mag, alpha) / (torch.pow(w_mag, 1 - alpha) + 1e-12)
        s = s.clamp(min=max_shrink, max=max_expand).to(device)
        
        # Scale weights and simulate quantization (8-bit for fast search)
        scaled_w = weight * s
        
        # Simple block-wise quantization simulation
        # q_w = quant(scaled_w) / s
        abs_max = torch.max(torch.abs(scaled_w), dim=1, keepdim=True)[0]
        scales = abs_max / 127.0
        q_w = torch.round(scaled_w / (scales + 1e-12)).clamp(-128, 127) * scales
        dequant_w = q_w / s
        
        # Measure error on a small subset of activations
        # error = || X @ W - X @ dequant_w ||
        # Approximate error via MSE on weights weighted by activation magnitude
        error = torch.mean(torch.abs(weight - dequant_w) * x_mag)
        
        if error < best_error:
            best_error = error
            best_s = s
            
    return best_s
