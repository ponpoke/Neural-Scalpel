import torch
from neural_scalpel.core.adapters import BaseAdapter
from neural_scalpel.core.diagnostic import DeltaHealthResult

def test_base_adapter_adaptive_scaling():
    health = DeltaHealthResult(
        verdict="MODERATELY_CONCENTRATED",
        concentration_score=0.5,
        outliers=["Layer 5"],
        effective_rank=1.2
    )
    
    adapter = BaseAdapter((4096, 32), (3584, 28), delta_health=health)
    
    # Expected scale for outlier Layer 5: 1.0 * 0.9 (MOD) * 0.8 (ER<2) * 0.7 (Outlier) = 0.504
    scale_5 = adapter.get_adaptive_scale("layers.5.self_attn.q_proj")
    assert abs(scale_5 - 0.504) < 1e-5
    
    # Expected scale for normal Layer 0: 1.0 * 0.9 * 0.8 = 0.72
    scale_0 = adapter.get_adaptive_scale("layers.0.self_attn.q_proj")
    assert abs(scale_0 - 0.72) < 1e-5
    
    print(f"Adaptive Scale Layer 5: {scale_5:.3f}")
    print(f"Adaptive Scale Layer 0: {scale_0:.3f}")
