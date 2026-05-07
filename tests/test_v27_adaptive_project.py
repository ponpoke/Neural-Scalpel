import torch
from neural_scalpel.core.adapters import BaseAdapter, AdaptiveScalingConfig
from neural_scalpel.core.diagnostic import DeltaHealthResult

def test_base_adapter_traceable_scaling():
    health = DeltaHealthResult(
        verdict="MODERATELY_CONCENTRATED",
        concentration_score=0.5,
        outliers=["Layer 5 (50.0%)"],
        normalized_spectral_entropy=0.3 # Trigger low entropy scale
    )
    
    # Custom config
    config = AdaptiveScalingConfig(
        moderately_concentrated_scale=0.9,
        low_normalized_entropy_scale=0.8,
        outlier_layer_scale=0.5 # More aggressive
    )
    
    adapter = BaseAdapter((4096, 32), (3584, 28), delta_health=health, scaling_config=config)
    
    # Layer 5 (Outlier): 1.0 * 0.9 (MOD) * 0.8 (Ent) * 0.5 (Outlier) = 0.36
    key_5 = "model.layers.5.self_attn.q_proj"
    scale_5 = adapter.get_adaptive_scale(key_5)
    assert abs(scale_5 - 0.36) < 1e-5
    
    # Layer 0 (Normal): 1.0 * 0.9 * 0.8 = 0.72
    key_0 = "model.layers.0.self_attn.q_proj"
    scale_0 = adapter.get_adaptive_scale(key_0)
    assert abs(scale_0 - 0.72) < 1e-5
    
    # Check traceability
    assert health.applied_scales[key_5] == scale_5
    assert health.applied_scales[key_0] == scale_0
    
    print(f"Traceable Scale Layer 5: {scale_5:.3f}")
    print(f"Traceable Scale Layer 0: {scale_0:.3f}")
