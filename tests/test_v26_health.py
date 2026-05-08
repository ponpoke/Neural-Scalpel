import torch
import pytest
import math
from neural_scalpel.core.diagnostic import DeltaHealthAnalyzer

class MockParam:
    def __init__(self, data):
        self.data = data

class MockModel:
    def __init__(self, state_dict):
        self.state = state_dict
    def named_parameters(self):
        for k, v in self.state.items():
            yield k, MockParam(v)

def test_spectral_entropy_normalization():
    state = {}
    # Distribute weights over 4 layers to avoid CRITICALLY_CONCENTRATED
    # but keep singular values skewed to trigger LOW_SPECTRAL_ENTROPY
    for i in range(4):
        A = torch.zeros(16, 128)
        A[0, 0] = 10.0 # Only 1st component is strong
        B = torch.eye(128, 16)
        state[f"model.layers.{i}.self_attn.q_proj.lora_A.weight"] = A
        state[f"model.layers.{i}.self_attn.q_proj.lora_B.weight"] = B
    
    model = MockModel(state)
    result = DeltaHealthAnalyzer.analyze(model)
    
    # Normalized entropy should be low (only 4 non-zero singular values total, and they are identical)
    # Actually with 4 identical non-zero S values, raw entropy is log(4) = 1.38
    # log(num_components) = log(16 * 4) = log(64) = 4.15
    # Normalized = 1.38 / 4.15 = 0.33 (< 0.4 threshold)
    assert result.normalized_spectral_entropy < 0.4
    assert result.verdict == "LOW_SPECTRAL_ENTROPY"
    print(f"Norm Entropy: {result.normalized_spectral_entropy:.4f}")

def test_regex_layer_extraction():
    state = {
        "base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight": torch.ones(16, 128),
        "base_model.model.model.layers.10.self_attn.q_proj.lora_B.weight": torch.ones(128, 16),
    }
    model = MockModel(state)
    result = DeltaHealthAnalyzer.analyze(model)
    
    # Check if Layer 10 was correctly identified despite long prefix
    assert any("Layer 10" in o for o in result.outliers)
    assert result.verdict == "CRITICALLY_CONCENTRATED"
