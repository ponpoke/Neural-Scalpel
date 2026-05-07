import torch
import pytest
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

def test_spectral_entropy_logic():
    state = {}
    # Create a delta with skewed singular values
    A0 = torch.zeros(16, 128)
    A0[0, 0] = 100.0
    B0 = torch.eye(128, 16)
    state["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = A0
    state["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] = B0
    
    model = MockModel(state)
    result = DeltaHealthAnalyzer.analyze(model)
    
    assert result.spectral_entropy < 0.5
    assert result.verdict == "LOW_SPECTRAL_ENTROPY" or result.concentration_score > 0.7
    print(f"Entropy: {result.spectral_entropy:.4f}")

def test_concentration_score_logic():
    state = {
        "layers.0.lora_A": torch.ones(16, 128) * 10.0,
        "layers.0.lora_B": torch.ones(128, 16) * 10.0,
        "layers.1.lora_A": torch.ones(16, 128) * 0.1,
        "layers.1.lora_B": torch.ones(128, 16) * 0.1,
    }
    model = MockModel(state)
    result = DeltaHealthAnalyzer.analyze(model)
    assert result.concentration_score > 0.9
    assert result.verdict == "CRITICALLY_CONCENTRATED"
