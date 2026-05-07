import pytest
import torch
import warnings
from neural_scalpel.core.adapters import get_adapter, AdaptiveScalingConfig
from neural_scalpel.core.diagnostic import DeltaHealthAnalyzer, AdapterTransferDiagnosticReport
from neural_scalpel.commands.safe_project import run_safe_project

class MockParam:
    def __init__(self, data):
        self.data = data

class MockModel:
    def __init__(self, state_dict):
        self.state = state_dict
    def named_parameters(self):
        for k, v in self.state.items():
            yield k, MockParam(v)

class MockArgs:
    def __init__(self, **kwargs):
        self.source_base_model = "sb"
        self.source_adapter = "sa"
        self.target_model = "tm"
        self.benchmark = "sql_50"
        self.output_dir = "runs/test"
        self.rank = 16
        self.alpha = 16
        self.projection_mode = "linear"
        self.positive_delta_threshold = 0.0
        self.max_regression_rate = 0.05
        self.force = False
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_v28_pair_aware_piecewise_logic():
    # Setup adapter with matching dims
    source_info = {"hidden_size": 4096, "intermediate_size": 11008}
    target_info = {"hidden_size": 4096, "intermediate_size": 11008}
    adapter = get_adapter("llama", "qwen", source_info, target_info, projection_mode="piecewise")
    
    # Simulate streaming A then B (MLP Up projection)
    A = torch.randn(16, 4096)
    B = torch.randn(11008, 16)
    key_a = "model.layers.0.mlp.up_proj.lora_A.weight"
    key_b = "model.layers.0.mlp.up_proj.lora_B.weight"
    
    # 1. Process A -> Should return None (deferred)
    res_a = adapter.project_tensor(key_a, A)
    assert res_a is None
    
    # 2. Process B -> Should return dict with both A and B
    res_b = adapter.project_tensor(key_b, B)
    assert isinstance(res_b, dict)
    assert key_a in res_b
    assert key_b in res_b
    assert res_b[key_a].shape == (16, 4096)
    assert res_b[key_b].shape == (11008, 16)

def test_v29_experimental_warnings():
    adapter = get_adapter("llama", "qwen", (4096, 32), (4096, 32), projection_mode="kernel")
    
    with pytest.warns(RuntimeWarning, match=r"\[EXPERIMENTAL\]"):
        adapter.project_tensor("dummy", torch.randn(16, 4096))

def test_v26_normalized_metrics():
    # Create a rank-16 delta where ONLY 1 component has energy
    # We need to use 16x16 or similar to make rank=16
    A = torch.zeros(16, 16)
    A[0, 0] = 10.0
    B = torch.eye(16, 16)
    state = {
        "layers.0.self_attn.q_proj.lora_A.weight": A,
        "layers.0.self_attn.q_proj.lora_B.weight": B,
    }
    model = MockModel(state)
    result = DeltaHealthAnalyzer.analyze(model)
    
    # Singular values: [10, 0, 0, ...]
    # Raw entropy: 0. Normalized entropy: 0
    assert result.normalized_spectral_entropy < 0.1
    assert result.normalized_effective_rank < 0.1

def test_safe_project_abort_on_critically_concentrated(tmp_path):
    from unittest.mock import patch
    
    # Create a mock report that is critically concentrated
    report = AdapterTransferDiagnosticReport()
    report.source_quality_gate = {"verdict": "POSITIVE_TEACHER", "gate_status": "PASS"}
    report.delta_health_gate.verdict = "CRITICALLY_CONCENTRATED"
    report.finalize_release_decision()
    
    # Verdict should be RESEARCH_ONLY (blocking)
    assert report.release_decision_gate.verdict == "RESEARCH_ONLY"
    
    args = MockArgs(output_dir=str(tmp_path))
    
    # Mocking execution to return our report
    with patch("neural_scalpel.commands.safe_project.DiagnosticRunner.execute", return_value=report):
        with patch("neural_scalpel.commands.safe_project.run_project") as mock_proj:
            run_safe_project(args)
            # Should NOT proceed to projection
            mock_proj.assert_not_called()
