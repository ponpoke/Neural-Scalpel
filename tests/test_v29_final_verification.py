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
        self.adaptive_scaling_config = None
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_v28_order_agnostic_pair_logic():
    source_info = {"hidden_size": 4096, "intermediate_size": 11008}
    target_info = {"hidden_size": 4096, "intermediate_size": 11008}
    adapter = get_adapter("llama", "qwen", source_info, target_info, projection_mode="piecewise")
    A = torch.randn(16, 4096); B = torch.randn(11008, 16)
    key_a = "model.layers.0.mlp.up_proj.lora_A.weight"; key_b = "model.layers.0.mlp.up_proj.lora_B.weight"
    assert adapter.project_tensor(key_b, B) is None
    assert isinstance(adapter.project_tensor(key_a, A), dict)

def test_v28_buffer_finalize_warning():
    adapter = get_adapter("llama", "qwen", (4096, 32), (4096, 32), projection_mode="piecewise")
    adapter.project_tensor("layers.0.mlp.up_proj.lora_A.weight", torch.randn(16, 4096))
    with pytest.warns(RuntimeWarning, match="Unprocessed LoRA pairs"):
        adapter.finalize()

def test_v29_experimental_warnings():
    adapter = get_adapter("llama", "qwen", (4096, 32), (4096, 32), projection_mode="kernel")
    with pytest.warns(RuntimeWarning, match=r"\[EXPERIMENTAL\]"):
        adapter.project_tensor("dummy", torch.randn(16, 4096))

def test_v26_normalized_metrics():
    A = torch.zeros(16, 16); A[0, 0] = 10.0; B = torch.eye(16, 16)
    model = MockModel({"layers.0.self_attn.q_proj.lora_A.weight": A, "layers.0.self_attn.q_proj.lora_B.weight": B})
    result = DeltaHealthAnalyzer.analyze(model)
    assert result.normalized_spectral_entropy < 0.1

def test_safe_project_mode_upgrade(tmp_path):
    from unittest.mock import patch
    report = AdapterTransferDiagnosticReport()
    report.source_quality_gate = {"verdict": "POSITIVE_TEACHER", "gate_status": "PASS"}
    report.delta_health_gate.verdict = "LOW_SPECTRAL_ENTROPY"
    # Use correct string constants from diagnostic.py
    report.compatibility_gate.verdict = "PASS"
    report.feasibility_gate.verdict = "FEASIBLE"
    report.target_evaluation_gate.verdict = "PENDING"
    report.finalize_release_decision()
    
    assert report.release_decision_gate.verdict == "PROJECTION_CANDIDATE"
    
    args = MockArgs(output_dir=str(tmp_path), projection_mode="linear")
    with patch("neural_scalpel.commands.safe_project.DiagnosticRunner.execute", return_value=report):
        with patch("neural_scalpel.commands.safe_project.run_project") as mock_proj:
            with patch("neural_scalpel.commands.safe_project.run_evaluate"):
                run_safe_project(args)
                call_args = mock_proj.call_args[0][0]
                assert call_args.projection_mode == "piecewise"
