import pytest
import torch
import torch.nn as nn
from neural_scalpel import (
    align,
    extract_behavior_delta,
    transport_delta,
    solve_activation_adapter,
    export_lora,
    validate_behavior,
    PairedActivationDataset,
    AlignmentMap,
    BehavioralDelta,
    TransportedDelta,
    ActivationAdapterSolution,
    ValidationReport
)

class MockModel(nn.Module):
    def __init__(self, in_dim=8, out_dim=8):
        super().__init__()
        self.config = nn.Module()
        self.config._name_or_path = "mock-model"
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x, **kwargs):
        return nn.Module() # Mock output

class MockTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.zeros(1, 1).long()}
    def apply_chat_template(self, messages, **kwargs):
        return "formatted_prompt"

# --- Task 1: API Hardening Tests ---

def test_align_rejects_empty_prompts():
    src = MockModel()
    tgt = MockModel()
    with pytest.raises(ValueError, match="prompts must not be empty"):
        align(src, tgt, [], MockTokenizer(), ["proj"], ["proj"])

def test_solve_activation_adapter_rejects_empty_modules():
    delta = TransportedDelta({"l1": torch.randn(1, 8)}, "tgt")
    with pytest.raises(ValueError, match="target_modules must not be empty"):
        solve_activation_adapter(MockModel(), delta, ["p"], MockTokenizer(), [])

def test_export_lora_rejects_empty_solution():
    sol = ActivationAdapterSolution({}, "tgt", {})
    with pytest.raises(RuntimeError, match="No modules were successfully solved"):
        export_lora(sol)

def test_validation_report_gate_schema():
    report = ValidationReport(phase="G8", status="PENDING")
    report.add_gate("G1", True, "Success", severity="info", metrics={"val": 1.0})
    
    gate = report.gates["G1"]
    assert gate["success"] is True
    assert gate["severity"] == "info"
    assert gate["metrics"]["val"] == 1.0

# --- Task 2: module_to_delta_layer Mapping ---

def test_solve_adapter_with_module_to_delta_layer_mapping():
    # Desired delta has key 'layer10'
    # Module is 'model.layers.10.proj'
    delta = TransportedDelta({"layer10": torch.randn(1, 8)}, "tgt")
    model = MockModel()
    
    # This should succeed by mapping 'model.layers.10.proj' -> 'layer10'
    # We need to mock the collector but let's assume it works if we use the same key in inputs_stacked
    # Actually, we need to implement it in ops.py first.
    pass

# --- Task 3: PEFT key prefix ---

def test_export_lora_with_custom_prefix():
    # Implementation will check if prefix is applied
    pass

# --- Task 4: prompt_formatter ---

def test_prompt_formatter_called_in_align():
    pass
