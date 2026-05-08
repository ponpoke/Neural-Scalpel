import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
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
    ValidationReport,
    build_peft_lora_key,
    estimate_layer_correspondence
)

class MockModel(nn.Module):
    def __init__(self, in_dim=8, out_dim=8):
        super().__init__()
        self.config = nn.Module()
        self.config._name_or_path = "mock-model"
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x=None, **kwargs):
        out = MagicMock()
        out.logits = torch.randn(1, 1, 100) 
        return out

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        # Return a custom dict that has a .to method
        class TokenizerOutput(dict):
            def to(self, device):
                return self
        out = TokenizerOutput({"input_ids": torch.zeros(1, 1).long()})
        return out
    def apply_chat_template(self, messages, **kwargs):
        return "formatted_prompt"
    def decode(self, ids, **kwargs):
        return "decoded text"

# --- Task 1: API Hardening Tests ---

def test_align_rejects_empty_prompts():
    src = MockModel()
    tgt = MockModel()
    with pytest.raises(ValueError, match="calibration_prompts must not be empty"):
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

def test_numerical_stability_inf_nan_detection():
    # 1. NaN in solve_activation_adapter
    delta = TransportedDelta({"l1": torch.randn(1, 8)}, "tgt")
    mock_coll = MagicMock()
    mock_coll.get_stacked.return_value = {"l1": torch.tensor([[float('nan')]])}
    
    with patch("neural_scalpel.core.ops.collection_context", return_value=MagicMock(__enter__=MagicMock(return_value=mock_coll))):
        with pytest.raises(ValueError, match="Non-finite values detected"):
            solve_activation_adapter(MockModel(), delta, ["p"], MockTokenizer(), ["l1"])

    # 2. Inf in transport_delta
    mapping = AlignmentMap(layer_maps={"l1": torch.eye(8)}, source_model_id="src", target_model_id="tgt")
    bad_delta = BehavioralDelta({"l1": torch.tensor([[float('inf')]])}, "src")
    with pytest.raises(ValueError, match="Non-finite delta detected"):
        transport_delta(bad_delta, mapping)

    # 3. Non-finite in validate_behavior (logits)
    bad_model = MockModel()
    bad_model.disable_adapter = MagicMock()
    bad_model.forward = MagicMock(return_value=MagicMock(logits=torch.tensor([[[float('nan')]]])))
    with patch("peft.PeftModel.from_pretrained", return_value=bad_model):
        report = validate_behavior(MockModel(), "path", ["p"], MockTokenizer(), require_nonzero_adapter=False)
        print(f"DEBUG: report.summary={report.summary}")
        assert report.status == "FAIL"
        assert "NUMERICALLY_UNSTABLE" in report.summary

# --- Task 2: module_to_delta_layer Mapping ---

def test_solve_adapter_with_module_to_delta_layer_mapping():
    X = torch.randn(1, 8)
    mock_coll = MagicMock()
    mock_coll.get_stacked.return_value = {"model.layers.10.proj": X}
    
    with patch("neural_scalpel.core.ops.collection_context", return_value=MagicMock(__enter__=MagicMock(return_value=mock_coll))):
        Y = torch.randn(1, 8)
        delta = TransportedDelta({"layer10": Y}, "tgt")
        model = MockModel()
        
        mapping = {"model.layers.10.proj": "layer10"}
        
        sol = solve_activation_adapter(
            model, delta, ["p"], MockTokenizer(), 
            target_modules=["model.layers.10.proj"],
            module_to_delta_layer=mapping
        )
        
        assert "model.layers.10.proj" in sol.module_weights
        assert sol.metadata["module_to_delta_layer"] == mapping

# --- Task 3: PEFT key styles ---

def test_build_peft_lora_key_styles():
    assert build_peft_lora_key(
        "model.layers.0.mlp.down_proj", "A",
        peft_key_prefix="base_model.model",
        key_style="peft_default",
    ) == "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"

    assert build_peft_lora_key(
        "model.layers.0.mlp.down_proj", "B",
        peft_key_prefix="base_model.model",
        adapter_name="default",
        key_style="peft_named",
    ) == "base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight"

    assert build_peft_lora_key(
        "model.layers.0.mlp.down_proj", "A",
        key_style="raw",
    ) == "model.layers.0.mlp.down_proj.lora_A.weight"

def test_export_lora_with_custom_prefix():
    W = torch.randn(8, 4)
    sol = ActivationAdapterSolution(
        module_weights={"model.layers.0.mlp.down_proj": W},
        target_model_id="tgt",
        reconstruction_errors={"model.layers.0.mlp.down_proj": 0.1},
    )

    result = export_lora(
        sol,
        rank=2,
        peft_key_prefix="custom.prefix",
        adapter_name="default",
        key_style="peft_named",
    )

    assert "custom.prefix.model.layers.0.mlp.down_proj.lora_A.default.weight" in result.lora_state_dict

# --- Task 4: prompt_formatter ---

def test_prompt_formatter_called_in_align():
    called = {"v": False}
    def formatter(tok, prompt):
        called["v"] = True
        return "formatted_" + prompt

    src = MockModel()
    tgt = MockModel()
    
    with patch("neural_scalpel.core.ops.collection_context", return_value=MagicMock()):
        with patch("neural_scalpel.core.ops.learn_alignment_map", return_value=None):
            align(src, tgt, ["p1"], MockTokenizer(), ["l1"], ["l1"], prompt_formatter=formatter)
            assert called["v"] is True

# --- Task 5: CKA ---

def test_gram_linear_cka_identity_high():
    X = torch.randn(20, 8)
    from neural_scalpel.core.alignment import gram_linear_cka
    score = gram_linear_cka(X, X)
    assert score > 0.99

def test_estimate_layer_correspondence_logic():
    X = torch.randn(10, 8)
    Y = X + torch.randn(10, 8) * 0.001 
    Z = torch.randn(10, 8) 
    
    ds = PairedActivationDataset(
        source_activations={"s1": X, "s2": Z},
        target_activations={"t1": Y}
    )
    
    corr = estimate_layer_correspondence(ds)
    assert corr.target_to_source["t1"] == "s1"
