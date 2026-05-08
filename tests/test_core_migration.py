import pytest
import torch
import torch.nn as nn
from neural_scalpel import (
    PairedActivationDataset,
    AlignmentMap,
    BehavioralDelta,
    TransportedDelta,
    learn_alignment_map
)
from neural_scalpel.core.math import solve_ridge, low_rank_decompose_for_peft

class MockModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        return self.proj(x)

def test_math_ridge_solver():
    # Simple X @ W = Y
    X = torch.randn(10, 8)
    W_true = torch.randn(8, 4)
    Y = X @ W_true
    
    W_solved = solve_ridge(X, Y, alpha=1e-5)
    assert W_solved.shape == (8, 4)
    assert torch.allclose(X @ W_solved, Y, atol=1e-3)

def test_math_low_rank_decomposition():
    W = torch.randn(64, 64)
    rank = 16
    A, B = low_rank_decompose_for_peft(W, rank)
    
    # A: (rank, d_in), B: (d_out, rank)
    assert A.shape == (rank, 64)
    assert B.shape == (64, rank)
    
    W_rec = A.t() @ B.t()
    assert W_rec.shape == (64, 64)

def test_alignment_map_projection():
    # Setup mapping
    P = torch.eye(8)
    mapping = AlignmentMap(
        layer_maps={"layer1": P},
        source_model_id="src",
        target_model_id="tgt"
    )
    
    delta_s = torch.ones(1, 8)
    delta_t = mapping.project("layer1", delta_s)
    
    assert torch.allclose(delta_s, delta_t)

def test_paired_activation_dataset_validation():
    src_acts = {"l1": torch.randn(5, 8)}
    tgt_acts = {"l1": torch.randn(5, 8)}
    
    # This should pass
    ds = PairedActivationDataset(src_acts, tgt_acts)
    assert ds.source_activations["l1"].shape[0] == 5
    
    # Mismatch should fail
    tgt_acts_fail = {"l1": torch.randn(6, 8)}
    with pytest.raises(ValueError, match="Sample count mismatch"):
        PairedActivationDataset(src_acts, tgt_acts_fail)

def test_learn_alignment_map():
    # Mock data where Target = Source * 2
    X = torch.randn(100, 8)
    Y = X * 2.0
    
    ds = PairedActivationDataset({"l1": X}, {"l1": Y})
    mapping = learn_alignment_map(ds, alpha=0.0)
    
    test_vec = torch.ones(1, 8)
    projected = mapping.project("l1", test_vec)
    
    # Should be approx 2.0
    assert torch.allclose(projected, test_vec * 2.0, atol=1e-2)
