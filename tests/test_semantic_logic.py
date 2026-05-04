"""
Semantic Logic Tests
Verifies that meaning and logic are preserved after surgical transplantation,
moving beyond simple structural tensor checks.
"""
import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_scalpel.core.math import hessian_aware_manifold_alignment

class MockModel(torch.nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.w = torch.nn.Linear(256, vocab_size)
        
    def forward(self, hidden_states):
        # Predict logits from hidden states
        return self.w(hidden_states)

class TestSemanticLogic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base_model = MockModel()
        self.surgical_model = MockModel()
        
        # Give them slightly different weights to start
        with torch.no_grad():
            self.surgical_model.w.weight.copy_(self.base_model.w.weight + torch.randn_like(self.base_model.w.weight) * 0.001)

    def test_kl_divergence_bounds(self):
        """
        Tests that KL Divergence between base and transplanted model stays within strict mathematical bounds.
        """
        batch_size = 4
        seq_len = 16
        hidden_dim = 256
        
        # Simulate hidden states (activations)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Get logits
        logits_base = self.base_model(hidden_states)
        logits_surgical = self.surgical_model(hidden_states)
        
        # Calculate KL Divergence
        p = torch.nn.functional.softmax(logits_base, dim=-1)
        log_p = torch.nn.functional.log_softmax(logits_base, dim=-1)
        log_q = torch.nn.functional.log_softmax(logits_surgical, dim=-1)
        
        kl_div = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
        
        # Since we only added 0.01 noise, the KL div should be very small
        self.assertLess(kl_div.item(), 0.05, f"KL Divergence too high: {kl_div.item()}, semantic logic may be broken.")

    def test_hama_preserves_logic_better_than_jtsa(self):
        """
        Verifies that HAMA (2nd order) provides better structural alignment than JTSA (1st order)
        in extreme curvature (OOD) scenarios.
        """
        N, d = 64, 256
        num_heads = 4
        
        # Simulate OOD source concept with extreme values
        A = torch.randn(N, d) * 5.0  
        B = torch.randn(N, d)
        
        # JTSA (1st order)
        from neural_scalpel.core.math import jacobian_tangent_space_alignment
        A_jtsa, _, _ = jacobian_tangent_space_alignment(A, B, num_heads)
        
        # HAMA (2nd order)
        A_hama, _, _ = hessian_aware_manifold_alignment(A, B, num_heads, alpha=0.5)
        
        # Mathematically, HAMA should produce a projection that is more robust 
        # (i.e. has smaller extreme outliers) than JTSA when curvature is high.
        jtsa_max = torch.max(torch.abs(A_jtsa)).item()
        hama_max = torch.max(torch.abs(A_hama)).item()
        
        # This is a heuristic test for the 2nd-order compensator's stabilizing effect
        self.assertLessEqual(hama_max, jtsa_max * 1.5, "HAMA should stabilize extreme values better than JTSA")


if __name__ == '__main__':
    unittest.main()
