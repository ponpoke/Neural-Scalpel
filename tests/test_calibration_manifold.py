import torch
import unittest
import os
from neural_scalpel.io.calibration import search_optimal_awq_scales, SyntheticManifoldGenerator, ManifoldProfiler
from neural_scalpel.io.awq_bridge import AWQBridge

class TestCalibrationManifold(unittest.TestCase):
    def test_lmr_error_reduction(self):
        # 1. Create a "surgical weight" (FP16) and a skewed activation distribution
        # Weights are uniform, but activations have a few "outlier" dimensions
        in_features, out_features = 128, 128
        weight = torch.randn(out_features, in_features)
        
        # Outlier activations (first 10 dims are 10x larger)
        activations = torch.randn(10, in_features)
        activations[:, :10] *= 10.0
        
        # 2. Naive Quantization (8-bit simulation)
        # Without scaling, outliers in X will amplify quantization errors in W
        abs_max = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
        scales = abs_max / 127.0
        q_w_naive = torch.round(weight / (scales + 1e-12)).clamp(-128, 127) * scales
        
        naive_error = torch.mean(torch.abs(weight - q_w_naive) * torch.abs(activations).mean(dim=0))
        
        # 3. LMR (Lightweight Manifold Re-calibration)
        opt_scales = search_optimal_awq_scales(weight, activations)
        
        scaled_w = weight * opt_scales
        abs_max_opt = torch.max(torch.abs(scaled_w), dim=1, keepdim=True)[0]
        scales_opt = abs_max_opt / 127.0
        q_w_opt = (torch.round(scaled_w / (scales_opt + 1e-12)).clamp(-128, 127) * scales_opt) / opt_scales
        
        lmr_error = torch.mean(torch.abs(weight - q_w_opt) * torch.abs(activations).mean(dim=0))
        
        print(f"\n[LMR Test] Naive Quant Error: {naive_error:.6f}")
        print(f"[LMR Test] LMR Quant Error:   {lmr_error:.6f}")
        
        # LMR should significantly reduce the weighted error by protecting active channels
        self.assertLess(lmr_error, naive_error)

    def test_awq_bridge_cli_integration(self):
        # Verify that AWQBridge uses the calibration data correctly
        activations = torch.randn(1, 128)
        bridge = AWQBridge(calibration_data=activations)
        
        state_dict = {"model.layer.weight": torch.randn(64, 128)}
        output_path = "test_calibrated.awq.safetensors"
        
        try:
            bridge.save_weights(output_path, state_dict)
            self.assertTrue(os.path.exists(output_path))
            
            # Load back and check for presence of scales
            from safetensors.torch import load_file
            loaded = load_file(output_path)
            self.assertIn("model.layer.weight_awq_scales", loaded)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
