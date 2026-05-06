import unittest
import torch
import torch.nn as nn
from types import SimpleNamespace
from scripts.prepare_actual_lora_payload import (
    infer_qwen_target_shape,
    build_interpolated_layer_mapping,
    resize_and_analyze,
    fuse_qwen_layers
)

class TestStructuralProjection(unittest.TestCase):
    def test_infer_qwen_target_shape(self):
        # Mock config for Qwen2.5-0.5B
        config = SimpleNamespace(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            intermediate_size=4864
        )
        
        # Q/O projs
        self.assertEqual(infer_qwen_target_shape("self_attn.q_proj", config), [896, 896])
        self.assertEqual(infer_qwen_target_shape("self_attn.o_proj", config), [896, 896])
        
        # K/V projs (GQA: 2 heads * 64 head_dim = 128)
        self.assertEqual(infer_qwen_target_shape("self_attn.k_proj", config), [128, 896])
        self.assertEqual(infer_qwen_target_shape("self_attn.v_proj", config), [128, 896])
        
        # MLP projs
        self.assertEqual(infer_qwen_target_shape("mlp.gate_proj", config), [4864, 896])
        self.assertEqual(infer_qwen_target_shape("mlp.down_proj", config), [896, 4864])

    def test_interpolated_layer_mapping(self):
        src_layers = list(range(28)) # 0-27
        t_layers = 24
        
        mapping = build_interpolated_layer_mapping(src_layers, t_layers)
        
        # Endpoints
        self.assertEqual(mapping["0"]["lower"], 0)
        self.assertEqual(mapping["0"]["upper"], 0)
        self.assertEqual(mapping[str(t_layers-1)]["lower"], 27)
        self.assertEqual(mapping[str(t_layers-1)]["upper"], 27)
        
        # Intermediate check
        for info in mapping.values():
            self.assertTrue(0 <= info["alpha"] <= 1.0)
            self.assertTrue(0 <= info["lower"] <= 27)
            self.assertTrue(0 <= info["upper"] <= 27)

    def test_resize_and_analyze_basic(self):
        # Toy delta [128, 128]
        delta = torch.randn(128, 128)
        target_shape = [64, 64]
        
        re_t, stats, svd = resize_and_analyze(delta, target_shape, rank=8)
        
        self.assertEqual(list(re_t.shape), target_shape)
        self.assertEqual(re_t.dtype, torch.float16)
        self.assertTrue(0 <= stats["energy_retention"] <= 1.0)
        self.assertIsNotNone(svd)

    def test_svd_rank_effect(self):
        delta = torch.randn(64, 64)
        target_shape = [64, 64]
        
        _, stats_r2, _ = resize_and_analyze(delta, target_shape, rank=2)
        _, stats_r16, _ = resize_and_analyze(delta, target_shape, rank=16)
        
        # More rank should keep more energy
        self.assertGreaterEqual(stats_r16["energy_retention"], stats_r2["energy_retention"])

    def test_fuse_qwen_layers_shapes(self):
        # Mock projected delta
        projected = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(896, 896),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 896),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 896),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(4864, 896),
            "model.layers.0.mlp.up_proj.weight": torch.randn(4864, 896),
            "model.layers.0.mlp.down_proj.weight": torch.randn(896, 4864)
        }
        
        fused = fuse_qwen_layers(projected)
        
        # Check vLLM keys
        self.assertIn("model.layers.0.self_attn.qkv_proj.weight", fused)
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight", fused)
        self.assertIn("model.layers.0.mlp.down_proj.weight", fused)
        
        # Check shapes
        # QKV: 896 + 128 + 128 = 1152
        self.assertEqual(list(fused["model.layers.0.self_attn.qkv_proj.weight"].shape), [1152, 896])
        # GateUp: 4864 + 4864 = 9728
        self.assertEqual(list(fused["model.layers.0.mlp.gate_up_proj.weight"].shape), [9728, 896])

if __name__ == "__main__":
    unittest.main()
