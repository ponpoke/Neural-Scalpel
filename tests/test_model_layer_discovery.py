"""
Neural-Scalpel Model Layer Discovery Tests

Tests for automatic layer discovery and classification across
different model architectures (Qwen, Llama, OPT).
"""

import torch
import pytest
from neural_scalpel.serving.model_layer_discovery import (
    discover_layers,
    LayerType,
    ModelLayerMap,
)


def _make_mock_qwen_state_dict():
    """Creates a mock state dict mimicking Qwen2.5 naming."""
    sd = {}
    for i in range(4):
        prefix = f"model.layers.{i}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(64, 64)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(64, 64)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(64, 64)
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(64, 64)
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(128, 64)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(128, 64)
        sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(64, 128)
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(64)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(64)
    sd["model.embed_tokens.weight"] = torch.randn(1000, 64)
    sd["lm_head.weight"] = torch.randn(1000, 64)
    return sd


def _make_mock_opt_state_dict():
    """Creates a mock state dict mimicking OPT naming."""
    sd = {}
    for i in range(2):
        prefix = f"model.decoder.layers.{i}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(32, 32)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(32, 32)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(32, 32)
        sd[f"{prefix}.self_attn.out_proj.weight"] = torch.randn(32, 32)
        sd[f"{prefix}.fc1.weight"] = torch.randn(64, 32)
        sd[f"{prefix}.fc2.weight"] = torch.randn(32, 64)
    return sd


class TestLayerDiscovery:

    def test_qwen_architecture_detection(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        assert layer_map.model_type == "qwen2"
        assert layer_map.num_blocks == 4

    def test_qwen_swappable_layers(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        # 4 blocks × 7 projections = 28 swappable layers
        assert len(layer_map.swappable_layers) == 28

    def test_qwen_layer_classification(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        q_layers = layer_map.by_type(LayerType.ATTENTION_Q)
        assert len(q_layers) == 4
        gate_layers = layer_map.by_type(LayerType.MLP_GATE)
        assert len(gate_layers) == 4

    def test_opt_fc_layers(self):
        sd = _make_mock_opt_state_dict()
        layer_map = discover_layers(sd)
        fc1 = layer_map.by_type(LayerType.MLP_FC1)
        fc2 = layer_map.by_type(LayerType.MLP_FC2)
        assert len(fc1) == 2
        assert len(fc2) == 2

    def test_block_index_extraction(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        block_0 = layer_map.by_block(0)
        assert len(block_0) > 0
        for layer in block_0:
            assert layer.block_index == 0

    def test_serialization(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        d = layer_map.to_dict()
        assert "model_type" in d
        assert "layers" in d
        assert isinstance(d["layers"], list)

    def test_empty_dict(self):
        layer_map = discover_layers({})
        assert layer_map.num_blocks == 0
        assert len(layer_map.layers) == 0

    def test_embedding_detection(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        embeddings = layer_map.by_type(LayerType.EMBEDDING)
        assert len(embeddings) == 1

    def test_lm_head_detection(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        heads = layer_map.by_type(LayerType.LM_HEAD)
        assert len(heads) == 1

    def test_architecture_notes(self):
        sd = _make_mock_qwen_state_dict()
        layer_map = discover_layers(sd)
        assert "separate Q/K/V" in layer_map.architecture_notes
