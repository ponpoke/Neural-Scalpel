"""
Neural-Scalpel Model Layer Discovery

Automatic discovery and classification of model layers for route payload mapping.
Supports common transformer architectures (Qwen, Llama, OPT, GPT-NeoX, etc.)
by introspecting named_parameters and matching against known layer patterns.

Usage:
    from neural_scalpel.serving.model_layer_discovery import discover_layers
    layer_map = discover_layers(model)
    # Returns: {"attention": [...], "mlp": [...], "norm": [...], ...}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LayerType(Enum):
    """Classification of transformer layer components."""
    ATTENTION_Q = "attention_q_proj"
    ATTENTION_K = "attention_k_proj"
    ATTENTION_V = "attention_v_proj"
    ATTENTION_O = "attention_o_proj"
    ATTENTION_QKV_FUSED = "attention_qkv_fused"
    MLP_GATE = "mlp_gate_proj"
    MLP_UP = "mlp_up_proj"
    MLP_DOWN = "mlp_down_proj"
    MLP_FC1 = "mlp_fc1"
    MLP_FC2 = "mlp_fc2"
    NORM = "norm"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    OTHER = "other"


@dataclass
class DiscoveredLayer:
    """A single discovered layer with its metadata."""
    name: str
    layer_type: LayerType
    shape: List[int]
    dtype: str
    block_index: Optional[int] = None  # transformer block number


@dataclass
class ModelLayerMap:
    """Complete layer map for a model, organized by component type."""
    model_type: str = "unknown"
    num_blocks: int = 0
    layers: List[DiscoveredLayer] = field(default_factory=list)
    swappable_layers: List[str] = field(default_factory=list)
    architecture_notes: str = ""

    def by_type(self, layer_type: LayerType) -> List[DiscoveredLayer]:
        return [l for l in self.layers if l.layer_type == layer_type]

    def by_block(self, block_index: int) -> List[DiscoveredLayer]:
        return [l for l in self.layers if l.block_index == block_index]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "num_blocks": self.num_blocks,
            "total_layers": len(self.layers),
            "swappable_count": len(self.swappable_layers),
            "architecture_notes": self.architecture_notes,
            "layers": [
                {
                    "name": l.name,
                    "type": l.layer_type.value,
                    "shape": l.shape,
                    "dtype": l.dtype,
                    "block": l.block_index,
                }
                for l in self.layers
            ],
        }


# ── Layer Pattern Matching ─────────────────────────────────────────────

# Patterns are (regex, LayerType) pairs. Order matters: first match wins.
_LAYER_PATTERNS: List[Tuple[re.Pattern, LayerType]] = [
    # Qwen2 / Llama / Mistral style
    (re.compile(r"\.q_proj\."), LayerType.ATTENTION_Q),
    (re.compile(r"\.k_proj\."), LayerType.ATTENTION_K),
    (re.compile(r"\.v_proj\."), LayerType.ATTENTION_V),
    (re.compile(r"\.o_proj\."), LayerType.ATTENTION_O),
    (re.compile(r"\.gate_proj\."), LayerType.MLP_GATE),
    (re.compile(r"\.up_proj\."), LayerType.MLP_UP),
    (re.compile(r"\.down_proj\."), LayerType.MLP_DOWN),
    # OPT / GPT-NeoX style
    (re.compile(r"\.qkv_proj\."), LayerType.ATTENTION_QKV_FUSED),
    (re.compile(r"\.query_key_value\."), LayerType.ATTENTION_QKV_FUSED),
    (re.compile(r"\.fc1\."), LayerType.MLP_FC1),
    (re.compile(r"\.fc2\."), LayerType.MLP_FC2),
    # Normalization
    (re.compile(r"layernorm|layer_norm|input_layernorm|post_attention_layernorm|ln_"), LayerType.NORM),
    (re.compile(r"\.norm"), LayerType.NORM),
    # Embeddings
    (re.compile(r"embed_tokens|wte|wpe|word_embedding"), LayerType.EMBEDDING),
    # LM Head
    (re.compile(r"lm_head|output\.weight"), LayerType.LM_HEAD),
]

# Block index extraction pattern
_BLOCK_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.|\.h\.(\d+)\.|\.blocks\.(\d+)\.")

# Swappable layer types (attention + MLP projections)
_SWAPPABLE_TYPES = {
    LayerType.ATTENTION_Q, LayerType.ATTENTION_K, LayerType.ATTENTION_V,
    LayerType.ATTENTION_O, LayerType.ATTENTION_QKV_FUSED,
    LayerType.MLP_GATE, LayerType.MLP_UP, LayerType.MLP_DOWN,
    LayerType.MLP_FC1, LayerType.MLP_FC2,
}

# Architecture detection patterns
_ARCH_PATTERNS = {
    "qwen2": re.compile(r"model\.layers\.\d+\.self_attn\.q_proj"),
    "llama": re.compile(r"model\.layers\.\d+\.self_attn\.q_proj"),
    "opt": re.compile(r"model\.decoder\.layers\.\d+\.self_attn"),
    "gpt_neox": re.compile(r"gpt_neox\.layers\.\d+\.attention\.query_key_value"),
    "phi": re.compile(r"model\.layers\.\d+\.self_attn\.qkv_proj"),
}


def _classify_layer(name: str) -> LayerType:
    """Classifies a parameter name into a LayerType using pattern matching."""
    for pattern, layer_type in _LAYER_PATTERNS:
        if pattern.search(name):
            return layer_type
    return LayerType.OTHER


def _extract_block_index(name: str) -> Optional[int]:
    """Extracts the transformer block index from a parameter name."""
    match = _BLOCK_INDEX_PATTERN.search(name)
    if match:
        for group in match.groups():
            if group is not None:
                return int(group)
    return None


def _detect_architecture(param_names: List[str]) -> str:
    """Detects the model architecture from parameter naming conventions."""
    joined = "\n".join(param_names[:100])
    for arch_name, pattern in _ARCH_PATTERNS.items():
        if pattern.search(joined):
            return arch_name
    return "unknown"


def discover_layers(
    model,
    include_non_weight: bool = False,
) -> ModelLayerMap:
    """
    Discovers and classifies all layers in a model.

    Args:
        model: A PyTorch nn.Module or a dict of {name: tensor}.
        include_non_weight: If True, includes bias and norm parameters.

    Returns:
        ModelLayerMap with complete layer classification.
    """
    if isinstance(model, dict):
        named_params = list(model.items())
    elif hasattr(model, "named_parameters"):
        named_params = list(model.named_parameters())
    else:
        raise TypeError(f"Cannot discover layers from {type(model)}")

    param_names = [name for name, _ in named_params]
    arch = _detect_architecture(param_names)

    layer_map = ModelLayerMap(model_type=arch)
    max_block = -1

    for name, param in named_params:
        # Skip bias unless explicitly included
        if not include_non_weight and name.endswith(".bias"):
            continue

        layer_type = _classify_layer(name)
        block_idx = _extract_block_index(name)

        if block_idx is not None and block_idx > max_block:
            max_block = block_idx

        shape = list(param.shape) if hasattr(param, "shape") else []
        dtype = str(param.dtype).replace("torch.", "") if hasattr(param, "dtype") else "unknown"

        discovered = DiscoveredLayer(
            name=name,
            layer_type=layer_type,
            shape=shape,
            dtype=dtype,
            block_index=block_idx,
        )
        layer_map.layers.append(discovered)

        if layer_type in _SWAPPABLE_TYPES:
            layer_map.swappable_layers.append(name)

    layer_map.num_blocks = max_block + 1 if max_block >= 0 else 0

    # Architecture notes
    fused_qkv = layer_map.by_type(LayerType.ATTENTION_QKV_FUSED)
    if fused_qkv:
        layer_map.architecture_notes = "Uses fused QKV projection"
    else:
        separate_q = layer_map.by_type(LayerType.ATTENTION_Q)
        if separate_q:
            layer_map.architecture_notes = "Uses separate Q/K/V projections"

    return layer_map
