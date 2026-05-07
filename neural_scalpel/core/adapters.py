import torch
import re
import math
import warnings
from typing import Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass, field

from neural_scalpel.core.math import (
    soft_routing_head_pooling, 
    pca_guided_subspace_injection,
    piecewise_svd_projection,
    factorize_to_lora
)

@dataclass
class AdaptiveScalingConfig:
    moderately_concentrated_scale: float = 0.9
    low_normalized_entropy_scale: float = 0.8
    outlier_layer_scale: float = 0.7
    min_scale: float = 0.2
    max_scale: float = 1.0
    enabled: bool = True

class BaseAdapter:
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], 
                 delta_health: Any = None, projection_mode: str = "linear",
                 scaling_config: Optional[AdaptiveScalingConfig] = None):
        
        if isinstance(source_info, dict):
            self.source_hidden = source_info.get("hidden_size", 4096)
            self.source_heads = source_info.get("num_attention_heads", 32)
            self.source_inter = source_info.get("intermediate_size", 14336)
        else:
            self.source_hidden, self.source_heads = source_info[:2]
            self.source_inter = source_info[2] if len(source_info) > 2 else 14336

        if isinstance(target_info, dict):
            self.target_hidden = target_info.get("hidden_size", 4096)
            self.target_heads = target_info.get("num_attention_heads", 32)
            self.target_inter = target_info.get("intermediate_size", 14336)
        else:
            self.target_hidden, self.target_heads = target_info[:2]
            self.target_inter = target_info[2] if len(target_info) > 2 else 14336
            
        self.source_head_dim = self.source_hidden // self.source_heads
        self.target_head_dim = self.target_hidden // self.target_heads
        
        self.delta_health = delta_health
        self.projection_mode = projection_mode
        self.scaling_config = scaling_config or AdaptiveScalingConfig()
        self.layer_pattern = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")
        
        # Internal buffer for pair-aware projection (v2.8)
        self._pair_buffer = {}

    def get_adaptive_scale(self, key: str) -> float:
        """Configurable Adaptive Scaling (v2.7)."""
        if not self.scaling_config.enabled or self.delta_health is None:
            return 1.0
            
        m = self.layer_pattern.search(key)
        layer_idx = int(m.group(1)) if m else None
        
        scale = 1.0
        if self.delta_health.verdict == "MODERATELY_CONCENTRATED":
            scale *= self.scaling_config.moderately_concentrated_scale
            
        if hasattr(self.delta_health, "normalized_spectral_entropy") and self.delta_health.normalized_spectral_entropy < 0.5:
            scale *= self.scaling_config.low_normalized_entropy_scale
            
        if layer_idx is not None and hasattr(self.delta_health, "outliers"):
            for outlier in self.delta_health.outliers:
                if f"Layer {layer_idx}" in outlier:
                    scale *= self.scaling_config.outlier_layer_scale
                    break
        
        final_scale = max(self.scaling_config.min_scale, min(self.scaling_config.max_scale, scale))
        if hasattr(self.delta_health, "applied_scales"):
            self.delta_health.applied_scales[key] = final_scale
        return final_scale

    def map_key(self, key: str) -> str:
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        return tensor

class Llama3ToQwen2Adapter(BaseAdapter):
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], 
                 routing_matrix: torch.Tensor = None, delta_health: Any = None, 
                 projection_mode: str = "linear", scaling_config: Optional[AdaptiveScalingConfig] = None):
        super().__init__(source_info, target_info, delta_health, projection_mode, scaling_config)
        self.routing_matrix = routing_matrix
        self._warned_experimental = False

    def project_tensor(self, key: str, tensor: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])
        
        # v2.9 Research Stub Warning
        if self.projection_mode in ["kernel", "jacobian"] and not self._warned_experimental:
            warnings.warn(f"[EXPERIMENTAL] {self.projection_mode} mode is a research stub. Falls back to linear.", RuntimeWarning)
            self._warned_experimental = True

        # v2.8 Piecewise: LoRA Pair-Aware Implementation
        if self.projection_mode == "piecewise" and is_mlp:
            if "lora_A" in key:
                self._pair_buffer[key.replace("lora_A", "pair")] = tensor
                return None # Defer until B arrives
            elif "lora_B" in key:
                pair_key = key.replace("lora_B", "pair")
                if pair_key not in self._pair_buffer: return tensor # Fallback
                A = self._pair_buffer.pop(pair_key)
                B = tensor
                
                # Joint Projection (v2.8 Core Logic)
                delta = B.float() @ A.float()
                r = A.shape[0]
                
                # Apply Piecewise SVD on Delta
                U, S, Vh = piecewise_svd_projection(delta, r)
                
                # Re-factorize to target shape
                # Calculate target intermediate/hidden dims
                dim = 1 # A is (r, in)
                src_in = A.shape[1]
                src_out = B.shape[0]
                
                if "down_proj" in key:
                    tgt_in, tgt_out = self.target_inter, self.target_hidden
                else:
                    tgt_in, tgt_out = self.target_hidden, self.target_inter
                
                # Resize delta first (shape projection)
                delta_resized = delta[:tgt_out, :tgt_in] if src_out > tgt_out or src_in > tgt_in else torch.nn.functional.pad(delta, (0, max(0, tgt_in - src_in), 0, max(0, tgt_out - src_out)))
                
                # Re-LoRA
                A_new, B_new = factorize_to_lora(delta_resized, r)
                
                scale = self.get_adaptive_scale(key)
                return {
                    key.replace("lora_B", "lora_A"): A_new * scale,
                    key: B_new * scale
                }

        # Legacy / Linear path
        scale = self.get_adaptive_scale(key)
        projected = self._legacy_project(key, tensor)
        return projected * scale

    def _legacy_project(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        is_qkv = any(proj in key for proj in ["q_proj", "k_proj", "v_proj"])
        is_o = "o_proj" in key
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])
        r = tensor.shape[1] if "lora_B" in key else tensor.shape[0]
        
        if "lora_A" in key:
            in_features = tensor.shape[1]
            if is_qkv and in_features == self.source_hidden:
                return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=1)
            elif is_mlp:
                src_dim = self.source_inter if "down_proj" in key else self.source_hidden
                tgt_dim = self.target_inter if "down_proj" in key else self.target_hidden
                return self._project_dim(tensor, src_dim, tgt_dim, dim=1)
            elif is_o and in_features == self.source_hidden:
                return self._apply_wdr_on_in_features(tensor, r) if self.routing_matrix is not None else self._apply_srhp_on_in_features(tensor, r)
        elif "lora_B" in key:
            out_features = tensor.shape[0]
            if is_qkv:
                if out_features == self.source_hidden:
                    return self._apply_wdr_on_out_features(tensor, r) if self.routing_matrix is not None else self._apply_srhp_on_out_features(tensor, r)
                return self._project_dim(tensor, out_features, self.target_hidden // (self.source_hidden // out_features), dim=0)
            elif is_o and out_features == self.source_hidden:
                return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0)
            elif is_mlp:
                src_dim = self.source_hidden if "down_proj" in key else self.source_inter
                tgt_dim = self.target_hidden if "down_proj" in key else self.target_inter
                return self._project_dim(tensor, src_dim, tgt_dim, dim=0)
        return tensor

    def _project_dim(self, tensor: torch.Tensor, src_dim: int, tgt_dim: int, dim: int) -> torch.Tensor:
        if src_dim == tgt_dim: return tensor
        if dim == 1:
            return tensor[:, :tgt_dim] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, tgt_dim - src_dim))
        else:
            return tensor[:tgt_dim, :] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, 0, 0, tgt_dim - src_dim))

    def _apply_srhp_on_out_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        if self.source_heads == self.target_heads: return tensor
        tensor_reshaped = tensor.view(self.source_heads, self.source_head_dim, r)
        pooled = soft_routing_head_pooling(tensor_reshaped.permute(2, 0, 1), self.target_heads)
        return pooled.permute(1, 2, 0).reshape(-1, r)

    def _apply_srhp_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        if self.source_heads == self.target_heads: return tensor
        tensor_reshaped = tensor.view(r, self.source_heads, self.source_head_dim)
        pooled = soft_routing_head_pooling(tensor_reshaped, self.target_heads)
        return pooled.reshape(r, -1)

    def _apply_wdr_on_out_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        s_heads, t_heads = self.routing_matrix.shape
        routed = torch.einsum('st,shr->thr', self.routing_matrix, tensor.view(s_heads, self.source_head_dim, r))
        return routed.reshape(-1, r)

    def _apply_wdr_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        s_heads, t_heads = self.routing_matrix.shape
        routed = torch.einsum('rsh,st->rth', tensor.view(r, s_heads, self.source_head_dim), self.routing_matrix)
        return routed.reshape(r, -1)

def get_adapter(source_arch: str, target_arch: str, source_info: Tuple[int, int], target_info: Tuple[int, int], 
                delta_health: Any = None, projection_mode: str = "linear", 
                scaling_config: Optional[AdaptiveScalingConfig] = None) -> BaseAdapter:
    pair = f"{source_arch}_to_{target_arch}".lower()
    if "llama" in pair and "qwen" in pair:
        return Llama3ToQwen2Adapter(source_info, target_info, delta_health=delta_health, 
                                     projection_mode=projection_mode, scaling_config=scaling_config)
    return BaseAdapter(source_info, target_info, delta_health=delta_health, 
                        projection_mode=projection_mode, scaling_config=scaling_config)
