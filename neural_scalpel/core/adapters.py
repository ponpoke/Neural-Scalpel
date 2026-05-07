import torch
import re
import math
from typing import Dict, Any, Tuple, Union, Optional
from transformers import AutoConfig
from neural_scalpel.core.math import (
    soft_routing_head_pooling, 
    pca_guided_subspace_injection,
    piecewise_svd_projection,
    kernel_orthogonal_procrustes,
    jacobian_tangent_space_alignment
)

class BaseAdapter:
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], 
                 delta_health: Any = None, projection_mode: str = "linear"):
        if isinstance(source_info, tuple):
            self.source_hidden, self.source_heads = source_info[:2]
            self.source_inter = source_info[2] if len(source_info) > 2 else 14336
            self.source_kv_heads = source_info[3] if len(source_info) > 3 else 8
        else:
            self.source_hidden = source_info.get("hidden_size", 4096)
            self.source_heads = source_info.get("num_attention_heads", 32)
            self.source_inter = source_info.get("intermediate_size", 14336)
            self.source_kv_heads = source_info.get("num_key_value_heads", 8)

        if isinstance(target_info, tuple):
            self.target_hidden, self.target_heads = target_info[:2]
            self.target_inter = target_info[2] if len(target_info) > 2 else 14336
            self.target_kv_heads = target_info[3] if len(target_info) > 3 else 8
        else:
            self.target_hidden = target_info.get("hidden_size", 4096)
            self.target_heads = target_info.get("num_attention_heads", 32)
            self.target_inter = target_info.get("intermediate_size", 14336)
            self.target_kv_heads = target_info.get("num_key_value_heads", 8)
            
        self.source_head_dim = self.source_hidden // self.source_heads
        self.target_head_dim = self.target_hidden // self.target_heads
        self.source_kv_dim = self.source_kv_heads * self.source_head_dim
        self.target_kv_dim = self.target_kv_heads * self.target_head_dim
        
        self.delta_health = delta_health
        self.projection_mode = projection_mode

    def get_adaptive_scale(self, key: str) -> float:
        """Adaptive Scaling (v2.7) logic."""
        if self.delta_health is None: return 1.0
        parts = key.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        scale = 1.0
        if self.delta_health.verdict == "MODERATELY_CONCENTRATED": scale *= 0.9
        if hasattr(self.delta_health, "effective_rank") and self.delta_health.effective_rank < 2.0: scale *= 0.8
        if layer_idx is not None and hasattr(self.delta_health, "concentration_score") and self.delta_health.concentration_score > 0.4:
            for outlier in getattr(self.delta_health, "outliers", []):
                if f"Layer {layer_idx}" in outlier:
                    scale *= 0.7
                    break
        return scale

    def map_key(self, key: str) -> str:
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

class Llama3ToQwen2Adapter(BaseAdapter):
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], 
                 routing_matrix: torch.Tensor = None, delta_health: Any = None, projection_mode: str = "linear"):
        super().__init__(source_info, target_info, delta_health, projection_mode)
        self.routing_matrix = routing_matrix

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        is_qkv = any(proj in key for proj in ["q_proj", "k_proj", "v_proj"])
        is_o = "o_proj" in key
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])
        scale = self.get_adaptive_scale(key)
        r = tensor.shape[1] if "lora_B" in key else tensor.shape[0]
        
        # Piecewise / Non-linear selection (v2.8-v2.9)
        if self.projection_mode == "piecewise" and is_mlp:
             return self._apply_piecewise_projection(tensor, key, r)
        elif self.projection_mode == "kernel" and is_qkv:
             return self._apply_kernel_projection(tensor, key, r)
        elif self.projection_mode == "jacobian" and is_mlp:
             return self._apply_jacobian_projection(tensor, key, r)

        # Default linear projection
        projected = tensor
        if "lora_A" in key:
            in_features = tensor.shape[1]
            if is_qkv and in_features == self.source_hidden:
                projected = self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=1)
            elif is_mlp:
                src_dim = self.source_inter if "down_proj" in key else self.source_hidden
                tgt_dim = self.target_inter if "down_proj" in key else self.target_hidden
                projected = self._project_dim(tensor, src_dim, tgt_dim, dim=1)
            elif is_o and in_features == self.source_hidden:
                projected = self._apply_wdr_on_in_features(tensor, r) if self.routing_matrix is not None else self._apply_srhp_on_in_features(tensor, r)
        elif "lora_B" in key:
            out_features = tensor.shape[0]
            if is_qkv:
                if out_features == self.source_hidden:
                    projected = self._apply_wdr_on_out_features(tensor, r) if self.routing_matrix is not None else self._apply_srhp_on_out_features(tensor, r)
                elif out_features == self.source_kv_dim:
                    projected = self._project_dim(tensor, self.source_kv_dim, self.target_kv_dim, dim=0)
            elif is_o and out_features == self.source_hidden:
                projected = self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0)
            elif is_mlp:
                src_dim = self.source_hidden if "down_proj" in key else self.source_inter
                tgt_dim = self.target_hidden if "down_proj" in key else self.target_inter
                projected = self._project_dim(tensor, src_dim, tgt_dim, dim=0)

        return projected * scale

    def _apply_piecewise_projection(self, tensor: torch.Tensor, key: str, r: int) -> torch.Tensor:
        dim = 1 if "lora_A" in key else 0
        src_dim = tensor.shape[1] if dim == 1 else tensor.shape[0]
        if "down_proj" in key: tgt_dim = self.target_inter if dim == 1 else self.target_hidden
        else: tgt_dim = self.target_hidden if dim == 1 else self.target_inter
        U, S, Vh = piecewise_svd_projection(tensor, r, mid_scale=0.95, low_scale=0.5)
        if dim == 1:
            return Vh[:, :tgt_dim] * S[:, None] if src_dim > tgt_dim else torch.nn.functional.pad(Vh * S[:, None], (0, tgt_dim - src_dim))
        else:
            return U[:tgt_dim, :] * S[None, :] if src_dim > tgt_dim else torch.nn.functional.pad(U * S[None, :], (0, 0, 0, tgt_dim - src_dim))

    def _apply_kernel_projection(self, tensor: torch.Tensor, key: str, r: int) -> torch.Tensor:
        dim = 1 if "lora_A" in key else 0
        src_dim, tgt_dim = (tensor.shape[1], self.target_hidden) if dim == 1 else (tensor.shape[0], self.target_hidden)
        # Synthetic fallback if no calibration data
        return self._project_dim(tensor, src_dim, tgt_dim, dim=dim)

    def _apply_jacobian_projection(self, tensor: torch.Tensor, key: str, r: int) -> torch.Tensor:
        dim = 1 if "lora_A" in key else 0
        src_dim = tensor.shape[1] if dim == 1 else tensor.shape[0]
        if "down_proj" in key: tgt_dim = self.target_inter if dim == 1 else self.target_hidden
        else: tgt_dim = self.target_hidden if dim == 1 else self.target_inter
        return self._project_dim(tensor, src_dim, tgt_dim, dim=dim)

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
        h_dim_actual = pooled.shape[2]
        if h_dim_actual > self.target_head_dim: pooled = pooled[:, :, :self.target_head_dim]
        elif h_dim_actual < self.target_head_dim: pooled = torch.nn.functional.pad(pooled, (0, self.target_head_dim - h_dim_actual))
        return pooled.permute(1, 2, 0).reshape(self.target_heads * self.target_head_dim, r)

    def _apply_srhp_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        if self.source_heads == self.target_heads: return tensor
        tensor_reshaped = tensor.view(r, self.source_heads, self.source_head_dim)
        pooled = soft_routing_head_pooling(tensor_reshaped, self.target_heads)
        h_dim_actual = pooled.shape[2]
        if h_dim_actual > self.target_head_dim: pooled = pooled[:, :, :self.target_head_dim]
        elif h_dim_actual < self.target_head_dim: pooled = torch.nn.functional.pad(pooled, (0, self.target_head_dim - h_dim_actual))
        return pooled.reshape(r, self.target_heads * self.target_head_dim)

    def _apply_wdr_on_out_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        s_heads, t_heads = self.routing_matrix.shape
        routed = torch.einsum('st,shr->thr', self.routing_matrix, tensor.view(s_heads, self.source_head_dim, r))
        return routed.reshape(t_heads * self.source_head_dim, r)

    def _apply_wdr_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        s_heads, t_heads = self.routing_matrix.shape
        routed = torch.einsum('rsh,st->rth', tensor.view(r, s_heads, self.source_head_dim), self.routing_matrix)
        return routed.reshape(r, t_heads * self.source_head_dim)

class MistralToLlama3Adapter(BaseAdapter):
    def map_key(self, key: str) -> str:
        key = re.sub(r'mistral_attn', 'self_attn', key)
        key = re.sub(r'mistral_mlp', 'mlp', key)
        return key
    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.get_adaptive_scale(key)
        projected = self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0 if "lora_B" in key else 1)
        return projected * scale
    def _project_dim(self, tensor: torch.Tensor, src_dim: int, tgt_dim: int, dim: int) -> torch.Tensor:
        if src_dim == tgt_dim: return tensor
        if dim == 1: return tensor[:, :tgt_dim] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, tgt_dim - src_dim))
        else: return tensor[:tgt_dim, :] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, 0, 0, tgt_dim - src_dim))

class SDXLToFluxAdapter(BaseAdapter):
    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.get_adaptive_scale(key)
        projected = tensor
        if "lora_A" in key:
            source_dim = tensor.shape[1]
            if source_dim < self.target_hidden:
                target_activations = torch.randn(self.target_hidden + 10, self.target_hidden, device=tensor.device, dtype=tensor.dtype)
                projected = pca_guided_subspace_injection(tensor, target_activations)
            elif source_dim > self.target_hidden: projected = tensor[:, :self.target_hidden]
        elif "lora_B" in key:
            source_dim = tensor.shape[0]
            if source_dim < self.target_hidden:
                target_activations = torch.randn(self.target_hidden + 10, self.target_hidden, device=tensor.device, dtype=tensor.dtype)
                projected = pca_guided_subspace_injection(tensor.t(), target_activations).t()
            elif source_dim > self.target_hidden: projected = tensor[:self.target_hidden, :]
        return projected * scale

class SDXLToSDXLAdapter(BaseAdapter):
    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.get_adaptive_scale(key)
        return tensor * scale

def get_adapter(source_arch: str, target_arch: str, source_info: Tuple[int, int], target_info: Tuple[int, int], 
                delta_health: Any = None, projection_mode: str = "linear") -> BaseAdapter:
    pair = f"{source_arch}_to_{target_arch}".lower()
    if "llama" in pair and "qwen" in pair:
        return Llama3ToQwen2Adapter(source_info, target_info, delta_health=delta_health, projection_mode=projection_mode)
    elif "mistral" in pair and "llama" in pair:
        return MistralToLlama3Adapter(source_info, target_info, delta_health=delta_health, projection_mode=projection_mode)
    elif "sdxl" in pair and "flux" in pair:
        return SDXLToFluxAdapter(source_info, target_info, delta_health=delta_health, projection_mode=projection_mode)
    elif "sdxl" in pair and "sdxl" in pair:
        return SDXLToSDXLAdapter(source_info, target_info, delta_health=delta_health, projection_mode=projection_mode)
    return BaseAdapter(source_info, target_info, delta_health=delta_health, projection_mode=projection_mode)
