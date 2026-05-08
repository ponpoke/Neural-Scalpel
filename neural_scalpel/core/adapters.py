import torch
import re
import math
import warnings
from typing import Dict, Any, Tuple, Union, Optional, List
from dataclasses import dataclass, field

from neural_scalpel.core.math import (
    soft_routing_head_pooling, 
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
                 scaling_config: Optional[AdaptiveScalingConfig] = None,
                 piecewise_modules: Optional[List[str]] = None,
                 piecewise_layers: Optional[List[int]] = None,
                 piecewise_max_layers: Optional[int] = None):
        
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
        
        self.piecewise_modules = piecewise_modules
        self.piecewise_layers = piecewise_layers
        self.piecewise_max_layers = piecewise_max_layers
        self._piecewise_counter = 0
        self._pair_buffer = {}

    def get_adaptive_scale(self, key: str) -> float:
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

    def finalize(self):
        if self._pair_buffer:
            orphans = list(self._pair_buffer.keys())
            warnings.warn(f"Unprocessed LoRA pairs found in buffer: {orphans}. These layers were skipped. "
                          f"Check your state_dict keys for consistency.", RuntimeWarning)

# TODO: Rename this class to LLMStructuralProjectionAdapter.
# It now handles general CausalLM LoRA structural projection, not only Llama3->Qwen2.
class Llama3ToQwen2Adapter(BaseAdapter):
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], 
                 target_rank: int = 16, routing_matrix: torch.Tensor = None, 
                 delta_health: Any = None, projection_mode: str = "linear", 
                 scaling_config: Optional[AdaptiveScalingConfig] = None,
                 piecewise_modules: Optional[List[str]] = None,
                 piecewise_layers: Optional[List[int]] = None,
                 piecewise_max_layers: Optional[int] = None):
        super().__init__(source_info, target_info, delta_health, projection_mode, scaling_config, 
                         piecewise_modules, piecewise_layers, piecewise_max_layers)
        self.routing_matrix = routing_matrix
        
        self.source_hidden = self.source_hidden
        self.source_inter = self.source_inter
        self.source_heads = self.source_heads
        
        # Safe access for optional/extended info
        if isinstance(source_info, dict):
            self.source_kv_heads = source_info.get("num_key_value_heads", self.source_heads)
        else:
            # Fallback for tuple inputs from tests
            self.source_kv_heads = self.source_heads
            
        self.source_head_dim = self.source_hidden // self.source_heads
        self.source_kv_hidden = self.source_kv_heads * self.source_head_dim

        self.target_hidden = self.target_hidden
        self.target_inter = self.target_inter
        self.target_heads = self.target_heads

        if isinstance(target_info, dict):
            self.target_kv_heads = target_info.get("num_key_value_heads", self.target_heads)
        else:
            self.target_kv_heads = self.target_heads
            
        self.target_head_dim = self.target_hidden // self.target_heads
        self.target_kv_hidden = self.target_kv_heads * self.target_head_dim

        self.target_rank = target_rank
        self._warned_experimental = False

    def project_tensor(self, key: str, tensor: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor], None]:
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])
        
        if self.projection_mode in ["kernel", "jacobian"] and not self._warned_experimental:
            warnings.warn(f"[EXPERIMENTAL] {self.projection_mode} mode is a research stub. Falls back to linear.", RuntimeWarning)
            self._warned_experimental = True

        if self.projection_mode == "piecewise" and is_mlp:
            # v2.9 Hardening: Check piecewise constraints
            m = self.layer_pattern.search(key)
            layer_idx = int(m.group(1)) if m else None
            
            use_piecewise = True
            if self.piecewise_modules and not any(m in key for m in self.piecewise_modules):
                use_piecewise = False
            if self.piecewise_layers and layer_idx is not None and layer_idx not in self.piecewise_layers:
                use_piecewise = False
            if self.piecewise_max_layers is not None and self._piecewise_counter >= self.piecewise_max_layers:
                # If we've already done enough piecewise layers, fallback to linear for others
                use_piecewise = False
            
            if not use_piecewise:
                scale = self.get_adaptive_scale(key)
                projected = self._legacy_project(key, tensor)
                return projected * scale

            # Piecewise logic continues...
            # Hardening (v2.8.1): Robust pair buffering for streaming
            # LoRA weights often appear as A then B (or vice versa). We must wait for the pair.
            base_key = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            if base_key not in self._pair_buffer:
                self._pair_buffer[base_key] = {}
            
            suffix = "A" if "lora_A" in key else "B"
            self._pair_buffer[base_key][suffix] = tensor
            
            if "A" in self._pair_buffer[base_key] and "B" in self._pair_buffer[base_key]:
                self._piecewise_counter += 1
                pair = self._pair_buffer.pop(base_key)
                A, B = pair["A"], pair["B"]
                
                # [Mathematical Rigor] Reconstruct the full delta manifold before projection.
                # ΔW = B @ A. This is the non-linear manifold we want to align.
                delta = B.float() @ A.float()
                
                # Piecewise SVD projection extracts the most significant energy components.
                # v2.8 fix: SVD projection is performed on the joint manifold.
                r_tgt = self.target_rank
                
                # Factorization back to LoRA pair (A_new, B_new)
                # Note: We must resize the joint delta to target dimensions first.
                src_in, src_out = A.shape[1], B.shape[0]
                if any(x in key for x in ["k_proj", "v_proj"]):
                    tgt_in, tgt_out = self.target_hidden, self.target_kv_hidden
                elif "down_proj" in key:
                    tgt_in, tgt_out = self.target_inter, self.target_hidden
                else:
                    # gate_proj, up_proj, q_proj, o_proj
                    tgt_in, tgt_out = (self.target_hidden, self.target_inter) if any(x in key for x in ["gate_proj", "up_proj"]) else (self.target_hidden, self.target_hidden)
                
                # Resize joint manifold
                delta_resized = delta[:tgt_out, :tgt_in] if src_out > tgt_out or src_in > tgt_in else torch.nn.functional.pad(delta, (0, max(0, tgt_in - src_in), 0, max(0, tgt_out - src_out)))
                
                # Factorize joint delta into new LoRA rank
                A_new, B_new = factorize_to_lora(delta_resized, r_tgt)
                
                # [Mathematical Rigor] Adaptive scaling application.
                # To maintain (B*s) @ (A*s) = B @ A @ s, we apply sqrt(scale) to both components.
                scale = self.get_adaptive_scale(key)
                sqrt_scale = scale ** 0.5
                
                return {
                    f"{base_key}.lora_A.weight": A_new * sqrt_scale,
                    f"{base_key}.lora_B.weight": B_new * sqrt_scale
                }
            
            # Key was buffered, return None to skip individual writing
            return None

        scale = self.get_adaptive_scale(key)
        projected = self._legacy_project(key, tensor)
        return projected * scale

    def _legacy_project(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        is_qkv = any(proj in key for proj in ["q_proj", "k_proj", "v_proj"])
        is_o = "o_proj" in key
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])
        
        # Determine feature dimensions based on module type and LoRA side (A/B)
        if "lora_A" in key:
            # lora_A: [rank, in_features] -> dim 1 is features
            src_dim = self.source_inter if "down_proj" in key else self.source_hidden
            tgt_dim = self.target_inter if "down_proj" in key else self.target_hidden
        else:
            # lora_B: [out_features, rank] -> dim 0 is features
            if any(x in key for x in ["k_proj", "v_proj"]):
                src_dim = self.source_kv_hidden
                tgt_dim = self.target_kv_hidden
            elif "down_proj" in key:
                src_dim = self.source_hidden
                tgt_dim = self.target_hidden
            else:
                # gate_proj, up_proj, q_proj, o_proj
                src_dim = self.source_inter if any(x in key for x in ["gate_proj", "up_proj"]) else self.source_hidden
                tgt_dim = self.target_inter if any(x in key for x in ["gate_proj", "up_proj"]) else self.target_hidden

        # Rank projection
        src_rank = tensor.shape[0] if "lora_A" in key else tensor.shape[1]
        tgt_rank = self.target_rank
        
        if "lora_A" in key:
            # First project features (dim 1)
            t = self._project_dim(tensor, src_dim, tgt_dim, dim=1)
            # Then project rank (dim 0)
            return t[:tgt_rank, :] if src_rank > tgt_rank else torch.nn.functional.pad(t, (0, 0, 0, max(0, tgt_rank - src_rank)))
        elif "lora_B" in key:
            # First project features (dim 0)
            t = self._project_dim(tensor, src_dim, tgt_dim, dim=0)
            # Then project rank (dim 1)
            return t[:, :tgt_rank] if src_rank > tgt_rank else torch.nn.functional.pad(t, (0, max(0, tgt_rank - src_rank), 0, 0))
        return tensor

    def _project_dim(self, tensor: torch.Tensor, src_dim: int, tgt_dim: int, dim: int) -> torch.Tensor:
        """Corrected bug (v2.8 fix): Unified variable names and ensured functional consistency."""
        if src_dim == tgt_dim:
            return tensor

        if dim == 1:
            if src_dim > tgt_dim:
                return tensor[:, :tgt_dim]
            return torch.nn.functional.pad(tensor, (0, tgt_dim - src_dim))

        # dim == 0 (Fix: removed undefined src_out / tgt_out)
        if src_dim > tgt_dim:
            return tensor[:tgt_dim, :]
        return torch.nn.functional.pad(tensor, (0, 0, 0, tgt_dim - src_dim))

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

# Backward Compatibility Aliases and Specialized Stubs
MistralToLlama3Adapter = Llama3ToQwen2Adapter

class SDXLToSDXLAdapter(BaseAdapter):
    """Passthrough adapter for same-architecture SDXL transplantation."""
    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

class SDXLToFluxAdapter(BaseAdapter):
    """Adapter for SDXL to Flux transplantation using PCSI."""
    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        from neural_scalpel.core.math import pca_guided_subspace_injection
        # Simple heuristic for SDXL->Flux dimension mapping
        if tensor.ndim == 2:
            src_dim = tensor.shape[1] if "lora_A" in key else tensor.shape[0]
            tgt_dim = 3072 if "lora_A" in key else 3072 # Flux default
            return pca_guided_subspace_injection(tensor, torch.randn(1, tgt_dim))
        return tensor

def get_adapter(source_arch: str, target_arch: str, source_info: Any, target_info: Any, 
                rank: int = 16, delta_health: Any = None, projection_mode: str = "linear", 
                scaling_config: Optional[AdaptiveScalingConfig] = None,
                piecewise_modules: Optional[List[str]] = None,
                piecewise_layers: Optional[List[int]] = None,
                piecewise_max_layers: Optional[int] = None) -> BaseAdapter:
    pair = f"{source_arch}_to_{target_arch}".lower()
    
    # [v2.9.1 Hardening] Restrict Structural Projection to LLM families
    source_arch_l = source_arch.lower()
    target_arch_l = target_arch.lower()
    llm_arches = {"llama", "qwen", "mistral", "gemma"}

    if source_arch_l in llm_arches and target_arch_l in llm_arches:
        return Llama3ToQwen2Adapter(source_info, target_info, target_rank=rank, delta_health=delta_health, 
                                     projection_mode=projection_mode, scaling_config=scaling_config,
                                     piecewise_modules=piecewise_modules, piecewise_layers=piecewise_layers,
                                     piecewise_max_layers=piecewise_max_layers)
    
    if source_arch_l == "sdxl" and target_arch_l == "flux":
        return SDXLToFluxAdapter(source_info, target_info, delta_health=delta_health,
                                 projection_mode=projection_mode, scaling_config=scaling_config)
                                 
    if source_arch_l == "sdxl" and target_arch_l == "sdxl":
        return SDXLToSDXLAdapter(source_info, target_info, delta_health=delta_health,
                                 projection_mode=projection_mode, scaling_config=scaling_config)
    
    return BaseAdapter(source_info, target_info, delta_health=delta_health, 
                        projection_mode=projection_mode, scaling_config=scaling_config,
                        piecewise_modules=piecewise_modules, piecewise_layers=piecewise_layers,
                        piecewise_max_layers=piecewise_max_layers)
