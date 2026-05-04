import torch
import re
from typing import Dict, Any, Tuple, Union
from transformers import AutoConfig
from neural_scalpel.core.math import soft_routing_head_pooling, pca_guided_subspace_injection

class BaseAdapter:
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict]):
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

    def map_key(self, key: str) -> str:
        """Translates the state dict key from source to target architecture."""
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Projects the tensor shape/values."""
        return tensor

class Llama3ToQwen2Adapter(BaseAdapter):
    """
    Adapter for mapping Llama-3 (e.g. 4096 dim, 32 heads) 
    to Qwen-2 (e.g. 3584 dim, 28 heads).
    Demonstrates Soft-Routing Head Pooling (SRHP) for head reduction,
    and supports Wasserstein Discrete Routing (WDR) to avoid robotomy.
    """
    def __init__(self, source_info: Union[Tuple, Dict], target_info: Union[Tuple, Dict], routing_matrix: torch.Tensor = None):
        super().__init__(source_info, target_info)
        self.routing_matrix = routing_matrix # (s_heads, t_heads)

    def map_key(self, key: str) -> str:
        # standard PEFT naming is mostly homogeneous, but we can intercept here
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        # Determine module type
        is_qkv = any(proj in key for proj in ["q_proj", "k_proj", "v_proj"])
        is_o = "o_proj" in key
        is_mlp = any(proj in key for proj in ["gate_proj", "up_proj", "down_proj"])

        if "lora_A" in key:
            # lora_A shape: (r, in_features)
            r = tensor.shape[0]
            in_features = tensor.shape[1]
            
            if is_qkv:
                # input is main hidden_size
                if in_features == self.source_hidden:
                    return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=1)
            elif is_mlp:
                if "down_proj" in key:
                    # down_proj input is intermediate_size
                    return self._project_dim(tensor, self.source_inter, self.target_inter, dim=1)
                else:
                    # gate/up input is hidden_size
                    return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=1)
            elif is_o:
                # input is num_heads * head_dim
                if in_features == self.source_hidden:
                    # Prioritize WDR if routing matrix is available
                    if self.routing_matrix is not None:
                        return self._apply_wdr_on_in_features(tensor, r)
                    return self._apply_srhp_on_in_features(tensor, r)
                    
        elif "lora_B" in key:
            # lora_B shape: (out_features, r)
            out_features = tensor.shape[0]
            r = tensor.shape[1]
            
            if is_qkv:
                # k_proj and v_proj outputs are kv_dim, q_proj is hidden_size
                if out_features == self.source_hidden:
                    # q_proj
                    if self.routing_matrix is not None:
                        return self._apply_wdr_on_out_features(tensor, r)
                    return self._apply_srhp_on_out_features(tensor, r)
                elif out_features == self.source_kv_dim:
                    # k_proj / v_proj
                    return self._project_dim(tensor, self.source_kv_dim, self.target_kv_dim, dim=0)
            elif is_o:
                # output is main hidden_size
                if out_features == self.source_hidden:
                    return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0)
            elif is_mlp:
                if "down_proj" in key:
                    # down_proj output is hidden_size
                    return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0)
                else:
                    # gate/up output is intermediate_size
                    return self._project_dim(tensor, self.source_inter, self.target_inter, dim=0)

        return tensor

    def _project_dim(self, tensor: torch.Tensor, src_dim: int, tgt_dim: int, dim: int) -> torch.Tensor:
        """Simple truncation or padding along a dimension."""
        if src_dim == tgt_dim:
            return tensor
            
        if dim == 1: # (r, dim)
            if src_dim > tgt_dim:
                return tensor[:, :tgt_dim]
            else:
                return torch.nn.functional.pad(tensor, (0, tgt_dim - src_dim))
        else: # (dim, r)
            if src_dim > tgt_dim:
                return tensor[:tgt_dim, :]
            else:
                return torch.nn.functional.pad(tensor, (0, 0, 0, tgt_dim - src_dim))

    def _apply_srhp_on_out_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        """Applies SRHP on lora_B (out_features, r) where out_features = heads * head_dim"""
        if self.source_heads == self.target_heads:
            return tensor
            
        # tensor shape: (s_heads * h_dim, r)
        tensor_reshaped = tensor.view(self.source_heads, self.source_head_dim, r)
        tensor_transposed = tensor_reshaped.permute(2, 0, 1) # (r, s_heads, h_dim)
        
        # Apply Soft-Routing Head Pooling
        pooled = soft_routing_head_pooling(tensor_transposed, self.target_heads) # (r, t_heads, h_dim)
        
        h_dim_actual = pooled.shape[2]
        
        # truncate/pad to target_head_dim
        if h_dim_actual > self.target_head_dim:
            pooled = pooled[:, :, :self.target_head_dim]
        elif h_dim_actual < self.target_head_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.target_head_dim - h_dim_actual))
            
        out_pooled = pooled.permute(1, 2, 0).reshape(self.target_heads * self.target_head_dim, r)
        return out_pooled

    def _apply_srhp_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        """Applies SRHP on lora_A (r, in_features) where in_features = heads * head_dim"""
        if self.source_heads == self.target_heads:
            return tensor
            
        # tensor shape: (r, s_heads * h_dim)
        tensor_reshaped = tensor.view(r, self.source_heads, self.source_head_dim)
        
        # Apply SRHP
        pooled = soft_routing_head_pooling(tensor_reshaped, self.target_heads) # (r, t_heads, h_dim)
        
        h_dim_actual = pooled.shape[2]
        
        # We must truncate/pad h_dim_actual to target_head_dim
        # pooled shape is (r, t_heads, h_dim_actual)
        if h_dim_actual > self.target_head_dim:
            pooled = pooled[:, :, :self.target_head_dim]
        elif h_dim_actual < self.target_head_dim:
            pooled = torch.nn.functional.pad(pooled, (0, self.target_head_dim - h_dim_actual))
            
        return pooled.reshape(r, self.target_heads * self.target_head_dim)

    def _apply_wdr_on_out_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        """Applies WDR (Wasserstein Discrete Routing) using pre-computed routing_matrix."""
        # tensor shape: (s_heads * h_dim, r)
        # routing_matrix shape: (s_heads, t_heads)
        s_heads, t_heads = self.routing_matrix.shape
        tensor_reshaped = tensor.view(s_heads, self.source_head_dim, r)
        
        # WDR: t_head_j = sum_i (P_ij * s_head_i)
        # Using einsum for routing: (s_heads, t_heads), (s_heads, h_dim, r) -> (t_heads, h_dim, r)
        routed = torch.einsum('st,shr->thr', self.routing_matrix, tensor_reshaped)
        
        return routed.reshape(t_heads * self.source_head_dim, r)

    def _apply_wdr_on_in_features(self, tensor: torch.Tensor, r: int) -> torch.Tensor:
        """Applies WDR (Wasserstein Discrete Routing) on lora_A."""
        # tensor shape: (r, s_heads * h_dim)
        s_heads, t_heads = self.routing_matrix.shape
        tensor_reshaped = tensor.view(r, s_heads, self.source_head_dim)
        
        # WDR: (r, s_heads, h_dim), (s_heads, t_heads) -> (r, t_heads, h_dim)
        routed = torch.einsum('rsh,st->rth', tensor_reshaped, self.routing_matrix)
        return routed.reshape(r, t_heads * self.source_head_dim)


class MistralToLlama3Adapter(BaseAdapter):
    """
    Adapter bridging Mistral-v0.3 and Llama-3.
    Demonstrates architectural dictionary generalization.
    """
    def map_key(self, key: str) -> str:
        # Standardizing typical key variations between architectures
        key = re.sub(r'mistral_attn', 'self_attn', key)
        key = re.sub(r'mistral_mlp', 'mlp', key)
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return self._project_dim(tensor, self.source_hidden, self.target_hidden, dim=0 if "lora_B" in key else 1)
        
    def _project_dim(self, tensor: torch.Tensor, src_dim: int, tgt_dim: int, dim: int) -> torch.Tensor:
        if src_dim == tgt_dim:
            return tensor
        if dim == 1:
            return tensor[:, :tgt_dim] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, tgt_dim - src_dim))
        else:
            return tensor[:tgt_dim, :] if src_dim > tgt_dim else torch.nn.functional.pad(tensor, (0, 0, 0, tgt_dim - src_dim))

class SDXLToFluxAdapter(BaseAdapter):
    """
    Adapter projecting SDXL (UNet) latent concepts into FLUX (DiT) space.
    Demonstrates Principal Component Subspace Injection (PCSI).
    """
    def map_key(self, key: str) -> str:
        # Vision models have completely different keys (e.g., UNet to DiT)
        key = re.sub(r'.*?attn1\.to_q', 'single_transformer_blocks.0.attn.to_q', key)
        key = re.sub(r'.*?attn1\.to_k', 'single_transformer_blocks.0.attn.to_k', key)
        key = re.sub(r'.*?attn1\.to_v', 'single_transformer_blocks.0.attn.to_v', key)
        key = re.sub(r'.*?attn1\.to_out\.0', 'single_transformer_blocks.0.attn.to_out', key)
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        # PCSI projects smaller latent spaces onto the principal components of larger ones
        if "lora_A" in key:
            source_dim = tensor.shape[1]
            if source_dim < self.target_hidden:
                # Simulate target activations (N > target_hidden for valid SVD)
                target_activations = torch.randn(self.target_hidden + 10, self.target_hidden, device=tensor.device, dtype=tensor.dtype)
                return pca_guided_subspace_injection(tensor, target_activations)
            elif source_dim > self.target_hidden:
                return tensor[:, :self.target_hidden]
        elif "lora_B" in key:
            source_dim = tensor.shape[0]
            if source_dim < self.target_hidden:
                target_activations = torch.randn(self.target_hidden + 10, self.target_hidden, device=tensor.device, dtype=tensor.dtype)
                tensor_t = tensor.t() # (r, source_dim)
                injected_t = pca_guided_subspace_injection(tensor_t, target_activations)
                return injected_t.t() # (target_hidden, r)
            elif source_dim > self.target_hidden:
                return tensor[:self.target_hidden, :]
        return tensor


class SDXLToSDXLAdapter(BaseAdapter):
    """
    Adapter for SDXL-to-SDXL or SD-Turbo transplantation.
    Ensures keys are preserved and correctly formatted for Diffusers.
    """
    def map_key(self, key: str) -> str:
        # For SDXL, standard LoRA keys like 'lora_unet_...' should be kept as is.
        # This allows diffusers to correctly identify them.
        return key

    def project_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        # In a real surgery, we might apply JTSA or WDR here.
        # For now, we ensure the tensor reaches the target with its stylistic data intact.
        return tensor


def get_adapter(source_arch: str, target_arch: str, source_info: Tuple[int, int], target_info: Tuple[int, int]) -> BaseAdapter:
    # A simple router. In a real scenario, infer `source_arch` and `target_arch` from the config.json `model_type`
    pair = f"{source_arch}_to_{target_arch}".lower()
    
    if "llama" in pair and "qwen" in pair:
        return Llama3ToQwen2Adapter(source_info, target_info)
    elif "mistral" in pair and "llama" in pair:
        return MistralToLlama3Adapter(source_info, target_info)
    elif "sdxl" in pair and "flux" in pair:
        return SDXLToFluxAdapter(source_info, target_info)
    elif "sdxl" in pair and "sdxl" in pair:
        return SDXLToSDXLAdapter(source_info, target_info)
    
    return BaseAdapter(source_info, target_info)
