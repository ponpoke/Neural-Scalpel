import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class PairedActivationDataset:
    """
    Dataset containing paired hidden states from source and target models
    collected over a common set of calibration prompts.
    """
    source_activations: Dict[str, torch.Tensor]  # {layer_name: tensor(n, d_s)}
    target_activations: Dict[str, torch.Tensor]  # {layer_name: tensor(n, d_t)}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate sample counts match
        for key in self.source_activations:
            if key in self.target_activations:
                s_n = self.source_activations[key].shape[0]
                t_n = self.target_activations[key].shape[0]
                if s_n != t_n:
                    raise ValueError(f"Sample count mismatch for {key}: source={s_n}, target={t_n}")

@dataclass
class AlignmentMap:
    """
    Represents the learned transformation (translation matrix P) 
    between source and target latent manifolds.
    """
    layer_maps: Dict[str, torch.Tensor]  # {source_layer: P_matrix(d_s, d_t)}
    source_model_id: str
    target_model_id: str
    method: str = "ridge"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def project(self, layer_name: str, source_delta: torch.Tensor) -> torch.Tensor:
        """Projects a source behavioral delta into the target manifold."""
        if layer_name not in self.layer_maps:
            raise KeyError(f"No alignment map found for layer: {layer_name}")
        
        P = self.layer_maps[layer_name]
        return torch.matmul(source_delta, P)

@dataclass
class BehavioralDelta:
    """
    Encapsulates the difference in hidden states (activations) 
    caused by an adapter in the source model.
    """
    layer_deltas: Dict[str, torch.Tensor]  # {layer_name: delta_tensor(n, d_s)}
    source_model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransportedDelta:
    """
    A BehavioralDelta that has been projected into the target manifold.
    """
    layer_deltas: Dict[str, torch.Tensor]  # {target_layer: delta_tensor(n, d_t)}
    target_model_id: str
    alignment_metadata: Dict[str, Any] = field(default_factory=dict)
