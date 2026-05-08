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

@dataclass
class LayerCorrespondence:
    """
    Represents the estimated correspondence between source and target layers.
    """
    target_to_source: Dict[str, str]
    scores: Dict[str, List[Dict[str, float]]]
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device) / n
    I = torch.eye(n, device=K.device)
    H = I - unit
    return H @ K @ H

def gram_linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Linear CKA between two feature matrices using Gram matrices.
    """
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.t()
    L = Y @ Y.t()

    Kc = center_gram(K)
    Lc = center_gram(L)

    denom = torch.norm(Kc) * torch.norm(Lc)
    if denom < eps:
        return 0.0

    return ((Kc * Lc).sum() / denom).item()

def estimate_layer_correspondence(
    dataset: PairedActivationDataset,
    method: str = "linear_cka",
    top_k: int = 3,
    device: str = "cuda"
) -> LayerCorrespondence:
    """
    Heuristically estimates which source layers correspond to which target layers.
    """
    target_to_source = {}
    scores = {}
    
    target_layers = list(dataset.target_activations.keys())
    source_layers = list(dataset.source_activations.keys())
    
    for t_layer in target_layers:
        t_acts = dataset.target_activations[t_layer].to(device)
        layer_scores = []
        
        for s_layer in source_layers:
            s_acts = dataset.source_activations[s_layer].to(device)
            
            if method == "linear_cka":
                score = gram_linear_cka(s_acts, t_acts)
            else:
                raise ValueError(f"Unknown correspondence method: {method}")
                
            layer_scores.append({"source_layer": s_layer, "score": score})
        
        # Sort by score descending
        layer_scores.sort(key=lambda x: x["score"], reverse=True)
        scores[t_layer] = layer_scores[:top_k]
        target_to_source[t_layer] = layer_scores[0]["source_layer"]
        
    return LayerCorrespondence(
        target_to_source=target_to_source,
        scores=scores,
        method=method,
        metadata={"num_source": len(source_layers), "num_target": len(target_layers)}
    )
