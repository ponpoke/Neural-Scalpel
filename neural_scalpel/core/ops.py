import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from neural_scalpel.core.alignment import (
    PairedActivationDataset, 
    AlignmentMap, 
    BehavioralDelta, 
    TransportedDelta
)
from neural_scalpel.core.math import solve_ridge, low_rank_decompose_for_peft
from neural_scalpel.core.collector import collection_context
from neural_scalpel.core.solutions import ActivationAdapterSolution, PeftExportResult
from neural_scalpel.core.validation import ValidationReport

def align(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    calibration_prompts: List[str],
    tokenizer: Any,
    source_layers: List[str],
    target_layers: List[str],
    method: str = "ridge",
    device: str = "cuda"
) -> AlignmentMap:
    """
    Learns the translation matrix P between source and target models.
    """
    if len(source_layers) != len(target_layers):
        raise ValueError("Source and target layer lists must match in length.")

    source_model.to(device).eval()
    target_model.to(device).eval()

    with collection_context(source_model, source_layers) as source_coll, \
         collection_context(target_model, target_layers) as target_coll:
        
        for prompt in calibration_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                source_model(**inputs)
                target_model(**inputs)
        
        source_acts = source_coll.get_stacked()
        target_acts = target_coll.get_stacked()

    dataset = PairedActivationDataset(
        source_activations=source_acts,
        target_activations=target_acts,
        metadata={"num_prompts": len(calibration_prompts)}
    )
    
    return learn_alignment_map(dataset, method=method)

def learn_alignment_map(
    dataset: PairedActivationDataset,
    method: str = "ridge",
    alpha: float = 1.0
) -> AlignmentMap:
    """
    Learns the translation matrix P between source and target manifolds 
    from a pre-collected paired dataset.
    """
    layer_maps = {}
    for s_layer in dataset.source_activations:
        if s_layer not in dataset.target_activations:
            continue
            
        X = dataset.source_activations[s_layer]
        Y = dataset.target_activations[s_layer]
        
        if method == "ridge":
            P = solve_ridge(X, Y, alpha=alpha)
            layer_maps[s_layer] = P
            
    return AlignmentMap(
        layer_maps=layer_maps,
        source_model_id=dataset.metadata.get("source_model_id", "unknown"),
        target_model_id=dataset.metadata.get("target_model_id", "unknown"),
        method=method
    )

def extract_behavior_delta(
    base_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    prompts: List[str],
    tokenizer: Any,
    layers: List[str],
    device: str = "cuda"
) -> BehavioralDelta:
    """
    Captures the activation shift caused by an adapter in the source model.
    """
    base_model.to(device).eval()
    adapted_model.to(device).eval()
    
    with collection_context(base_model, layers) as base_coll, \
         collection_context(adapted_model, layers) as adapted_coll:
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                base_model(**inputs)
                adapted_model(**inputs)
        
        base_acts = base_coll.get_stacked()
        adapted_acts = adapted_coll.get_stacked()
        
    deltas = {layer: (adapted_acts[layer] - base_acts[layer]) for layer in layers}
    
    return BehavioralDelta(
        layer_deltas=deltas,
        source_model_id=getattr(base_model.config, "_name_or_path", "unknown")
    )

def transport_delta(
    delta: BehavioralDelta,
    mapping: AlignmentMap
) -> TransportedDelta:
    """
    Projects behavioral deltas into the target manifold.
    """
    transported = {}
    for layer, d_s in delta.layer_deltas.items():
        if layer in mapping.layer_maps:
            transported[layer] = mapping.project(layer, d_s)
            
    return TransportedDelta(
        layer_deltas=transported,
        target_model_id=mapping.target_model_id
    )

def solve_activation_adapter(
    target_model: torch.nn.Module,
    desired_delta: TransportedDelta,
    prompts: List[str],
    tokenizer: Any,
    target_modules: List[str],
    device: str = "cuda"
) -> ActivationAdapterSolution:
    """
    Computes the weight changes required to replicate a transported delta.
    """
    target_model.to(device).eval()
    
    with collection_context(target_model, target_modules) as coll:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                target_model(**inputs)
        inputs_stacked = coll.get_stacked()
        
    weights = {}
    errors = {}
    for module in target_modules:
        if module in inputs_stacked and module in desired_delta.layer_deltas:
            X = inputs_stacked[module]
            Y = desired_delta.layer_deltas[module]
            W = solve_ridge(X, Y)
            weights[module] = W
            
            # Error metric
            Y_pred = torch.matmul(X, W)
            errors[module] = (torch.norm(Y - Y_pred) / torch.norm(Y)).item()
            
    return ActivationAdapterSolution(
        module_weights=weights,
        target_model_id=desired_delta.target_model_id,
        reconstruction_errors=errors
    )

def export_lora(
    solution: ActivationAdapterSolution,
    rank: int = 16,
    lora_alpha: Optional[int] = None,
    target_modules: Optional[List[str]] = None
) -> PeftExportResult:
    """
    Compresses full-rank solutions into PEFT-compatible low-rank adapters.
    """
    if lora_alpha is None:
        lora_alpha = rank * 2
        
    lora_state_dict = {}
    for module_name, W in solution.module_weights.items():
        # Decompose
        A, B = low_rank_decompose_for_peft(W, rank)
        
        # PEFT Key Formatting (e.g., base_model.model.model.layers.N.mlp.down_proj.lora_A.weight)
        # Note: This prefixing depends on the specific loading context, but we use a standard one.
        prefix = f"base_model.model.{module_name}"
        lora_state_dict[f"{prefix}.lora_A.weight"] = A
        lora_state_dict[f"{prefix}.lora_B.weight"] = B
        
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules or list(solution.module_weights.keys()),
        "bias": "none"
    }
    
    mean_err = sum(solution.reconstruction_errors.values()) / len(solution.reconstruction_errors)
    
    return PeftExportResult(
        lora_state_dict=lora_state_dict,
        config=config,
        rank=rank,
        lora_alpha=lora_alpha,
        mean_error=mean_err
    )

def validate_behavior(
    base_model: torch.nn.Module,
    adapter_path: Union[str, Path],
    prompts: List[str],
    tokenizer: Any,
    checks: List[str] = ["logit_kl", "top1_shift"],
    device: str = "cuda"
) -> ValidationReport:
    """
    Executes a behavioral validation suite to confirm signal delivery.
    """
    from peft import PeftModel
    import torch.nn.functional as F
    
    report = ValidationReport(phase="G8", status="PENDING")
    
    try:
        # G7: PEFT Load Check
        base_model.to(device).eval()
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        report.add_gate("G7", True, "PEFT Adapter loaded successfully.")
        
        results = {"kl": [], "shifts": 0}
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                # Correct comparison: Use the same 'model' instance for both
                with model.disable_adapter():
                    base_logits = model(**inputs).logits[:, -1, :]
                
                # Adapter logit
                adapter_logits = model(**inputs).logits[:, -1, :]
                
            # Logit KL
            p = F.softmax(base_logits, dim=-1)
            q = F.softmax(adapter_logits, dim=-1)
            kl = F.kl_div(q.log(), p, reduction="batchmean").item()
            results["kl"].append(kl)
            
            # Top-1 Shift
            if base_logits.argmax() != adapter_logits.argmax():
                results["shifts"] += 1
                
        mean_kl = sum(results["kl"]) / len(results["kl"])
        shift_rate = results["shifts"] / len(prompts)
        
        report.metrics = {"mean_kl": mean_kl, "shift_rate": shift_rate}
        
        # G8: Behavioral Shift Check
        if shift_rate > 0:
            report.status = "SUCCESS"
            report.add_gate("G8", True, f"Behavioral shift observed: {shift_rate:.1%} tokens changed.")
        else:
            report.status = "WARNING"
            report.add_gate("G8", False, "No Top-1 token shifts observed. Signal might be too weak.")
            
    except Exception as e:
        report.status = "FAILURE"
        report.add_gate("G7", False, f"Validation failed: {str(e)}")
        
    return report
