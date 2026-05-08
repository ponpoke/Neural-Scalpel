import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from neural_scalpel.core.alignment import (
    PairedActivationDataset, 
    AlignmentMap, 
    BehavioralDelta, 
    TransportedDelta,
    estimate_layer_correspondence
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
    prompt_formatter: Optional[Any] = None,
    auto_correspondence: bool = False,
    correspondence_method: str = "linear_cka",
    device: str = "cuda"
) -> AlignmentMap:
    """
    Learns the translation matrix P between source and target models.
    """
    if not calibration_prompts:
        raise ValueError("calibration_prompts must not be empty.")
    if not source_layers or not target_layers:
        raise ValueError("source_layers and target_layers must not be empty.")
    if not auto_correspondence and len(source_layers) != len(target_layers):
        raise ValueError("Source and target layer lists must match in length unless auto_correspondence=True.")

    source_model.to(device).eval()
    target_model.to(device).eval()

    with collection_context(source_model, source_layers) as source_coll, \
         collection_context(target_model, target_layers) as target_coll:
        
        for prompt in calibration_prompts:
            text = prompt
            if prompt_formatter:
                text = prompt_formatter(tokenizer, prompt)
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                source_model(**inputs)
                target_model(**inputs)
        
        source_acts = source_coll.get_stacked()
        target_acts = target_coll.get_stacked()

    metadata = {
        "num_prompts": len(calibration_prompts),
        "source_layers": source_layers,
        "target_layers": target_layers,
        "prompt_formatting": {
            "used": prompt_formatter is not None,
            "formatter": prompt_formatter.__name__ if prompt_formatter else None
        }
    }

    dataset = PairedActivationDataset(
        source_activations=source_acts,
        target_activations=target_acts,
        metadata=metadata
    )
    
    if auto_correspondence:
        correspondence = estimate_layer_correspondence(dataset, method=correspondence_method, device=device)
        metadata["auto_correspondence"] = {
            "method": correspondence_method,
            "mapping": correspondence.target_to_source
        }
        
        layer_maps = {}
        for t_layer, s_layer in correspondence.target_to_source.items():
            X = dataset.source_activations[s_layer]
            Y = dataset.target_activations[t_layer]
            P = solve_ridge(X, Y)
            layer_maps[t_layer] = P
            
        return AlignmentMap(
            layer_maps=layer_maps,
            source_model_id=getattr(source_model.config, "_name_or_path", "src"),
            target_model_id=getattr(target_model.config, "_name_or_path", "tgt"),
            method=method,
            metadata={**metadata, "map_key_semantics": "target_layer"}
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
        
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError(f"Non-finite activations in source layer {s_layer}")
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            raise ValueError(f"Non-finite activations in target layer {s_layer}")

        if method == "ridge":
            P = solve_ridge(X, Y, alpha=alpha)
            layer_maps[s_layer] = P
            
    if not layer_maps:
        raise RuntimeError("No matching layers found to learn alignment map.")

    return AlignmentMap(
        layer_maps=layer_maps,
        source_model_id=dataset.metadata.get("source_model_id", "unknown"),
        target_model_id=dataset.metadata.get("target_model_id", "unknown"),
        method=method,
        metadata={**dataset.metadata, "map_key_semantics": "source_layer"}
    )

def extract_behavior_delta(
    base_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    prompts: List[str],
    tokenizer: Any,
    layers: List[str],
    prompt_formatter: Optional[Any] = None,
    device: str = "cuda"
) -> BehavioralDelta:
    """
    Captures the activation shift caused by an adapter in the source model.
    """
    if not prompts:
        raise ValueError("prompts must not be empty.")
    if not layers:
        raise ValueError("layers must not be empty.")

    base_model.to(device).eval()
    adapted_model.to(device).eval()
    
    with collection_context(base_model, layers) as base_coll, \
         collection_context(adapted_model, layers) as adapted_coll:
        
        for prompt in prompts:
            text = prompt
            if prompt_formatter:
                text = prompt_formatter(tokenizer, prompt)
                
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                base_model(**inputs)
                adapted_model(**inputs)
        
        base_acts = base_coll.get_stacked()
        adapted_acts = adapted_coll.get_stacked()
        
    deltas = {layer: (adapted_acts[layer] - base_acts[layer]) for layer in layers}
    
    return BehavioralDelta(
        layer_deltas=deltas,
        source_model_id=getattr(base_model.config, "_name_or_path", "unknown"),
        metadata={
            "num_prompts": len(prompts),
            "layers": layers,
            "prompt_formatting": prompt_formatter is not None
        }
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
        if torch.isnan(d_s).any() or torch.isinf(d_s).any():
            raise ValueError(f"Non-finite delta detected in layer {layer}")

        if layer in mapping.layer_maps:
            d_t = mapping.project(layer, d_s)
            
            if torch.isnan(d_t).any() or torch.isinf(d_t).any():
                 raise ValueError(f"Non-finite projected delta produced for layer {layer}")
                 
            transported[layer] = d_t
            
    return TransportedDelta(
        layer_deltas=transported,
        target_model_id=mapping.target_model_id,
        alignment_metadata=mapping.metadata
    )

def solve_activation_adapter(
    target_model: torch.nn.Module,
    desired_delta: TransportedDelta,
    prompts: List[str],
    tokenizer: Any,
    target_modules: List[str],
    module_to_delta_layer: Optional[Dict[str, str]] = None,
    prompt_formatter: Optional[Any] = None,
    strict: bool = True,
    device: str = "cuda"
) -> ActivationAdapterSolution:
    """
    Computes the weight changes required to replicate a transported delta.
    """
    if not prompts:
        raise ValueError("prompts must not be empty.")
    if not target_modules:
        raise ValueError("target_modules must not be empty.")

    target_model.to(device).eval()
    
    with collection_context(target_model, target_modules) as coll:
        for prompt in prompts:
            text = prompt
            if prompt_formatter:
                text = prompt_formatter(tokenizer, prompt)
                
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                target_model(**inputs)
        inputs_stacked = coll.get_stacked()
        
    weights = {}
    errors = {}
    solved_modules = []
    skipped_modules = []

    for module in target_modules:
        delta_key = module_to_delta_layer.get(module, module) if module_to_delta_layer else module
        
        if module not in inputs_stacked:
            if strict:
                raise ValueError(f"Activation not collected for module: {module}")
            skipped_modules.append(module)
            continue
            
        if delta_key not in desired_delta.layer_deltas:
            if strict:
                raise ValueError(f"No transported delta found for key: {delta_key} (mapped from {module})")
            skipped_modules.append(module)
            continue

        X = inputs_stacked[module]
        Y = desired_delta.layer_deltas[delta_key]
        
        if torch.isnan(X).any() or torch.isinf(X).any() or torch.isnan(Y).any() or torch.isinf(Y).any():
            raise ValueError(f"Non-finite values detected in activations for module {module}")

        W = solve_ridge(X, Y)
        weights[module] = W
        
        # Error metric
        Y_pred = torch.matmul(X, W)
        errors[module] = (torch.norm(Y - Y_pred) / torch.norm(Y)).item()
        solved_modules.append(module)
            
    metadata = {
        "num_prompts": len(prompts),
        "module_to_delta_layer": module_to_delta_layer,
        "solved_modules": solved_modules,
        "skipped_modules": skipped_modules,
        "prompt_formatting": prompt_formatter is not None
    }

    return ActivationAdapterSolution(
        module_weights=weights,
        target_model_id=desired_delta.target_model_id,
        reconstruction_errors=errors,
        metadata=metadata
    )

def build_peft_lora_key(
    module_name: str,
    which: str, # "A" or "B"
    peft_key_prefix: str = "base_model.model",
    adapter_name: Optional[str] = None,
    key_style: str = "peft_default",
) -> str:
    """
    Constructs a PEFT-compatible parameter key.
    """
    if key_style == "peft_default":
        # base_model.model.model.layers.10.mlp.down_proj.lora_A.weight
        return f"{peft_key_prefix}.{module_name}.lora_{which}.weight"
    elif key_style == "peft_named":
        # base_model.model.model.layers.10.mlp.down_proj.lora_A.default.weight
        name = adapter_name or "default"
        return f"{peft_key_prefix}.{module_name}.lora_{which}.{name}.weight"
    elif key_style == "raw":
        return f"{module_name}.lora_{which}.weight"
    else:
        raise ValueError(f"Unknown key_style: {key_style}")

def export_lora(
    solution: ActivationAdapterSolution,
    rank: int = 16,
    lora_alpha: Optional[int] = None,
    target_modules: Optional[List[str]] = None,
    peft_key_prefix: str = "base_model.model",
    adapter_name: Optional[str] = None,
    key_style: str = "peft_default"
) -> PeftExportResult:
    """
    Compresses full-rank solutions into PEFT-compatible low-rank adapters.
    """
    if not solution.module_weights:
        raise RuntimeError("No modules were successfully solved in the solution.")

    if lora_alpha is None:
        lora_alpha = rank * 2
        
    lora_state_dict = {}
    for module_name, W in solution.module_weights.items():
        # Decompose
        A, B = low_rank_decompose_for_peft(W, rank)
        
        # PEFT Key Formatting
        key_a = build_peft_lora_key(module_name, "A", peft_key_prefix, adapter_name, key_style)
        key_b = build_peft_lora_key(module_name, "B", peft_key_prefix, adapter_name, key_style)
        
        lora_state_dict[key_a] = A
        lora_state_dict[key_b] = B
        
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules or list(solution.module_weights.keys()),
        "bias": "none"
    }
    
    mean_err = sum(solution.reconstruction_errors.values()) / len(solution.reconstruction_errors)
    
    metadata = {
        "peft_key_prefix": peft_key_prefix,
        "adapter_name": adapter_name,
        "key_style": key_style,
        "target_model_id": solution.target_model_id
    }
    
    return PeftExportResult(
        lora_state_dict=lora_state_dict,
        config=config,
        rank=rank,
        lora_alpha=lora_alpha,
        mean_error=mean_err,
        metadata=metadata
    )

def validate_behavior(
    base_model: torch.nn.Module,
    adapter_path: Union[str, Path],
    prompts: List[str],
    tokenizer: Any,
    prompt_formatter: Optional[Any] = None,
    checks: List[str] = ["logit_kl", "top1_shift"],
    kl_threshold: float = 1e-6,
    require_nonzero_adapter: bool = True,
    device: str = "cuda"
) -> ValidationReport:
    """
    Executes a behavioral validation suite to confirm signal delivery.
    """
    from peft import PeftModel
    import torch.nn.functional as F
    
    if not prompts:
        raise ValueError("validate_behavior requires at least one prompt.")

    report = ValidationReport(phase="G8", status="PENDING")
    
    try:
        # G7: PEFT Load Check
        base_model.to(device).eval()
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
            report.add_gate("G7", True, "PEFT Adapter loaded successfully.")
        except Exception as e:
            report.status = "FAIL"
            report.summary = "ARTIFACT_LOAD_FAILED"
            report.add_gate("G7", False, f"PEFT load failed: {str(e)}", severity="critical")
            return report

        # Check for zero weights
        if require_nonzero_adapter:
            has_params = False
            is_zero = True
            for name, param in model.named_parameters():
                if "lora_" in name:
                    has_params = True
                    if torch.norm(param) > 0:
                        is_zero = False
                        break
            if not has_params:
                report.status = "FAIL"
                report.summary = "ADAPTER_NOT_ACTIVE"
                report.add_gate("G7", False, "No LoRA parameters found in the model.", severity="critical")
                return report
            if is_zero:
                report.status = "FAIL"
                report.summary = "ADAPTER_NOT_ACTIVE"
                report.add_gate("G7", False, "LoRA parameters are all zero.", severity="critical")
                return report

        results = {"kl": [], "shifts": 0}
        
        for prompt in prompts:
            text = prompt
            if prompt_formatter:
                text = prompt_formatter(tokenizer, prompt)
                
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                # Correct comparison: Use the same 'model' instance for both
                with model.disable_adapter():
                    base_logits = model(**inputs).logits[:, -1, :]
                
                # Adapter logit
                adapter_logits = model(**inputs).logits[:, -1, :]
                
            if torch.isnan(base_logits).any() or torch.isinf(base_logits).any() or \
               torch.isnan(adapter_logits).any() or torch.isinf(adapter_logits).any():
                report.status = "FAIL"
                report.summary = "NUMERICALLY_UNSTABLE"
                report.add_gate("G8", False, "Non-finite values detected in logits.", severity="critical")
                return report

            # Logit KL - Using log_softmax for numerical stability
            p = F.softmax(base_logits, dim=-1)
            log_q = F.log_softmax(adapter_logits, dim=-1)
            kl = F.kl_div(log_q, p, reduction="batchmean").item()
            results["kl"].append(kl)
            
            # Top-1 Shift
            if base_logits.argmax() != adapter_logits.argmax():
                results["shifts"] += 1
                
        mean_kl = sum(results["kl"]) / len(results["kl"])
        shift_rate = results["shifts"] / len(prompts)
        
        report.metrics = {"mean_kl": mean_kl, "shift_rate": shift_rate}
        
        # G8: Behavioral Shift Check
        if shift_rate > 0:
            report.status = "PASS"
            report.summary = "BEHAVIORAL_SHIFT_DETECTED"
            report.add_gate("G8", True, f"Behavioral shift observed: {shift_rate:.1%} tokens changed.")
        elif mean_kl >= kl_threshold:
            report.status = "PASS"
            report.summary = "LOGIT_SIGNAL_OBSERVED"
            report.add_gate("G8", True, f"Logit signal observed (KL={mean_kl:.2e}) but no Top-1 shift.")
        else:
            report.status = "WARNING"
            report.summary = "NO_MEANINGFUL_SIGNAL"
            report.add_gate("G8", False, f"No meaningful signal (KL={mean_kl:.2e}, shift={shift_rate:.1%}).", severity="warning")
            
    except Exception as e:
        report.status = "FAIL"
        report.add_gate("G7", False, f"Validation failed due to unexpected error: {str(e)}", severity="critical")
        
    return report
