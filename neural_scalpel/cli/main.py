import argparse
import os
import torch
import json
import warnings
from transformers import AutoConfig
from safetensors.torch import save_file, load_file

from neural_scalpel.core.adapters import get_adapter
from neural_scalpel.router.manager import ScalpelRouteManager
from neural_scalpel.experimental.hot_swap import VRAMHotSwapAPI
from neural_scalpel.io.factory import IOBridgeFactory

def get_model_info(model_path_or_name: str) -> dict:
    """Parses config.json to dynamically get architecture sizes."""
    try:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
        
        hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 4096))
        num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 32))
        intermediate_size = getattr(config, "intermediate_size", getattr(config, "hidden_size", 14336))
        kv_heads = getattr(config, "num_key_value_heads", getattr(config, "n_heads", num_heads))
        
        return {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate_size,
            "num_key_value_heads": kv_heads
        }
    except Exception as e:
        # Fallback for known architectures if config is missing (e.g. single file LoRA)
        print(f"[!] Warning: Could not fetch config for {model_path_or_name}: {e}. Using heuristics.")
        path_lower = model_path_or_name.lower()
        if "stable-diffusion-xl" in path_lower or "sdxl" in path_lower:
            return {"hidden_size": 2048, "num_attention_heads": 32, "intermediate_size": 2048, "num_key_value_heads": 32}
        if "0.5b" in path_lower and "qwen" in path_lower:
            return {"hidden_size": 896, "num_attention_heads": 14, "intermediate_size": 4864, "num_key_value_heads": 2}
        if "1.5b" in path_lower and "qwen" in path_lower:
            return {"hidden_size": 1536, "num_attention_heads": 12, "intermediate_size": 8960, "num_key_value_heads": 2}
        if "7b" in path_lower and "qwen" in path_lower:
            return {"hidden_size": 3584, "num_attention_heads": 28, "intermediate_size": 18944, "num_key_value_heads": 4}
            
        return {"hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 14336, "num_key_value_heads": 8} # Llama-3-8B default

def detect_architecture(path_or_name: str) -> str:
    """Generically detects model architecture from config or tensor keys."""
    # 1. Try AutoConfig (Hugging Face / Local Dir)
    try:
        config = AutoConfig.from_pretrained(path_or_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        if any(x in model_type for x in ["llama", "mistral", "gemma"]): return "llama"
        if "qwen" in model_type: return "qwen"
        if "stable-diffusion" in model_type or "unet" in model_type: return "sdxl"
    except Exception:
        pass

    # 2. Peek at Safetensors Keys (Single File LoRA)
    if os.path.isfile(path_or_name) and path_or_name.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(path_or_name, framework="pt") as f:
                keys = f.keys()
                if any("lora_unet" in k for k in keys): return "sdxl"
                if any("base_model.model.model.layers" in k for k in keys): return "llama"
        except Exception:
            pass

    # 3. Heuristic Fallback (Last Resort)
    path_lower = path_or_name.lower()
    if any(x in path_lower for x in ["sdxl", "diffusion", "stable", "animagine"]): return "sdxl"
    if "qwen" in path_lower: return "qwen"
    return "llama" # Default fallback

def port_lora(args):
    warnings.warn("[DEPRECATED] 'port' command is legacy. Use 'project-adapter' instead.", DeprecationWarning)
    print(f"Starting Concept-Projector (Neural-Scalpel) Transplantation Pipeline")
    
    # [v2.9.1 Hardening] Resolve Source Base vs Source Adapter
    source_adapter_path = args.source
    source_base = getattr(args, "source_base_model", None) or source_adapter_path
    
    print(f"Source Adapter: {source_adapter_path}")
    print(f"Source Base: {source_base}")
    print(f"Target Base: {args.target}")
    
    # [Error Handling] Validate source existence
    if not os.path.exists(source_adapter_path) and ("/" in source_adapter_path):
        print(f"[HF] Source adapter '{source_adapter_path}' not found locally. Attempting to download from Hugging Face...")
        try:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(repo_id=source_adapter_path)
            source_adapter_path = local_path
            print(f"[HF] Download complete. Using local path: {source_adapter_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download source adapter '{source_adapter_path}' from Hugging Face: {e}")
    elif not os.path.exists(source_adapter_path):
        raise FileNotFoundError(f"Source adapter '{source_adapter_path}' does not exist and is not a valid Hugging Face repository.")

    # Generic Architecture Detection - derived from BASE model
    source_arch = detect_architecture(source_base)
    target_arch = detect_architecture(args.target)
    
    source_info = get_model_info(source_base)
    target_info = get_model_info(args.target)
    
    print(f"Detected Source Arch: {source_arch.upper()} ({source_info['hidden_size']} dim, {source_info['num_attention_heads']} heads)")
    print(f"Detected Target Arch: {target_arch.upper()} ({target_info['hidden_size']} dim, {target_info['num_attention_heads']} heads)")
    
    # Initialize Bridges
    source_bridge = IOBridgeFactory.get_bridge(source_adapter_path)
    output_bridge = IOBridgeFactory.get_bridge(args.output)
    
    routing_matrix = None
    if args.routing_path and os.path.exists(args.routing_path):
        routing_matrix = torch.load(args.routing_path, weights_only=True)

    adapter = get_adapter(source_arch, target_arch, source_info, target_info, 
                          rank=getattr(args, "rank", 16),
                          delta_health=getattr(args, "delta_health", None),
                          projection_mode=getattr(args, "projection_mode", "linear"),
                          scaling_config=getattr(args, "scaling_config", None),
                          piecewise_modules=getattr(args, "piecewise_modules", None),
                          piecewise_layers=getattr(args, "piecewise_layers", None),
                          piecewise_max_layers=getattr(args, "piecewise_max_layers", None))
    if hasattr(adapter, "routing_matrix") and routing_matrix is not None:
        adapter.routing_matrix = routing_matrix

    # Ensure output directory exists
    output_path = args.output
    if output_path.endswith(".safetensors") or output_path.endswith(".gguf"):
        output_dir = os.path.dirname(output_path) or "."
    else:
        output_dir = output_path
        output_path = os.path.join(output_dir, "adapter_model.safetensors")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # [v2.9.1 Hardening] Module Inclusion Filtering
    include_modules = getattr(args, "include_modules", None)
    if isinstance(include_modules, str):
        include_modules = [m.strip() for m in include_modules.split(",")]
    
    # [v2.10] Module Alpha Map Parsing
    module_alpha_map = {}
    raw_alpha_map = getattr(args, "module_alpha_map", None)
    if raw_alpha_map:
        try:
            for item in raw_alpha_map.split(","):
                m_name, m_alpha = item.split("=")
                module_alpha_map[m_name.strip()] = float(m_alpha.strip())
            print(f"[v2.10] Using Module Alpha Map: {module_alpha_map}")
        except Exception as e:
            print(f"[!] Warning: Failed to parse --module-alpha-map '{raw_alpha_map}': {e}")

    standard_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def apply_module_alpha_scaling(key, new_key, new_tensor):
        if new_tensor is None:
            return None
            
        current_module = None
        for mod_candidate in standard_modules:
            if mod_candidate in key:
                current_module = mod_candidate
                break
        
        if current_module and current_module in module_alpha_map:
            target_alpha = module_alpha_map[current_module]
            reference_alpha = getattr(args, "alpha", 16)
            if target_alpha != reference_alpha:
                scale_factor = target_alpha / reference_alpha
                if isinstance(new_tensor, dict):
                    for k in new_tensor:
                        if "lora_B" in k: # Convention: scale B
                            new_tensor[k] = new_tensor[k] * scale_factor
                elif "lora_B" in new_key:
                    new_tensor = new_tensor * scale_factor
                print(f"    [v2.10] Scaled {current_module} by {scale_factor:.4f} (target alpha={target_alpha})")
        return new_tensor
    
    def get_module_from_key(k: str):
        for m in standard_modules:
            if m in k:
                return m
        return None

    def should_include_key(key):
        if "lora_" not in key:
            return True
            
        m = get_module_from_key(key)
        
        # 1. Standard exclusion by --include-modules
        if include_modules and not any(mod in key for mod in include_modules):
            return False
            
        # 2. Strict Gating: alpha=0 means exclusion
        if m and m in module_alpha_map and module_alpha_map[m] <= 0:
            return False
            
        return True

    actually_projected_modules = set()
    standard_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def record_projected_modules_from_key(k: str):
        for m in standard_modules:
            if m in k:
                actually_projected_modules.add(m)

    # Open the incremental writer
    output_bridge.open_writer(output_path)
    
    try:
        # [v2.9.1 Hardening] Use the resolved adapter path
        source_path = source_adapter_path
        
        print(f"[IO] Starting streaming iterator from {source_path}...")
        for key, tensor in source_bridge.iter_layers(source_path):
            if not should_include_key(key):
                continue
                
            print(f"  Surgery on {key}...")
            new_key = adapter.map_key(key)
            new_tensor = adapter.project_tensor(key, tensor)
            
            # [v2.10] Apply module-specific alpha scaling
            new_tensor = apply_module_alpha_scaling(key, new_key, new_tensor)
            
            if new_tensor is not None:
                if isinstance(new_tensor, dict):
                    # For pair-aware projection returning multiple tensors
                    for k, v in new_tensor.items():
                        record_projected_modules_from_key(k)
                        output_bridge.write_layer(k, v)
                else:
                    record_projected_modules_from_key(new_key)
                    output_bridge.write_layer(new_key, new_tensor)

            # Manual memory reclamation
            del tensor
            del new_tensor
        
        adapter.finalize()
        
        # [v2.11] Save PEFT config and Surgery Metadata
        config_path = os.path.join(output_dir, "adapter_config.json")
        peft_config = {
            "peft_type": "LORA",
            "r": getattr(args, "rank", 16),
            "lora_alpha": getattr(args, "alpha", 16),
            "target_modules": list(actually_projected_modules),
            "base_model_name_or_path": args.target
        }
        with open(config_path, "w") as f:
            json.dump(peft_config, f, indent=2)
            
        metadata_path = os.path.join(output_dir, "projection_metadata.json")
        metadata = {
            "global_alpha": getattr(args, "alpha", 16),
            "module_alpha_map": module_alpha_map,
            "source_adapter": source_adapter_path,
            "target_base": args.target,
            "actually_projected": list(actually_projected_modules)
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"[SUCCESS] Projected adapter saved to {output_dir}")

    except Exception as e:
        print(f"Streaming Surgery failed or not supported: {e}. Falling back to legacy load-all logic.")
        try:
            state_dict = source_bridge.load_weights(source_adapter_path)
        except Exception:
            # v2.9.1 Hardening: Prohibit dummy fallback unless explicitly allowed
            if not getattr(args, "allow_dummy_fallback", False):
                 raise RuntimeError(f"Physical weights not found for '{source_adapter_path}' and allow_dummy_fallback is False. Aborting to protect experimental integrity.")
                 
            # CI Fallback: Generate dummy tensors if all else fails
            print("Physical files not found. Simulating fallback state dict for verification...")
            s_hidden = source_info["hidden_size"]
            state_dict = {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, s_hidden),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(s_hidden, 16),
                "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.randn(16, s_hidden), 
                "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.randn(s_hidden, 16),
            }

        for key, tensor in state_dict.items():
            if not should_include_key(key):
                continue

            new_key = adapter.map_key(key)
            new_tensor = adapter.project_tensor(key, tensor)
            
            # [v2.10] Apply module-specific alpha scaling (fallback path)
            new_tensor = apply_module_alpha_scaling(key, new_key, new_tensor)
            
            if new_tensor is not None:
                if isinstance(new_tensor, dict):
                    for k, v in new_tensor.items():
                        record_projected_modules_from_key(k)
                        output_bridge.write_layer(k, v)
                else:
                    record_projected_modules_from_key(new_key)
                    output_bridge.write_layer(new_key, new_tensor)
        
        adapter.finalize()

    finally:
        output_bridge.close_writer()
    
    # Deduce 'r' from the first lora_A tensor (shape: [r, in_features])
    detected_r = 16 # fallback
    try:
        # Load just one tensor from source to get the rank
        # [v2.9.1 Hardening] source_path is the resolved directory or file
        if os.path.isdir(source_path):
            probe_path = os.path.join(source_path, "adapter_model.safetensors")
            if not os.path.exists(probe_path):
                probe_path = os.path.join(source_path, "model.safetensors")
        else:
            probe_path = source_path

        for k, t in load_file(probe_path).items():
            if "lora_A" in k:
                detected_r = t.shape[0]
                break
    except Exception:
        pass
        
    # Save adapter_config.json
    # v2.9 Hardening: Ensure 'r' reflects the TARGET rank (args.rank), not the source rank.
    target_r = getattr(args, "rank", detected_r)
    
    # Dynamic target modules based on what was actually projected
    # [v2.9.1 Hardening] Use deterministic order
    if actually_projected_modules:
        target_modules = [m for m in standard_modules if m in actually_projected_modules]
    else:
        target_modules = standard_modules

    adapter_config = {
        "peft_type": "LORA",
        "r": target_r,
        "lora_alpha": getattr(args, "alpha", target_r * 2),
        "target_modules": target_modules,
        "base_model_name_or_path": args.target
    }
    
    config_path = os.path.join(output_dir if output_dir else ".", "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
        
    # [v2.10] Save Projection Metadata
    from datetime import datetime, timezone
    metadata_path = os.path.join(output_dir if output_dir else ".", "projection_metadata.json")
    projection_metadata = {
        "source_adapter": args.source,
        "target_model": args.target,
        "global_alpha": getattr(args, "alpha", 16),
        "module_alpha_map": module_alpha_map,
        "zero_alpha_modules_excluded": True,
        "rank": target_r,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(metadata_path, "w") as f:
        json.dump(projection_metadata, f, indent=4)
        
    print(f"\n[SUCCESS] Transplanted LoRA saved to: {output_path}")
    print(f"Config saved to: {config_path}")
    print(f"Included modules (verified output keys): {target_modules}")
    print(f"You can now load this directory directly using peft.PeftModel.from_pretrained()")




def create_route_cli(args):
    warnings.warn("[DEPRECATED] 'route' command is legacy. Routing is now handled internally.", DeprecationWarning)
    print("\n[Layer 3: Semantic Router] Generating `.scalpel_route` manifest...")
    manager = ScalpelRouteManager(route_dir=args.output)
    
    # Mock transformation matrices for demonstration
    mock_R_matrix = [[1.0, 0.0], [0.0, 1.0]] 
    mock_s_factor = 1.05
    
    manager.create_route(
        source_id=args.source, 
        target_id=args.target, 
        domain=args.domain, 
        R_matrix=mock_R_matrix, 
        s_factor=mock_s_factor
    )
    print("[SUCCESS] Semantic route generated with strict SHA-256 validation.")

def hotswap_cli(args):
    warnings.warn("[DEPRECATED] 'hotswap' command is experimental and legacy.", DeprecationWarning)
    print("\n[Layer 4: Experimental VRAM Hot-Swap] Initializing...")
    
    # Create a mock live model for the CLI demonstration
    class MockLiveModel:
        def __init__(self):
            self._state = {args.layer: torch.zeros(10, 10)}
        def state_dict(self):
            return self._state
            
    live_model = MockLiveModel()
    api = VRAMHotSwapAPI(target_model=live_model)
    
    # Generate a mock task vector representing the "concept" to inject/unlearn
    task_vector = torch.ones(10, 10) * args.intensity
    
    print(f"Target Layer: {args.layer}")
    print(f"Action: {args.action.upper()} (Intensity: {args.intensity})")
    
    if args.action == "inject":
        api.inject_concept(task_vector, args.layer)
    elif args.action == "unlearn":
        api.remove_concept(task_vector, args.layer)
        
    # Guardrail Check
    current_norm = live_model.state_dict()[args.layer].norm().item()
    print(f"Post-operation L2 Norm of layer: {current_norm:.4f}")
    
    # Test Drift Monitor
    api.register_baseline(args.layer, 0.0) # Assume 0 is baseline
    api.monitor_drift(args.layer, current_norm)


def evaluate_cli(args):
    """Integrated evaluation entry point for v2.10."""
    print(f"\n[Layer 6: Evaluation] Running benchmark: {args.benchmark}")
    from neural_scalpel.core.evaluator import SQLCapabilityEvaluator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    print(f"Loading base model: {args.target}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    
    if args.adapter:
        print(f"Applying adapter: {args.adapter} (Merge: {getattr(args, 'merge_adapter', False)})")
        model = PeftModel.from_pretrained(model, args.adapter)
        if getattr(args, "merge_adapter", False):
            model = model.merge_and_unload()
    
    evaluator = SQLCapabilityEvaluator(model=model, tokenizer=tokenizer)
    
    # Load dataset based on benchmark name
    if args.benchmark == "sql_50":
        import sys
        benchmark_path = os.path.abspath("qwen2.5-0.5b-sql-structural-projection")
        if benchmark_path not in sys.path:
            sys.path.append(benchmark_path)
        try:
            from eval.sql_50_suite_definition import get_sql_50_suite
            suite = get_sql_50_suite()
            results = evaluator.evaluate_suite(suite)
        except ImportError as e:
            raise RuntimeError(f"Failed to load SQL-50 benchmark from {benchmark_path}. Ensure the directory exists. Error: {e}")
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # Prepare results
    results = {
        "stats": results["stats"],
        "results": results["results"],
        "eval_metadata": {
            "eval_dtype": args.dtype,
            "adapter_merge": bool(getattr(args, "merge_adapter", False)),
        }
    }
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SUCCESS] Evaluation results saved to {args.output}")
    
    # Print summary metrics
    stats = results["stats"]
    print(f"Accuracy: {stats['execution_accuracy']:.2%}")
    print(f"Syntax Valid: {stats['syntax_valid']}/{stats['total']}")

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel: VRAM Hot-Swap & Concept Projection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'project-adapter' (New Standard for v2.10)
    project_parser = subparsers.add_parser("project-adapter", help="Project a LoRA from one architecture to another")
    # Add an alias 'port' for backward compatibility
    subparsers.add_parser("port", help="[DEPRECATED] Use project-adapter instead")

    for p in [project_parser, subparsers.choices["port"]]:
        p.add_argument("--source", type=str, help="Path or HF ID to the source LoRA")
        p.add_argument("--source-adapter", type=str, help="Alias for --source")
        p.add_argument("--source-base", dest="source_base_model", type=str, default=None, 
                                 help="Explicit source base model path or ID")
        p.add_argument("--target", type=str, required=True, help="Path or HF ID to the target base model")
        p.add_argument("--output", type=str, required=True, help="Output directory for the translated LoRA")
        p.add_argument("--rank", type=int, default=16, help="Target rank (default: 16)")
        p.add_argument("--alpha", type=int, default=16, help="Target alpha (default: 16)")
        p.add_argument("--module-alpha-map", type=str, help="Comma-separated module=alpha mappings (e.g. 'q_proj=4,gate_proj=0.125')")
        p.add_argument("--piecewise-modules", type=str, help="Comma-separated modules for piecewise projection")
        p.add_argument("--include-modules", type=str, help="Comma-separated modules to include")
        # Add missing attributes for port_lora compatibility
        p.add_argument("--routing_path", type=str, default=None)
        p.add_argument("--calibrate", type=str, default=None)
        p.add_argument("--domain", type=str, default="general")
        p.add_argument("--piecewise-layers", type=str, default=None)
        p.add_argument("--piecewise-max-layers", type=int, default=None)

    # 'evaluate-projected' (New Standard for v2.10)
    eval_parser = subparsers.add_parser("evaluate-projected", help="Evaluate a projected adapter")
    eval_parser.add_argument("--target", type=str, required=True, help="Target base model ID")
    eval_parser.add_argument("--adapter", type=str, help="Path to projected adapter (optional for baseline)")
    eval_parser.add_argument("--benchmark", type=str, default="sql_50", help="Benchmark name")
    eval_parser.add_argument("--output", type=str, help="Path to save JSON results")
    eval_parser.add_argument("--dtype", type=str, default="float16", help="Precision for evaluation")
    eval_parser.add_argument("--merge-adapter", action="store_true", help="Merge adapter weights before evaluation (default: False)")

    # Legacy commands (keep for now but warn)
    subparsers.add_parser("route", help="[DEPRECATED] Create a route manifest")
    subparsers.add_parser("hotswap", help="[DEPRECATED] Live VRAM hotswap")
    subparsers.add_parser("diagnose", help="[DEPRECATED] Use diagnose-adapter instead")

    args = parser.parse_args()
    
    if args.command in ["port", "project-adapter"]:
        if not getattr(args, "source", None) and not getattr(args, "source_adapter", None):
            parser.error("Either --source or --source-adapter must be provided for projection.")
        # Map --source-adapter to --source if provided
        if hasattr(args, "source_adapter") and args.source_adapter:
            args.source = args.source_adapter
        port_lora(args)
    elif args.command == "evaluate-projected":
        evaluate_cli(args)
    elif args.command == "route":
        create_route_cli(args)
    elif args.command == "hotswap":
        hotswap_cli(args)
    elif args.command == "diagnose":
        diagnose_cli(args)

if __name__ == "__main__":
    main()
