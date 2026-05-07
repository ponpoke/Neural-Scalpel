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

    # Open the incremental writer
    output_bridge.open_writer(output_path)
    
    try:
        # [v2.9.1 Hardening] Use the resolved adapter path
        source_path = source_adapter_path
        
        print(f"[IO] Starting streaming iterator from {source_path}...")
        for key, tensor in source_bridge.iter_layers(source_path):
            print(f"  Surgery on {key}...")
            new_key = adapter.map_key(key)
            new_tensor = adapter.project_tensor(key, tensor)
            
            if new_tensor is not None:
                if isinstance(new_tensor, dict):
                    # For pair-aware projection returning multiple tensors
                    for k, v in new_tensor.items():
                        output_bridge.write_layer(k, v)
                else:
                    output_bridge.write_layer(new_key, new_tensor)

            # Manual memory reclamation
            del tensor
            del new_tensor
        
        adapter.finalize()
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
            new_key = adapter.map_key(key)
            new_tensor = adapter.project_tensor(key, tensor)
            if new_tensor is not None:
                if isinstance(new_tensor, dict):
                    for k, v in new_tensor.items():
                        output_bridge.write_layer(k, v)
                else:
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
    adapter_config = {
        "peft_type": "LORA",
        "r": target_r,
        "lora_alpha": getattr(args, "alpha", target_r * 2),
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "base_model_name_or_path": args.target
    }
    
    config_path = os.path.join(output_dir if output_dir else ".", "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
        
    print(f"\n[SUCCESS] Transplanted LoRA saved to: {output_path}")
    print(f"Config saved to: {config_path}")
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


def diagnose_cli(args):
    warnings.warn("[DEPRECATED] 'diagnose' command is legacy. Use 'diagnose-adapter' instead.", DeprecationWarning)
    print("\n[Layer 5: Diagnostic Suite] Generating Portability Feasibility Report...")
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    try:
        from verification_demo.run_migration_diagnostics import run_diagnostics
        run_diagnostics(args)
    except ImportError as e:
        print(f"Error loading diagnostics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Neural-Scalpel: VRAM Hot-Swap & Concept Projection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'port' command (Layer 2)
    port_parser = subparsers.add_parser("port", help="Port a LoRA from one architecture to another")
    port_parser.add_argument("--source", type=str, required=True, help="Path or HF ID to the source LoRA")
    port_parser.add_argument("--source-base", dest="source_base_model", type=str, default=None, 
                             help="Explicit source base model path or ID")
    port_parser.add_argument("--target", type=str, required=True, help="Path or HF ID to the target base model")
    port_parser.add_argument("--routing_path", type=str, default=None, help="Optional path to a .pt or .scalpel_route WDR matrix")
    port_parser.add_argument("--calibrate", type=str, default=None, help="Path to a calibration dataset (.pt activations) for AWQ re-calibration")
    port_parser.add_argument("--domain", type=str, default="general", help="Domain semantic anchor (e.g., coding, medical)")
    port_parser.add_argument("--output", type=str, required=True, help="Output directory for the translated LoRA")
    port_parser.add_argument("--rank", type=int, default=16, help="Target rank (default: 16)")
    port_parser.add_argument("--alpha", type=int, default=16, help="Target alpha (default: 16)")
    port_parser.add_argument("--piecewise-modules", type=str, help="Comma-separated modules for piecewise projection")
    port_parser.add_argument("--piecewise-layers", type=str, help="Comma-separated layer indices for piecewise projection")
    port_parser.add_argument("--piecewise-max-layers", type=int, help="Maximum number of layers to use piecewise projection")
    
    # 'route' command (Layer 3)
    route_parser = subparsers.add_parser("route", help="Create a domain-specific .scalpel_route mapping file")
    route_parser.add_argument("--source", type=str, required=True, help="Path or HF ID to the source architecture")
    route_parser.add_argument("--target", type=str, required=True, help="Path or HF ID to the target architecture")
    route_parser.add_argument("--domain", type=str, required=True, help="Domain semantic anchor (e.g., coding, medical)")
    route_parser.add_argument("--output", type=str, default="./routes", help="Directory to save the route file")

    # 'hotswap' command (Layer 4)
    hotswap_parser = subparsers.add_parser("hotswap", help="Experimentally inject or unlearn concepts in live VRAM")
    hotswap_parser.add_argument("--action", type=str, choices=["inject", "unlearn"], required=True, help="Action to perform")
    hotswap_parser.add_argument("--layer", type=str, default="model.layers.0.self_attn.q_proj.weight", help="Target layer in the live model")
    hotswap_parser.add_argument("--intensity", type=float, default=1.0, help="Intensity scalar for the task vector")

    # 'diagnose' command (Layer 5)
    diagnose_parser = subparsers.add_parser("diagnose", help="Run a portability and risk diagnostic on a LoRA mapping")
    diagnose_parser.add_argument("--source", type=str, required=True, help="Path or HF ID to the source LoRA")
    diagnose_parser.add_argument("--target", type=str, required=True, help="Path or HF ID to the target base model")
    diagnose_parser.add_argument("--calibrate", type=str, default=None, help="Path to calibration data (required for LLMs)")
    diagnose_parser.add_argument("--eval", type=str, default=None, help="Evaluation config/data for empirical benchmarks")
    diagnose_parser.add_argument("--ablation", type=str, default="none", help="Ablation mode to run (e.g., 'all')")
    diagnose_parser.add_argument("--output", type=str, default="./reports", help="Output directory for the report")

    args = parser.parse_args()
    
    if args.command == "port":
        port_lora(args)
    elif args.command == "route":
        create_route_cli(args)
    elif args.command == "hotswap":
        hotswap_cli(args)
    elif args.command == "diagnose":
        diagnose_cli(args)

if __name__ == "__main__":
    main()
