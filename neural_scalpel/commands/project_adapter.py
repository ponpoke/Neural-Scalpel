import argparse
import os
from neural_scalpel.cli.main import port_lora

def run_project(args):
    """Wrapper for project-adapter, delegating core logic to hardened main.port_lora."""
    # Ensure source is correctly mapped for port_lora compatibility
    if hasattr(args, "source_adapter") and args.source_adapter and not getattr(args, "source", None):
        args.source = args.source_adapter
    
    # Normalize piecewise args for port_lora if provided as strings
    if hasattr(args, "piecewise_layers") and isinstance(args.piecewise_layers, str):
        try:
            args.piecewise_layers = [int(x.strip()) for x in args.piecewise_layers.split(",")]
        except Exception:
            pass

    return port_lora(args)

def add_project_adapter_parser(subparsers):
    parser = subparsers.add_parser(
        "project-adapter",
        help="Perform structural projection of a LoRA adapter (Hardened v2.11+)."
    )
    # Support both --source and --source-adapter for maximum compatibility
    parser.add_argument("--source", help="Path or ID of the source adapter")
    parser.add_argument("--source-adapter", help="Alias for --source")
    
    parser.add_argument("--source-base", dest="source_base_model", required=False,
                        help="Source base model used by the adapter")
    parser.add_argument("--target", dest="target", required=True,
                        help="Target base model path or ID")
    parser.add_argument("--output", required=True,
                        help="Path to save the projected adapter")
    
    parser.add_argument("--rank", type=int, default=16, help="Target rank (default: 16)")
    parser.add_argument("--alpha", type=int, default=16, help="Target alpha (default: 16)")
    
    parser.add_argument("--projection-mode", choices=["linear", "piecewise", "kernel", "jacobian"],
                        default="linear", help="Projection strategy (default: linear)")
    
    parser.add_argument("--include-modules", type=str,
                        help="Comma-separated modules to include (e.g. 'q_proj,v_proj')")
    parser.add_argument("--module-alpha-map", type=str,
                        help="Comma-separated module=alpha mappings (e.g. 'q_proj=8,down_proj=1')")
    
    parser.add_argument("--allow-dummy-fallback", action="store_true",
                        help="Allow generating dummy weights if physical files are missing (CI/Test only)")
    
    # Piecewise constraints
    parser.add_argument("--piecewise-modules", type=str,
                        help="Comma-separated modules for piecewise (e.g. 'up_proj,down_proj')")
    parser.add_argument("--piecewise-layers", type=str,
                        help="Comma-separated layer indices for piecewise")
    parser.add_argument("--piecewise-max-layers", type=int,
                        help="Maximum number of layers to use piecewise projection")
    
    # Metadata for legacy compatibility in port_lora
    parser.add_argument("--routing_path", type=str, default=None)
    parser.add_argument("--delta_health", type=str, default=None)

    parser.set_defaults(func=run_project)
