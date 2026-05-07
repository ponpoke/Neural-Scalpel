import argparse
import os
import json
from neural_scalpel.cli.main import port_lora
from neural_scalpel.core.adapters import AdaptiveScalingConfig

def run_project(args):
    # Load scaling config if provided
    scaling_config = None
    if hasattr(args, "adaptive_scaling_config") and args.adaptive_scaling_config:
        if os.path.exists(args.adaptive_scaling_config):
            with open(args.adaptive_scaling_config, "r") as f:
                config_data = json.load(f)
                scaling_config = AdaptiveScalingConfig(**config_data)

    delta_health = None
    if hasattr(args, "delta_health") and args.delta_health:
        if os.path.exists(args.delta_health):
            with open(args.delta_health, "r") as f:
                report_data = json.load(f)
                # If it's a full report, extract the health gate
                if "delta_health_gate" in report_data:
                    from types import SimpleNamespace
                    delta_health = SimpleNamespace(**report_data["delta_health_gate"])
                else:
                    from types import SimpleNamespace
                    delta_health = SimpleNamespace(**report_data)

    class LegacyArgs:
        def __init__(self, source, target, output, rank, alpha, routing_path=None, 
                     delta_health=None, projection_mode="linear", scaling_config=None,
                     piecewise_modules=None, piecewise_layers=None, piecewise_max_layers=None,
                     source_base_model=None, allow_dummy_fallback=False, include_modules=None,
                     module_alpha_map=None):
            self.source = source
            self.target = target
            self.output = output
            self.rank = rank
            self.alpha = alpha
            self.routing_path = routing_path
            self.delta_health = delta_health
            self.projection_mode = projection_mode
            self.scaling_config = scaling_config
            self.piecewise_modules = piecewise_modules
            self.piecewise_layers = piecewise_layers
            self.piecewise_max_layers = piecewise_max_layers
            self.source_base_model = source_base_model
            self.allow_dummy_fallback = allow_dummy_fallback
            self.include_modules = include_modules
            self.module_alpha_map = module_alpha_map
            
    if not getattr(args, "source_base_model", None):
        print(f"[WARNING] --source-base was not provided. Falling back to source adapter '{args.source_adapter}' for config resolution. This may be unreliable.")

    print(f"[Project] Starting {args.projection_mode} projection (rank={args.rank}, alpha={args.alpha})...")
    # Parse list-like arguments
    p_modules = args.piecewise_modules.split(",") if getattr(args, "piecewise_modules", None) else None
    p_layers = [int(x) for x in args.piecewise_layers.split(",")] if getattr(args, "piecewise_layers", None) else None

    legacy_args = LegacyArgs(
        args.source_adapter, 
        args.target_model, 
        args.output,
        rank=args.rank,
        alpha=args.alpha,
        delta_health=delta_health,
        projection_mode=getattr(args, "projection_mode", "linear"),
        scaling_config=scaling_config,
        piecewise_modules=p_modules,
        piecewise_layers=p_layers,
        piecewise_max_layers=getattr(args, "piecewise_max_layers", None),
        source_base_model=getattr(args, "source_base_model", None),
        allow_dummy_fallback=getattr(args, "allow_dummy_fallback", False),
        include_modules=getattr(args, "include_modules", None),
        module_alpha_map=getattr(args, "module_alpha_map", None)
    )
    port_lora(legacy_args)

def add_project_adapter_parser(subparsers):
    parser = subparsers.add_parser(
        "project-adapter",
        help="Perform structural projection of a LoRA adapter."
    )
    parser.add_argument("--source-adapter", required=True,
                        help="Path or ID of the source adapter")
    parser.add_argument("--source-base", dest="source_base_model", required=False,
                        help="Source base model used by the adapter")
    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model")
    parser.add_argument("--output", required=True,
                        help="Path to save the projected adapter")
    parser.add_argument("--rank", type=int, default=16,
                        help="Target rank (default: 16)")
    parser.add_argument("--alpha", type=int, default=16,
                        help="Target alpha (default: 16)")
    parser.add_argument("--projection-mode", choices=["linear", "piecewise", "kernel", "jacobian"],
                        default="linear", help="Projection strategy (default: linear)")
    parser.add_argument("--adaptive-scaling-config", 
                        help="Path to adaptive scaling JSON config")
    parser.add_argument("--delta-health",
                        help="Path to diagnostic report JSON for adaptive scaling")
    parser.add_argument("--allow-dummy-fallback", action="store_true",
                        help="Allow generating dummy weights if physical files are missing (CI only)")
    
    # Piecewise constraints
    parser.add_argument("--piecewise-modules", type=str,
                        help="Comma-separated modules for piecewise (e.g. 'up_proj,down_proj')")
    parser.add_argument("--piecewise-layers", type=str,
                        help="Comma-separated layer indices for piecewise")
    parser.add_argument("--piecewise-max-layers", type=int,
                        help="Maximum number of layers to use piecewise projection")
    parser.add_argument("--include-modules", type=str,
                        help="Comma-separated modules to include (e.g. 'q_proj,v_proj')")
    parser.add_argument("--module-alpha-map", type=str,
                        help="Comma-separated module=alpha mappings (e.g. 'q_proj=8,down_proj=1')")

    parser.set_defaults(func=run_project)
