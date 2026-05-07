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

    # Map CLI args to what port_lora expects
    class LegacyArgs:
        def __init__(self, source, target, output, rank, alpha, routing_path=None, 
                     delta_health=None, projection_mode="linear", scaling_config=None):
            self.source = source
            self.target = target
            self.output = output
            self.rank = rank
            self.alpha = alpha
            self.routing_path = routing_path
            self.delta_health = delta_health
            self.projection_mode = projection_mode
            self.scaling_config = scaling_config
            
    print(f"[Project] Starting {args.projection_mode} projection (rank={args.rank}, alpha={args.alpha})...")
    legacy_args = LegacyArgs(
        args.source_adapter, 
        args.target_model, 
        args.output,
        rank=args.rank,
        alpha=args.alpha,
        delta_health=getattr(args, "delta_health", None),
        projection_mode=getattr(args, "projection_mode", "linear"),
        scaling_config=scaling_config
    )
    port_lora(legacy_args)

def add_project_adapter_parser(subparsers):
    parser = subparsers.add_parser(
        "project-adapter",
        help="Perform structural projection of a LoRA adapter."
    )
    parser.add_argument("--source-adapter", required=True,
                        help="Path or ID of the source adapter")
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

    parser.set_defaults(func=run_project)
