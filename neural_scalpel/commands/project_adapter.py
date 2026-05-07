import argparse
import os
from neural_scalpel.cli.main import port_lora

def run_project(args):
    # Map CLI args to what port_lora expects
    class LegacyArgs:
        def __init__(self, source, target, output, rank, alpha, routing_path=None, delta_health=None, projection_mode="linear"):
            self.source = source
            self.target = target
            self.output = output
            self.rank = rank
            self.alpha = alpha
            self.routing_path = routing_path
            self.delta_health = delta_health
            self.projection_mode = projection_mode
            
    print(f"[Project] Starting {args.projection_mode} projection (rank={args.rank}, alpha={args.alpha})...")
    legacy_args = LegacyArgs(
        args.source_adapter, 
        args.target_model, 
        args.output,
        rank=args.rank,
        alpha=args.alpha,
        delta_health=getattr(args, "delta_health", None),
        projection_mode=getattr(args, "projection_mode", "linear")
    )
    port_lora(legacy_args)

def add_project_parser(subparsers):
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

    parser.set_defaults(func=run_project)
