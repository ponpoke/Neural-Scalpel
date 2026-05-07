import argparse
from neural_scalpel.cli.main import port_lora

def run_project(args):
    # Map CLI args to what port_lora expects
    class LegacyArgs:
        def __init__(self, source, target, output, rank, alpha, routing_path=None):
            self.source = source
            self.target = target
            self.output = output
            self.rank = rank
            self.alpha = alpha
            self.routing_path = routing_path
            
    print(f"[Project] Starting experimental projection (rank={args.rank}, alpha={args.alpha})...")
    legacy_args = LegacyArgs(
        args.source_adapter, 
        args.target_model, 
        args.output,
        rank=args.rank,
        alpha=args.alpha
    )
    port_lora(legacy_args)

def add_project_adapter_parser(subparsers):
    parser = subparsers.add_parser(
        "project-adapter",
        help="Project source adapter weights into target model architecture."
    )

    parser.add_argument("--source-base", dest="source_base_model", required=True,
                        help="Source base model (for info)")
    parser.add_argument("--source-adapter", required=True,
                        help="Path to source adapter")
    parser.add_argument("--target", dest="target_model", required=True,
                        help="Target base model")
    parser.add_argument("--output", required=True,
                        help="Output path for projected adapter")
    parser.add_argument("--rank", type=int, default=16,
                        help="Target rank (default: 16)")
    parser.add_argument("--alpha", type=int, default=16,
                        help="Target alpha (default: 16)")

    parser.set_defaults(func=run_project)
