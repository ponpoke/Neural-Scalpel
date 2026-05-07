import argparse
import sys
from neural_scalpel.commands.diagnose_adapter import add_diagnose_adapter_parser
from neural_scalpel.commands.project_adapter import add_project_adapter_parser
from neural_scalpel.commands.evaluate_projected import add_evaluate_projected_parser
from neural_scalpel.commands.safe_project import add_safe_project_parser

def main():
    parser = argparse.ArgumentParser(
        prog="neural-scalpel",
        description="Neural-Scalpel: Multi-Stage Adapter Transfer & Diagnostic Toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register subcommands
    add_diagnose_adapter_parser(subparsers)
    add_project_adapter_parser(subparsers)
    add_evaluate_projected_parser(subparsers)
    add_safe_project_parser(subparsers)

    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
