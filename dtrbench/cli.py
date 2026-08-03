"""
Command-line interface for the DTRBench benchmarking framework.

This module provides a command-line interface (CLI) for running benchmarks and generating reports using the DTRBench framework. It allows users to execute benchmark runs based on a YAML configuration file and generate reports from the resulting data.
"""

import argparse

from dtrbench.analysis.report_results import report
from dtrbench.pipeline.runner import run


def main():

    parser = argparse.ArgumentParser(
        prog="dtrbench",
        description="Benchmarking framework for decision tree representations",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run benchmark from config file")

    run_parser.add_argument("--config", nargs="?", default="benchmark_config.yaml", type=str, help="Path to benchmark YAML configuration")

    run_parser.add_argument("--dataset", type=str, help="Name of dataset")

    run_parser.add_argument(
        "--mode",
        choices=["perturbation", "subforest", "resource"],
        help="Which benchmark to run",
    )

    report_parser = sub.add_parser(
        "report", help="Generate reports from benchmark results"
    )

    report_parser.add_argument("--config", nargs="?", default="report_config.yaml", type=str, help="Path to report YAML configuration")

    args = parser.parse_args()
    if args.command == "run":
        run(config_path=args.config, dataset=args.dataset, mode=args.mode)

    elif args.command == "report":
        report(config_path=args.config)
