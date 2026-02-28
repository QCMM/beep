"""
BEEP CLI — single JSON-config entry point for all workflows.

Usage:
    beep --config input.json

The JSON file must contain a "workflow" key that selects which workflow to run.
Valid workflow values: sampling, be_hess, extract, pre_exp, geom_benchmark, energy_benchmark
"""
import argparse
import json
import sys
import logging
from pathlib import Path

from .models import (
    SamplingConfig,
    BeHessConfig,
    ExtractConfig,
    PreExpConfig,
    GeomBenchmarkConfig,
    EnergyBenchmarkConfig,
)
from .adapters.qcfractal_adapter import connect
from .core.logging_utils import setup_logging

WORKFLOW_MODELS = {
    "sampling": SamplingConfig,
    "be_hess": BeHessConfig,
    "extract": ExtractConfig,
    "pre_exp": PreExpConfig,
    "geom_benchmark": GeomBenchmarkConfig,
    "energy_benchmark": EnergyBenchmarkConfig,
}

WELCOME_BANNER = """
---------------------------------------------------------------
  BEEP — Binding Energy Evaluation Platform
---------------------------------------------------------------
"""


def main():
    parser = argparse.ArgumentParser(
        description="BEEP — Binding Energy Evaluation Platform CLI"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON configuration file",
    )
    args = parser.parse_args()

    # Load and parse config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(config_path.read_text())
    workflow = raw.get("workflow")

    if workflow not in WORKFLOW_MODELS:
        print(
            f"Error: unknown workflow '{workflow}'. "
            f"Valid options: {', '.join(WORKFLOW_MODELS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate with the appropriate Pydantic model
    config = WORKFLOW_MODELS[workflow](**raw)

    # Set up logging
    logger = setup_logging(f"beep_{workflow}", workflow)
    logger.info(WELCOME_BANNER)
    logger.info(f"Running workflow: {workflow}")

    # Connect to QCFractal server
    client = connect(
        address=config.server.address,
        username=config.server.username,
        password=config.server.password,
        verify=config.server.verify,
    )

    # Import and dispatch to the appropriate workflow
    if workflow == "sampling":
        from .workflows.sampling import run
    elif workflow == "be_hess":
        from .workflows.be_hess import run
    elif workflow == "extract":
        from .workflows.extract import run
    elif workflow == "pre_exp":
        from .workflows.pre_exp import run
    elif workflow == "geom_benchmark":
        from .workflows.geom_benchmark import run
    elif workflow == "energy_benchmark":
        from .workflows.energy_benchmark import run

    run(config, client)

    logger.info(f"\nWorkflow '{workflow}' finished successfully.")


if __name__ == "__main__":
    main()
