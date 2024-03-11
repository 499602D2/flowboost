"""
A simple, barebones Python script meant to run the off-loaded acquisition
in an external environment. This program is expected to be invoked through
a shell script, and not run directly.

The acquisition job is expected to produce one json-file, with the results
of the acquisition.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.utilities.time import td_format

OFFLOAD_SCRIPT = "acquisition_offload.sh"


def run_acquisition_job(optimizer: str,
                        model_snapshot: str,
                        data_snapshot: str,
                        num_trials: int,
                        output_path: str) -> None:
    """
    Evaluates the acquisition job.

    Args:
        optimizer (str): Optimizer backend class name (implementing `Backend`)
        model_snapshot (str): Snapshot of model config that can be restored
        data_snapshot (str): Pre-processed objective outputs, case params
        num_trials (int): Number of new trials to generate
        output_path (str): Path to save the new trials in

    Raises:
        ValueError: If optimizer backend is not implemented
    """
    match optimizer:
        case "AxBackend":
            backend = AxBackend
        case _:
            raise ValueError(f"Unknown optimizer backend: {optimizer}")

    start = datetime.now(timezone.utc)
    backend.offloaded_acquisition(
        model_snapshot=Path(model_snapshot),
        data_snapshot=Path(data_snapshot),
        num_trials=num_trials,
        output_path=Path(output_path)
    )

    elapsed = td_format(start - datetime.now(timezone.utc))
    logging.info(f"Finished acquisition job, took {elapsed}")


if __name__ == "__main__":
    logging.basicConfig(filename="external_acquisition.log",
                        filemode="a",
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Invoke offloaded acquisition method on a cluster.")

    parser.add_argument("--optimizer", type=str, required=True,
                        help="Name of backend class to use (e.g. AxBackend)")
    parser.add_argument("--model_snapshot", type=str, required=True,
                        help="Path to the model snapshot file")
    parser.add_argument("--data_snapshot", type=str, required=True,
                        help="Path to the data snapshot file")
    parser.add_argument("--num_trials", type=int, required=True,
                        help="Number of new trials to generate")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the generated trials")

    # Parse the arguments
    args = parser.parse_args()

    # Invoke the method with provided arguments
    run_acquisition_job(optimizer=args.optimizer,
                        model_snapshot=args.model_snapshot,
                        data_snapshot=args.data_snapshot,
                        num_trials=args.num_trials,
                        output_path=args.output_path)
