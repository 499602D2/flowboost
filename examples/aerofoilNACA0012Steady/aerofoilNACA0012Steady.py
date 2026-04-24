"""
FlowBoost optimization example for NACA0012 airfoil.

This script demonstrates a multi-parameter Bayesian optimization workflow
for maximizing lift coefficient on a NACA0012 airfoil by varying angle of
attack and freestream velocity.
"""

import warnings
from pathlib import Path

import coloredlogs
import polars as pl

from flowboost import (
    Case,
    Constraint,
    Dictionary,
    Dimension,
    Manager,
    Objective,
    Session,
)

# Suppress FutureWarnings from Ax library
warnings.filterwarnings("ignore", category=FutureWarning, module="ax.core.data")


def lift_to_drag_ratio(case: Case):
    """Calculate the lift to drag ratio from simulation results."""
    dataframe = case.data.simple_function_object_reader("forceCoeffsCompressible")
    if dataframe is None:
        return None
    return (
        dataframe.select(pl.last("Cl")).item() / dataframe.select(pl.last("Cd")).item()
    )


def lift_objective(case: Case):
    """Calculate the lift coefficient from simulation results."""
    dataframe = case.data.simple_function_object_reader("forceCoeffsCompressible")
    if dataframe is None:
        return None
    return dataframe.select(pl.last("Cl")).item()


if __name__ == "__main__":
    coloredlogs.install(level="INFO")

    # --- Session ---
    session = Session(
        name="aerofoilNACA0012Steady",
        data_dir=Path("flowboost_data"),
        clone_method="copy",
        random_seed=0,
        target_value=60.8,  # Target lift-to-drag ratio
        max_evaluations=50,
    )

    # --- Template case ---
    naca_case = Case.from_tutorial(
        "fluid/aerofoilNACA0012Steady",
        Path(session.data_dir, "aerofoilNACA0012Steady_template"),
        method="copy",
    )
    naca_case.dictionary("system/controlDict").entry("writeInterval").set("5000")
    session.attach_template_case(case=naca_case)

    # --- Objectives ---

    objective = Objective(
        name="Lift_to_Drag",
        minimize=False,
        objective_function=lift_to_drag_ratio,
    )

    constraint = Constraint(
        name="Lift_Constraint",
        objective_function=lift_objective,
        gte=1,  # Ensure lift coefficient is above 1
    )

    # --- Search space ---
    DICT_FILE = "0/U"

    aoa_dim = Dimension.range(
        name="angleOfAttack",
        link=Dictionary.link(DICT_FILE).entry("angleOfAttack"),
        lower=-20,
        upper=40,
    )

    speed_dim = Dimension.choice(
        name="speed",
        link=Dictionary.link(DICT_FILE).entry("speed"),
        choices=[10, 15, 20],
    )

    session.configure_optimization(
        objectives=[objective],
        constraints=[constraint],
        search_space=[aoa_dim, speed_dim],
    )

    # --- Job manager ---
    session.job_manager = session.job_manager or Manager.create(
        scheduler="Local",
        wdir=session.data_dir,
        job_limit=2,
    )
    session.job_manager.monitoring_interval = 10

    # --- Run ---
    session.backend.initialization_trials = 8
    session.clean_pending_cases()
    session.start()
