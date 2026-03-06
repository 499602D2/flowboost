"""
FlowBoost optimization example for NACA0012 airfoil.

This script demonstrates a multi-parameter Bayesian optimization workflow
for maximizing lift coefficient on a NACA0012 airfoil by varying angle of
attack and freestream velocity.
"""
from pathlib import Path
import warnings

import coloredlogs
import polars as pl

from flowboost.manager.manager import Manager
from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import Dictionary
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension
from flowboost.session.session import Session

# Suppress FutureWarnings from Ax library
warnings.filterwarnings('ignore', category=FutureWarning, module='ax.core.data')


def max_lift_drag_objective(case: Case):
    """
    Calculate the lift coefficient from simulation results.

    Args:
        case: OpenFOAM case to extract lift coefficient from

    Returns:
        float: Last lift coefficient value, or None if data unavailable
    """
    my_func_obj = "forceCoeffsCompressible"
    dataframe = case.data.simple_function_object_reader(my_func_obj)

    if dataframe is None:
        return None

    last_cl = dataframe.select(pl.last("Cl")).item()
    last_cd = dataframe.select(pl.last("Cd")).item()

    return last_cl/last_cd


if __name__ == "__main__":
    coloredlogs.install(level="INFO")

    data_dir = Path("flowboost_data")

    session = Session(
        name="aerofoilNACA0012Steady",
        data_dir=data_dir,
        clone_method="copy",
        max_evaluations=50
    )


    # Define a template case
    case_dir = Path(data_dir, "aerofoilNACA0012Steady_template")
    naca_case = Case.from_tutorial("fluid/aerofoilNACA0012Steady", case_dir, method="copy")

    # Set writeInterval to 5000 to avoid writing excessive fields
    control_dict = naca_case.dictionary("system/controlDict")
    control_dict.entry("writeInterval").set("5000")

    # Attach template case to session
    session.attach_template_case(case=naca_case)

    # Define optimization objective
    objective = Objective(
        name="L/D",
        minimize=False,
        objective_function=max_lift_drag_objective,
        normalization_step="yeo-johnson",
    )

    session.backend.set_objectives([objective])

    # Define search space dimensions
    DICT_FILE = "0/U"
    # Angle of attack dimension
    ENTRY_PATH_AOA = "angleOfAttack"
    entry_link_aoa = Dictionary.link(DICT_FILE).entry(ENTRY_PATH_AOA)
    aoa_dim = Dimension.range(
        name="angleOfAttack",
        link=entry_link_aoa,
        lower=-20,
        upper=40,
        log_scale=False
    )

    # Speed dimension
    ENTRY_PATH_AOA = "speed"
    entry_link_speed = Dictionary.link(DICT_FILE).entry(ENTRY_PATH_AOA)
    speed_dim = Dimension.choice(
        name="speed",
        link=entry_link_speed,
        choices=[10, 15, 20]
    )

    session.backend.set_search_space([aoa_dim, speed_dim])

    # Configure job manager
    scheduler = "Local"  # Change to "sge" for cluster

    if not session.job_manager:
        session.job_manager = Manager.create(
            scheduler=scheduler,
            wdir=session.data_dir,
            job_limit=5
        )

    session.job_manager.monitoring_interval = 10
    session.backend.initialization_trials = 4
    session.clean_pending_cases()
    session.start()
