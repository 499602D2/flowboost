from __future__ import annotations

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

def averageT(case: Case):
    """Calculate the lift to drag ratio from simulation results."""
    dataframe = case.data.simple_function_object_reader("fluid/cylinderT")

    if dataframe is None:
        return None
    return (
        dataframe.select(pl.last("areaAverage(T)")).item()
    )





if __name__ == "__main__":
    coloredlogs.install(level="INFO")

    # --- Session ---
    session = Session(
        name="coolingCylinder2D",
        data_dir=Path("flowboost_data"),
        clone_method="copy",
        max_evaluations=50,
        bo_concurrency=3
    )

    # --- Template case ---
    cooling_case = Case.from_tutorial(
        "multiRegion/CHT/coolingCylinder2D",
        Path(session.data_dir, "coolingCylinder2D_template"),
        method="copy",
    )
    cooling_case.dictionary("system/controlDict").entry("writeInterval").set("20")

    session.attach_template_case(case=cooling_case)

    # --- Objectives ---

    objective = Objective(
        name="averageT",
        minimize=False,
        objective_function=averageT,
    )



    # --- Search space ---
    DICT_FILE = "system/blockMeshDict"

    radius_dim = Dimension.range(
        name="cylinderRadius",
        link=Dictionary.link(DICT_FILE).entry("cylinderRadius"),
        lower=-0.001,
        upper=0.01,
    )



    session.configure_optimization(
        objectives=[objective],
        search_space=[radius_dim]
    )

    # --- Job manager ---
    session.job_manager = session.job_manager or Manager.create(
        scheduler="Local",
        wdir=session.data_dir,
        job_limit=1,
    )
    session.job_manager.monitoring_interval = 10

    # --- Run ---
    session.backend.initialization_trials = 10
    session.clean_pending_cases()
    session.start()

