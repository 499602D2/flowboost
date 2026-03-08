"""
FlowBoost example: local Docker-based optimization of a backward-facing step.

Demonstrates running the full optimization loop on macOS (or any system
without a native OpenFOAM install) using Docker via flowboost's Docker manager.
Uses the Pandas dataframe backend for post-processing.

Tutorial case: incompressibleFluid/pitzDaily (k-epsilon, blockMesh)
Objective:     minimize inlet pressure (proxy for pressure drop; the outlet
               pressure BC is fixed, so lower inlet p = lower pressure drop)
Search space:  inlet turbulent kinetic energy (k) and dissipation rate (epsilon)
"""

import warnings
from pathlib import Path

import coloredlogs

from flowboost import Case, Dictionary, Dimension, Manager, Objective, Session

warnings.filterwarnings("ignore", category=FutureWarning, module="ax.core.data")


def pressure_drop_objective(case: Case):
    """
    Read the area-averaged inlet pressure from postProcessing.

    The patchAverage function object writes areaAverage(p) at each timestep.
    A higher inlet pressure (for the same outlet BC) means a larger pressure
    drop, which we want to minimize.
    """
    fo_name = "patchAverage(patch=inlet,fields=(pU))"
    df = case.data.simple_function_object_reader(fo_name, backend="pandas")

    if df is None or df.empty:
        return None

    # Last timestep's area-averaged pressure at the inlet
    return df["areaAverage(p)"].iloc[-1]


def main():
    coloredlogs.install(level="INFO")

    data_dir = Path("flowboost_data")

    session = Session(
        name="pitzDaily",
        data_dir=data_dir,
        max_evaluations=20,
    )

    # Clone the tutorial case as template
    template_dir = data_dir / "pitzDaily_template"
    template = Case.from_tutorial("incompressibleFluid/pitzDaily", template_dir)

    # Add patchAverage function object for the inlet patch
    template.foam_get("patchAverage")
    patch_avg = template.dictionary("system/patchAverage")
    patch_avg.set("patch", "inlet")
    patch_avg.set("fields", "(p U)")

    # Shorten run time for the example (100 timesteps ≈ 5 sec in Docker on Apple M1)
    control_dict = template.dictionary("system/controlDict")
    control_dict.entry("endTime").set("0.01")

    session.attach_template_case(case=template)

    # Objective: minimize inlet pressure (proxy for pressure drop)
    objective = Objective(
        name="inlet_pressure",
        minimize=True,
        objective_function=pressure_drop_objective,
    )
    session.backend.set_objectives([objective])

    # Search space: inlet turbulence parameters (both are scalars)
    inlet_k = Dictionary.link("0/k").entry("boundaryField/inlet/value")
    k_dim = Dimension.range(
        name="inlet_k",
        link=inlet_k,
        lower=0.1,
        upper=1.5,
        log_scale=True,
    )

    inlet_eps = Dictionary.link("0/epsilon").entry("boundaryField/inlet/value")
    eps_dim = Dimension.range(
        name="inlet_epsilon",
        link=inlet_eps,
        lower=1.0,
        upper=50.0,
        log_scale=True,
    )

    session.backend.set_search_space([k_dim, eps_dim])

    # Use DockerLocal manager — runs simulations in per-job Docker containers
    if not session.job_manager:
        session.job_manager = Manager.create(
            scheduler="dockerlocal", wdir=session.data_dir, job_limit=2
        )

    session.job_manager.monitoring_interval = 1  # Fast polling in local execution
    session.backend.initialization_trials = 4  # Ax recommends 2 * dim_search_space
    session.clean_pending_cases()
    session.start()


if __name__ == "__main__":
    main()
