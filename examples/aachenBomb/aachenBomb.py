from pathlib import Path

import coloredlogs
import polars as pl

from flowboost.manager.manager import Manager
from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import Dictionary
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension
from flowboost.session.session import Session


def max_temp_objective(case: Case):
    # Assume case has as function object computing average pressure and temp
    my_func_obj = "averagePT"

    # Load data from the function object into a dataframe
    # The backend can be either Polars or Pandas: here, we are using Polars.
    df = case.data.simple_function_object_reader(my_func_obj)

    if df is None:
        return None

    # Get peak temperature from function object output (as a Polars DF)
    max_T = df.select(pl.max("volAverage(T)")).item()

    # Want to use Pandas instead?
    # df = df.to_pandas()

    # Objective functions _must_ return a value! Uncaught errors will terminate
    # the optimization, while `None` values will be translated to a failed
    # case.
    #
    # You can add normalization steps and other arbitrary post-processing
    # tasks, operating on the array of outputs of an objective function, to the
    # Objective() itself.
    return max_T


if __name__ == "__main__":
    # Custom logging
    coloredlogs.install(level="INFO")

    # Create optimization session with Ax backend
    data_dir = Path("flowboost_data")

    # FlowBoost session
    session = Session(name="aachenBomb", data_dir=data_dir)

    # Define a template case
    case_dir = Path(data_dir, "aachenBomb_template")
    aachen_case = Case.from_tutorial("multicomponentFluid/aachenBomb", case_dir)

    session.attach_template_case(case=aachen_case)

    # Define an optimization objective. Objectives are simply user-defined
    # functions, which accept one or more arguments. By default, it should
    # only accept one parameter: the case that the objective value is being
    # computed for, of type `flowboost.openfoam.foam_case.Case`
    #
    # If you require arbitrary data during computation, you can not only do
    # it within the function, but you can also pass data to the function using
    # keyword-arguments (kwargs).
    #
    # If you wanted to pass a Case containing baseline data to the function,
    # you could simply add the argument
    # `kwargs={"my_baseline_case": baseline_case_object}`.

    # Objective, which calls max_temp_objective() for every case during
    # optimization. The normalization step is applied to the array of outputs
    # produced by the objective.
    objective = Objective(
        name="Peak temperature",
        minimize=True,
        objective_function=max_temp_objective,
        normalization_step="yeo-johnson",
    )

    # Add objective to the session's optimizer
    session.backend.set_objectives([objective])

    # Next, let's define the search space for the problem. This is done by
    # first defining the configuration entries to modify in your template case,
    # as well as dimensions that determine the range of values that can be
    # produced during optimization.
    #
    # Start off by creating a dictionary link: this is relative, and is
    # converted to a Dictionary _reader_ on demand. This determines where
    # the search space values get written.
    #
    # This entry should already exist in your template case.

    # Dictionary link for entry to modify
    dict_file = "constant/cloudProperties"
    entry_path = "subModels/injectionModels/model1/massTotal"
    entry_link = Dictionary.link(dict_file).entry(entry_path)

    # Next, define the search space dimension for tolerance and link it to
    # the configuration dictionary.
    mass_dim = Dimension.range(
        name="Injected mass", link=entry_link, lower=1e-7, upper=1e-4, log_scale=False
    )

    # Dictionary link for entry to modify
    dict_file = "constant/cloudProperties"
    entry_path = "subModels/injectionModels/model1/SOI"
    entry_link = Dictionary.link(dict_file).entry(entry_path)

    # Next, define the search space dimension for tolerance and link it to
    # the configuration dictionary.
    soi_dim = Dimension.range(
        name="SOI", link=entry_link, lower=0.0, upper=1e-3, log_scale=False
    )

    # Remember to add the dimension(s) to your Session
    session.backend.set_search_space([mass_dim, soi_dim])

    # Finally, if you are working in an HPC enviroment, with a job manager such
    # as Slurm or Sun Grid Engine (SGE), you can define a JobManager to
    # automatically manage the optimization jobs.
    #
    # If you do not define a job manager, flowboost will generate the next cases
    # for you to evaluate, and exit once it has done so. This leaves you with a
    # set of case directories, ready to use.

    # Here, we will use SGE and be nice and only hog three compute nodes at
    # once (maximum).
    if not session.job_manager:
        # Avoid reloading twice (safe, but unnecessary)
        session.job_manager = Manager.create(
            scheduler="Local", wdir=session.data_dir, job_limit=1
        )

    # Reduce monitoring interval for demonstration purposes
    session.job_manager.monitoring_interval = 5

    # Enable model update + acquisition offloading to cluster env
    # session.backend.offload_acquisition = True

    # Ready to start!
    # The startup goes as follows:
    # 1. A new data directory is created and the session starts
    # 2. The first batch of simulations is generated
    # 3. Simulations are submitted using the job manager
    # 4. The job manager does lightweight monitoring of the jobs until one
    #    of them finishes.
    # 5. Data is ingressed and processed for the finished case.
    session.start()
