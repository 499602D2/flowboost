import logging

import pytest

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension


@pytest.fixture
def Ax_backend() -> AxBackend:
    # Test that AxBackend initializes correctly
    backend = AxBackend()

    # Add objective
    objective = Objective(
        name="Test Objective",
        minimize=True,
        objective_function=lambda x: 1
    )

    # Define something to modify
    dict_link = DictionaryLink(
        "constant/chemistryProperties").entry("tabulation/tolerance")

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="Test Dim",
        link=dict_link,
        lower=1e-5,
        upper=1e-1,
        log_scale=True)

    # Define something to modify
    dict_link = DictionaryLink(
        "constant/cloudProperties").entry("subModels/injectionModels/model1/SOI")

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="Test Dim",
        link=dict_link,
        lower=1e-5,
        upper=1e-1,
        log_scale=True)

    # Set search space + objectives
    backend.set_search_space([dim])
    backend.set_objectives([objective])
    return backend


def test_initialization(Ax_backend):
    Ax_backend.initialize()


def test_tell(Ax_backend, test_case, foam_in_env):
    # Evaluate an objective
    obj = Ax_backend.objectives[0]

    # Run evaluation for objective
    out = obj.batch_process(cases=[test_case])
    logging.info(f"Batch-processed: {out}")

    # Initialize backend
    Ax_backend.initialize()

    logging.info("Running tell()")
    Ax_backend.tell([test_case])
