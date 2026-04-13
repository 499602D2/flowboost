import json
import logging

import pytest

from flowboost.openfoam.case import Case
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
        name="test_objective",
        minimize=True,
        objective_function=lambda x: 1
    )

    # Define something to modify
    dict_link = DictionaryLink(
        "constant/chemistryProperties").entry("tabulation/tolerance")

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="test_dim",
        link=dict_link,
        lower=1e-5,
        upper=1e-1,
        log_scale=True)

    # Define something to modify
    dict_link = DictionaryLink(
        "constant/cloudProperties").entry("subModels/injectionModels/model1/SOI")

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="test_dim",
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
    outputs = obj.batch_evaluate(cases=[test_case])
    out = obj.batch_post_process(cases=[test_case], outputs=outputs)
    logging.info(f"Batch-processed: {out}")

    # Initialize backend
    Ax_backend.initialize()

    logging.info("Running tell()")
    Ax_backend.tell([test_case])


def _make_copied_case(data_dir, tmp_path, name: str) -> Case:
    case = Case.copy(data_dir, tmp_path / name)
    case.update_metadata(
        {"test_dim": {"value": 0.5}},
        entry_header="optimizer-suggestion",
    )
    return case


def _make_normalized_backend(case: Case) -> tuple[AxBackend, Objective]:
    objective = Objective(
        name="normalized_objective",
        minimize=True,
        objective_function=lambda _: 1.0,
        normalization_step="min-max",
    )
    outputs = objective.batch_evaluate([case])
    objective.batch_post_process([case], outputs)

    backend = AxBackend()
    backend.set_search_space(
        [
            Dimension.range(
                name="test_dim",
                link=DictionaryLink("constant/chemistryProperties").entry(
                    "tabulation/tolerance"
                ),
                lower=0.0,
                upper=1.0,
            )
        ]
    )
    backend.set_objectives([objective])
    return backend, objective


def test_tell_accepts_normalized_scalar_like_outputs(data_dir, tmp_path):
    case = _make_copied_case(data_dir, tmp_path, "normalized-case")
    backend, _ = _make_normalized_backend(case)

    backend.tell([case])

    trial_index = backend._trial_index_case_mapping[case]
    trial = backend.client.experiment.trials[trial_index]
    assert trial.status.is_completed


def test_prepare_for_acquisition_offload_serializes_normalized_outputs(
    data_dir, tmp_path
):
    case = _make_copied_case(data_dir, tmp_path, "offload-case")
    backend, _ = _make_normalized_backend(case)
    backend.offload_acquisition = True
    backend.initialize()

    _, data_snapshot = backend.prepare_for_acquisition_offload([case], [], tmp_path)
    snapshot_data = json.loads(data_snapshot.read_text())
    objective_value = snapshot_data["finished_cases"][case.name]["objectives"][
        "normalized_objective"
    ]

    assert type(objective_value) is float
    assert objective_value == 0.0
