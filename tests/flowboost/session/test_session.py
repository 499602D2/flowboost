import json
import logging
from pathlib import Path

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension
from flowboost.session.session import Session


@pytest.fixture
def test_session(tmp_path):
    session = Session(
        name="My cool optimization campaign",
        data_dir=tmp_path / "flowboost_test",
    )
    yield session
    session._delete_all_data()


def test_persistence(test_session: Session):
    # Try persisting session
    test_session.persist()
    print("Persisted session")


def test_restore(test_session: Session):
    test_session.persist()
    test_session.restore()
    logging.info(json.dumps(test_session.state(), indent=4))
    print("Restored session")


def test_session_random_seed_is_applied_to_backend(tmp_path):
    session = Session(
        name="Seeded session",
        data_dir=tmp_path / "seeded_session",
        random_seed=123,
    )

    assert session.random_seed == 123
    assert hasattr(session.backend, "random_seed")
    assert session.backend.random_seed == 123


def test_session_restore_reapplies_random_seed_to_backend(tmp_path):
    data_dir = tmp_path / "restored_seed_session"
    created = Session(
        name="Restored seed session",
        data_dir=data_dir,
        random_seed=321,
    )
    created.persist()

    restored = Session(
        name="ignored",
        data_dir=data_dir,
        random_seed=None,
    )

    assert restored.random_seed == 321
    assert hasattr(restored.backend, "random_seed")
    assert restored.backend.random_seed == 321


def test_incorrect_startup():
    # Objective missing linked entry
    # Test missing dictionary
    # Test only dictionary provided (no entry)
    # Test missing dictionary entry
    pass


def test_simple_blank_start(foam_in_env, tmp_path, monkeypatch):
    # Add objective function
    objective = Objective(
        name="test_objective", minimize=True, objective_function=lambda x: 1
    )

    # Define what to modify
    dict_link = DictionaryLink("constant/chemistryProperties").entry("odeCoeffs/eps")

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="test_dim", link=dict_link, lower=1e-3, upper=1e-1, log_scale=True
    )

    session = Session(
        name="My cool optimization campaign",
        data_dir=tmp_path / "test_campaign_flowboost",
        dataframe_format="polars",
    )

    # Set search space + objectives
    session.backend.set_search_space([dim])
    session.backend.set_objectives([objective])

    # Template
    aachen_case = Case.from_tutorial(
        "multicomponentFluid/aachenBomb", Path(session.data_dir, "aachenBomb_template")
    )

    session.attach_template_case(aachen_case)

    # session.start() prompts for number of cases when no job manager is set
    monkeypatch.setattr("builtins.input", lambda _: "2")
    new_cases = session.start()

    assert new_cases, "Optimizer did not provide new cases"

    # Evaluate objectives
    output = session.backend.batch_process(new_cases)
    print("Objective function evaluated")
    for i, out in enumerate(output, 1):
        print(f"[{i}] {out}")

    session._delete_all_data()
