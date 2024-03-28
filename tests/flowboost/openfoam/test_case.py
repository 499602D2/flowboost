import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest
import tomlkit

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import Dictionary


@pytest.fixture
def new_case_dir(tmp_path):
    dir_path = Path(tmp_path / "newCaseDir")
    dir_path.mkdir()
    yield dir_path


@pytest.fixture
def tutorial_case(foam_in_env, tmp_path):
    case_path = tmp_path / "tutorialCase"
    case = Case.from_tutorial("multicomponentFluid/aachenBomb", case_path)
    yield case
    case._delete_all_data()


def test_dictionary_entries(foam_in_env, tutorial_case):
    # Test reading a dictionary and verifying an entry
    chemistry_props = tutorial_case.dictionary("constant/chemistryProperties")
    assert chemistry_props.entry("odeCoeffs/solver") == "seulex"

    cloud_props = tutorial_case.dictionary("constant/cloudProperties")
    assert cloud_props.entry("subModels/injectionModels/model1/massTotal") == 6.0e-6


def test_dictionary_link(foam_in_env, tutorial_case):
    # Test Dictionary Links
    link = Dictionary.link("constant/cloudProperties")
    reader = link.reader(tutorial_case.path)
    assert reader
    assert reader.entry("type") == "sprayCloud"

    # Test case-based link access
    reader = tutorial_case.dictionary(link)
    assert reader.entry("type") == "sprayCloud"


def test_case_clone_and_removal(foam_in_env, tutorial_case, new_case_dir):
    # Copy tutorial case to a new folder and test removal
    new_case = tutorial_case.clone(clone_to=new_case_dir)
    assert new_case.path.exists()

    # Test safe deletion
    assert new_case._delete_all_data(), "Safe removal failed"


def test_case_safe_deletion(foam_in_env, tmp_path):
    # Test case removal safety enforcement
    empty_dir = tmp_path / str(Path(str(uuid4())))
    empty_dir.mkdir(exist_ok=False)
    faux_case = Case(empty_dir)

    # Should not delete non-OpenFOAM directories unless forced
    assert not faux_case._delete_all_data(), "Unsafe removal detected!"
    assert faux_case._delete_all_data(
        skip_familiarity_checks=True
    ), "Forced unsafe removal failed"


def test_property_decorated_data(new_case_dir):
    p = Path(new_case_dir)
    p.mkdir()

    c = Case(p)
    assert c.data is not None


def test_persistence(new_case_dir):
    c = Case(new_case_dir)
    c.persist_to_file()
    logging.info("\n" + json.dumps(c.state(), indent=4))


def test_metadata(new_case_dir):
    c = Case(new_case_dir)
    c.persist_to_file()
    logging.info("\n" + json.dumps(c.state(), indent=4))

    metadata = {"tolerance": 1e-3, "nodes": 1000}
    logging.info(f"Adding metadata: {metadata}")
    c.update_metadata(metadata, "optimizer-suggestions")

    toml_d = c.read_metadata()
    assert toml_d, "Loading metadata failed"

    logging.info(f"Read metadata:\n{tomlkit.dumps(toml_d)}")

    # Test modifying
    metadata = {"tolerance": 10e-2, "nodes": 5}
    logging.info(f"Updating metadata: {metadata}")
    c.update_metadata(metadata, "optimizer-suggestions")

    toml_d = c.read_metadata()
    assert toml_d, "Loading metadata failed"

    logging.info(f"Read metadata:\n{tomlkit.dumps(toml_d)}")
