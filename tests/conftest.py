from pathlib import Path

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.interface import FOAM


@pytest.fixture(scope="session")
def data_dir():
    """Fixture to provide the path to the data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def foam_in_env():
    """
    Ensures OpenFOAM is sourced. If not, the test is skipped.

    Returns:
        bool: FOAM sourced?
    """
    if not FOAM.in_env():
        pytest.skip("OpenFOAM not sourced")
    return True


@pytest.fixture(scope="session")
def test_case(data_dir):
    case_path = Path(data_dir)
    assert (
        case_path.exists()
    ), f"Path to example case data not found: [{case_path.resolve()}]"
    case = Case(case_path)
    yield case
