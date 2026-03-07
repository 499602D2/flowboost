from pathlib import Path

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.interface import FOAM
from flowboost.openfoam.runtime import FOAMRuntime, get_runtime


@pytest.fixture(scope="session")
def data_dir():
    """Fixture to provide the path to the data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def foam_runtime(tmp_path_factory):
    """Session-scoped fixture that provides a configured FOAMRuntime."""
    try:
        runtime = get_runtime()
    except RuntimeError:
        pytest.skip("OpenFOAM not available (no native install, no Docker)")
        return

    if not runtime.is_available():
        pytest.skip("OpenFOAM runtime not usable (Docker image missing or cannot be built)")
        return

    if runtime.mode != FOAMRuntime.Mode.NATIVE:
        # Mount tmp dir for test output
        mount_root = tmp_path_factory.getbasetemp()
        runtime.add_mount(mount_root, "/work")
        # Mount test data dir so foamDictionary can access test case files
        tests_dir = Path(__file__).parent
        runtime.add_mount(tests_dir, "/testdata")

    yield runtime
    runtime.cleanup()


@pytest.fixture(scope="session")
def foam_in_env(foam_runtime):
    """Ensures OpenFOAM is available (native or Docker). Skips if not."""
    return True


@pytest.fixture(scope="session")
def test_case(data_dir):
    case_path = Path(data_dir)
    if not case_path.exists():
        pytest.skip(f"Test data not found: {case_path.resolve()}")
    case = Case(case_path)
    yield case
