import logging
from pathlib import Path

import pytest

from flowboost.manager.manager import SUPPORTED_SCHEDULERS, Manager
from flowboost.openfoam.case import Case, unique_id


@pytest.fixture
def manager(tmp_path, request) -> Manager:
    """
    Generates a manager from a manager name. Test is skipped for said manager
    if it is not available.
    """
    manager_name = request.param

    try:
        return Manager.create(scheduler=manager_name, wdir=tmp_path, job_limit=3)
    except ValueError:
        pytest.skip(f"Manager '{manager_name}' not available")


@pytest.mark.parametrize("manager", SUPPORTED_SCHEDULERS, indirect=True)
def test_abstract_methods(foam_in_env, manager: Manager):
    """
    Test abstract methods for each supported, available manager. Requires
    OpenFOAM to be sourced for case utilities.
    """
    # Test with a tutorial case
    case_path = Path(manager.wdir, f"TEST_flowboost_{unique_id()}")
    case = Case.from_tutorial("multicomponentFluid/aachenBomb", case_path)

    logging.debug(f"Testing submission with {case}")

    # Submit
    success = manager.submit_case(case)
    assert success, "Submission failed"

    # Check the job pool
    logging.debug(f"Manager job pool status:\n{manager._status_print()}")

    # Try free slots (also runs _has_finished)
    slots = manager.free_slots()
    assert (
        slots == manager.job_limit - 1
    ), f"Free slots incorrect: {slots} != {manager.job_limit-1}"

    # Cancel
    for job in manager.job_pool:
        manager.cancel_job(job)
