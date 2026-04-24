from pathlib import Path
from typing import Callable, Union

import pytest

from flowboost.openfoam.case import Case


@pytest.fixture
def make_suggestion_case() -> Callable[[Path, str, dict], Case]:
    """Factory that creates a Case directory with `params` written under
    `optimizer-suggestion`. Used by tests that need a Case representing a
    finished trial with known parameter values, without real OpenFOAM data.

    Takes the parent path explicitly (rather than capturing one from the
    fixture scope) so callers can place cases under per-cycle subdirectories.
    """

    def _make(
        parent: Path, name: str, params: dict[str, Union[int, float, bool, str]]
    ) -> Case:
        case_dir = parent / name
        case_dir.mkdir(parents=True, exist_ok=True)
        case = Case(case_dir)
        case.update_metadata(
            {key: {"value": value} for key, value in params.items()},
            entry_header="optimizer-suggestion",
        )
        return case

    return _make
