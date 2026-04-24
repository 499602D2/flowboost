from pathlib import Path
from typing import Callable, Union

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.search_space import Dimension


@pytest.fixture
def case(tmp_path: Path) -> Case:
    """A bare Case under tmp_path. No metadata, no parameters."""
    d = tmp_path / "test_case"
    d.mkdir()
    return Case(d)


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


@pytest.fixture
def make_dim() -> Callable[[str], Dimension]:
    """Factory for a generic 0..1 range Dimension. Used by backend/objective
    unit tests that need a search-space dimension but don't care about the
    underlying dictionary entry."""

    def _make(name: str = "dim") -> Dimension:
        link = DictionaryLink("constant/foo").entry("bar")
        return Dimension.range(name=name, link=link, lower=0.0, upper=1.0)

    return _make
