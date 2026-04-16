"""Unit tests for Backend validation — no OpenFOAM or Ax initialization needed."""

import pytest

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension


def _make_objective(name="obj"):
    return Objective(name=name, minimize=True, objective_function=lambda c: 1.0)


def _make_dim(name="dim"):
    link = DictionaryLink("constant/foo").entry("bar")
    return Dimension.range(name=name, link=link, lower=0.0, upper=1.0)


class TestVerifyConfiguration:
    def test_no_objectives_raises(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        with pytest.raises(ValueError, match="not defined"):
            backend._verify_configuration()

    def test_no_dimensions_raises(self):
        backend = AxBackend()
        backend.set_objectives([_make_objective()])
        with pytest.raises(ValueError, match="not defined"):
            backend._verify_configuration()

    def test_duplicate_objective_names_raises(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        backend.set_objectives([_make_objective("dup"), _make_objective("dup")])
        with pytest.raises(ValueError, match="unique"):
            backend._verify_configuration()

    def test_duplicate_dim_names_raises(self):
        backend = AxBackend()
        backend.set_objectives([_make_objective()])
        backend.set_search_space([_make_dim("dup"), _make_dim("dup")])
        with pytest.raises(ValueError, match="unique"):
            backend._verify_configuration()

    def test_valid_config_passes(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim("x")])
        backend.set_objectives([_make_objective("y")])
        backend._verify_configuration()  # should not raise


class TestNameLookups:
    def test_objective_lookup(self):
        backend = AxBackend()
        obj = _make_objective("target")
        backend.set_objectives([obj])
        assert backend._objective_name_to_objective("target") is obj

    def test_objective_lookup_missing_raises(self):
        backend = AxBackend()
        backend.set_objectives([_make_objective("other")])
        with pytest.raises(ValueError, match="not found"):
            backend._objective_name_to_objective("missing")

    def test_dim_lookup(self):
        backend = AxBackend()
        dim = _make_dim("target")
        backend.set_search_space([dim])
        assert backend._dim_name_to_dimension("target") is dim

    def test_dim_lookup_missing_raises(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim("other")])
        with pytest.raises(ValueError, match="not found"):
            backend._dim_name_to_dimension("missing")


class TestEnsureInitialized:
    def test_ask_before_init_raises(self):
        backend = AxBackend()
        with pytest.raises(RuntimeError, match="called before initialize"):
            backend._ensure_initialized("ask")


class TestFixedDimensionEncoding:
    def test_fixed_int_ax_encoding(self):
        backend = AxBackend()
        dim = Dimension.fixed("x", DictionaryLink("constant/foo").entry("bar"), 42)
        backend.set_search_space([dim])
        ax_params = backend._get_ax_search_space()
        assert len(ax_params) == 1
        p = ax_params[0]
        assert p["type"] == "fixed"
        assert p["value"] == 42
        assert p["value_type"] == "int"
