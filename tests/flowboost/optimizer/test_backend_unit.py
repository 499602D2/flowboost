"""Unit tests for Backend validation — no OpenFOAM or Ax initialization needed."""

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import AggregateObjective, Objective
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


class TestBackendCreate:
    def test_ax_lowercase(self):
        backend = AxBackend.create("ax")
        assert isinstance(backend, AxBackend)

    def test_axbackend_mixed_case(self):
        backend = AxBackend.create("AxBackend")
        assert isinstance(backend, AxBackend)

    def test_unknown_raises(self):
        with pytest.raises(NotImplementedError):
            AxBackend.create("unknown_backend")


class TestPostProcessSuggestionParametrizations:
    def test_maps_str_keys_to_dimensions(self):
        backend = AxBackend()
        dim_a = _make_dim("alpha")
        dim_b = _make_dim("beta")
        backend.set_search_space([dim_a, dim_b])

        parametrizations = {0: {"alpha": 1.0, "beta": 2.0}}
        result = backend._post_process_suggestion_parametrizations(parametrizations)

        assert len(result) == 1
        assert result[0][dim_a] == 1.0
        assert result[0][dim_b] == 2.0

    def test_unknown_key_raises(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim("known")])

        with pytest.raises(ValueError, match="not found"):
            backend._post_process_suggestion_parametrizations({0: {"unknown_dim": 1.0}})

    def test_empty_parametrizations(self):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        assert backend._post_process_suggestion_parametrizations({}) == []


class TestDimToAxEdgeCases:
    def test_fixed_with_no_values_raises(self):
        backend = AxBackend()
        dim = Dimension("x", "fixed")
        dim.value_type = "int"
        dim.values = None
        backend.set_search_space([dim])
        with pytest.raises(ValueError, match="specified value"):
            backend._get_ax_search_space()

    def test_unknown_dim_type_raises(self):
        backend = AxBackend()
        dim = Dimension("x", "bogus")
        dim.value_type = "int"
        backend.set_search_space([dim])
        with pytest.raises(ValueError, match="not supported"):
            backend._get_ax_search_space()


class TestBatchProcessAllFailed:
    def test_all_cases_failed_returns_empty(self, tmp_path):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        obj = Objective("test", minimize=True, objective_function=lambda c: None)
        backend.set_objectives([obj])

        cases = []
        for i in range(3):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        result = backend.batch_process(cases)
        assert result == []
        # All should be marked failed
        assert all(c.success is False for c in cases)


class TestBatchProcessSuccessStateSemantics:
    def test_success_none_case_is_processed_and_persisted(self, tmp_path):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        backend.set_objectives([_make_objective("score")])

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)

        result = backend.batch_process([case])

        assert result == [[1.0]]
        metadata = case.read_metadata()
        assert metadata is not None
        assert metadata["objective-outputs"]["score"]["value"] == 1.0
        assert metadata["objective-values-raw"]["score"] == 1.0

    def test_success_none_case_is_processed_for_aggregate_objective(self, tmp_path):
        backend = AxBackend()
        backend.set_search_space([_make_dim()])
        backend.set_objectives(
            [
                AggregateObjective(
                    "agg",
                    minimize=True,
                    objectives=[
                        Objective("a", minimize=True, objective_function=lambda c: 2.0),
                        Objective("b", minimize=True, objective_function=lambda c: 3.0),
                    ],
                    threshold=0.0,
                )
            ]
        )

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)

        result = backend.batch_process([case])

        assert result == [[5.0]]
        metadata = case.read_metadata()
        assert metadata is not None
        assert metadata["objective-outputs"]["agg"]["value"] == 5.0
        assert metadata["objective-values-raw"]["agg"] == 5.0
