"""Canary: Ax must continue to standardize outcomes for us.

flowboost relies on Ax's default BoTorch transform stack to handle outcome
scale. In every supported Ax version that stack includes BilogY and
StandardizeY at the modelbridge/adapter layer, plus BoTorch's model-level
standardization. We deliberately removed our own normalization layer because
it duplicated and fought with Ax's.

If Ax ever drops StandardizeY from the default BoTorch setup, GP fitting on
objectives at large absolute scales degrades sharply. These tests assert the
dependency contract directly instead of using stochastic convergence as a
proxy.
"""

from importlib import import_module
from typing import Any

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective, ScalarizedObjective
from flowboost.optimizer.search_space import Dimension


def test_scalarized_optimization_config_survives_initialize(tmp_path):
    """The post-create swap in `_maybe_install_scalarized_objective` relies
    on Ax not capturing the optimization_config at GenerationStrategy build
    time. If Ax ever changes that, BO would silently optimize against the
    wrong (pre-swap MOO) config — convergence quality drops, no crash.

    This canary asserts the live optimization_config after initialize() is
    the Ax-native ScalarizedObjective we installed, not the placeholder
    MultiObjective that `create_experiment` generated."""
    from ax.core.objective import ScalarizedObjective as AxScalarizedObjective

    backend = AxBackend()
    backend.random_seed = 0
    backend.initialization_trials = 2
    backend.set_search_space(
        [
            Dimension.range(
                name="x",
                link=DictionaryLink("constant/setup").entry("x"),
                lower=0.0,
                upper=1.0,
            ),
        ]
    )
    inner_a = Objective("a", minimize=False, objective_function=lambda c: 1.0)
    inner_b = Objective("b", minimize=True, objective_function=lambda c: 2.0)
    backend.set_objectives(
        ScalarizedObjective(
            "agg", minimize=False, objectives=[inner_a, inner_b], weights=[0.7, -0.3]
        )
    )
    backend.initialize()

    objective = backend.client.experiment.optimization_config.objective
    assert isinstance(objective, AxScalarizedObjective), (
        f"Post-create swap to Ax-native ScalarizedObjective regressed; got "
        f"{type(objective).__name__}. flowboost relies on Ax modeling each "
        f"inner metric independently."
    )
    assert [m.name for m in objective.metrics] == ["a", "b"]
    assert list(objective.weights) == [0.7, -0.3]
    assert objective.minimize is False


def test_ax_default_transforms_still_include_outcome_standardization():
    """Direct introspection: assert the transforms flowboost relies on are
    still present in the default BoTorch generator's setup."""
    transform_names = _default_botorch_transform_names()

    for required in ("BilogY", "StandardizeY"):
        assert required in transform_names, (
            f"Ax's default BoTorch transforms no longer include {required!r}; "
            f"got {transform_names}. flowboost relies on Ax-side outcome "
            f"standardization — see test docstring."
        )


def _default_botorch_transform_names() -> list[str]:
    """Return Ax's default BoTorch transform names across supported Ax APIs."""
    setup = _find_default_botorch_setup()
    return [transform.__name__ for transform in setup.transforms]


def _find_default_botorch_setup() -> Any:
    registries = (
        ("ax.adapter.registry", "GENERATOR_KEY_TO_GENERATOR_SETUP"),
        ("ax.adapter.registry", "MODEL_KEY_TO_MODEL_SETUP"),
        ("ax.modelbridge.registry", "MODEL_KEY_TO_MODEL_SETUP"),
    )
    errors: list[str] = []

    for module_name, mapping_name in registries:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        mapping = getattr(module, mapping_name, None)
        if mapping is None:
            errors.append(f"{module_name}: missing {mapping_name}")
            continue

        if "BoTorch" in mapping:
            return mapping["BoTorch"]

        for key, setup in mapping.items():
            model_class = getattr(setup, "model_class", None)
            generator_class = getattr(setup, "generator_class", None)
            names = (str(key), repr(model_class), repr(generator_class))
            if any("BoTorch" in name for name in names):
                return setup

        errors.append(f"{module_name}.{mapping_name}: no BoTorch setup")

    raise AssertionError(
        "Could not locate Ax's default BoTorch setup in supported registries: "
        + "; ".join(errors)
    )
