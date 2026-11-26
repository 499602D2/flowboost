import pytest

from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective


@pytest.mark.parametrize(
    ("acquisition", "expected_name", "expected_node"),
    [
        ("ei", "Sobol+EI", "BoTorch-EI"),
        ("pi", "Sobol+PI", "BoTorch-PI"),
        ("ucb", "Sobol+UCB", "BoTorch-UCB"),
    ],
)
def test_acquisition_builds_generation_strategy(
    acquisition, expected_name, expected_node
):
    backend = AxBackend()
    backend.acquisition = acquisition

    strategy = backend._build_generation_strategy(num_init=5)

    assert strategy.name == expected_name
    assert list(strategy.nodes_by_name) == ["Sobol", expected_node]
    assert (
        "botorch_acqf_class"
        in strategy.nodes_by_name[expected_node].generator_specs[0].generator_kwargs
    )
    assert all(node.should_deduplicate for node in strategy.nodes_by_name.values())
    assert all(
        node.pausing_criteria[0].threshold == 3
        for node in strategy.nodes_by_name.values()
    )


def test_invalid_acquisition_raises():
    backend = AxBackend()
    backend.acquisition = "bogus"

    with pytest.raises(ValueError, match="acquisition must be one of"):
        backend._build_generation_strategy(num_init=5)


def test_initialize_accepts_custom_generation_strategy(make_dim):
    backend = AxBackend()
    backend.acquisition = "pi"
    backend.initialization_trials = 2
    backend.max_parallelism = 1
    backend.should_deduplicate = False
    backend.set_search_space([make_dim("x")])
    backend.set_objectives(
        [Objective("score", minimize=True, objective_function=lambda case: 1.0)]
    )

    backend.initialize()
    parameters, _ = backend.client.get_next_trial()
    strategy = backend._build_generation_strategy(num_init=2)

    assert backend._initialized
    assert set(parameters) == {"x"}
    assert all(
        node.pausing_criteria[0].threshold == 1
        for node in strategy.nodes_by_name.values()
    )
    assert all(not node.should_deduplicate for node in strategy.nodes_by_name.values())
