"""Compare Ax acquisition functions on the Six-Hump Camel benchmark."""

import numpy as np
from ax.adapter.registry import Generators
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.acquisition import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

ACQUISITIONS = {
    "ei": qExpectedImprovement,
    "pi": qProbabilityOfImprovement,
    "ucb": qUpperConfidenceBound,
}
GLOBAL_MINIMUM = -1.031628453489877


def six_hump_camel(x: float, y: float) -> float:
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def build_strategy(acquisition: str, seed: int, n_init: int = 5) -> GenerationStrategy:
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {list(ACQUISITIONS)}")

    botorch_node = GenerationNode(
        name=f"BoTorch-{acquisition.upper()}",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                generator_kwargs={
                    "botorch_acqf_class": ACQUISITIONS[acquisition],
                },
            )
        ],
    )
    sobol_node = GenerationNode(
        name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
                generator_kwargs={"seed": seed},
            )
        ],
        transition_criteria=[
            MinTrials(
                threshold=n_init,
                transition_to=botorch_node.name,
                use_all_trials_in_exp=True,
            )
        ],
    )
    return GenerationStrategy(
        name=f"Sobol+{acquisition.upper()}",
        nodes=[sobol_node, botorch_node],
    )


def run(
    acquisition: str, seed: int, n_init: int = 5, n_trials: int = 25
) -> list[float]:
    client = AxClient(generation_strategy=build_strategy(acquisition, seed, n_init))
    client.create_experiment(
        name=f"six-hump-camel-{acquisition}-{seed}",
        parameters=[
            {
                "name": "x",
                "type": "range",
                "bounds": [-3.0, 3.0],
                "value_type": "float",
            },
            {
                "name": "y",
                "type": "range",
                "bounds": [-2.0, 2.0],
                "value_type": "float",
            },
        ],
        objectives={"camel": ObjectiveProperties(minimize=True)},
    )

    best_so_far: list[float] = []
    best = float("inf")
    for _ in range(n_trials):
        parameters, trial_index = client.get_next_trial()
        value = six_hump_camel(parameters["x"], parameters["y"])
        client.complete_trial(trial_index, raw_data={"camel": value})
        best = min(best, value)
        best_so_far.append(best)
    return best_so_far


if __name__ == "__main__":
    for acquisition in ACQUISITIONS:
        runs = np.vstack([run(acquisition, seed=seed) for seed in range(10)])
        print(
            f"{acquisition.upper()}: final best = "
            f"{runs[:, -1].mean():.4f} +/- {runs[:, -1].std():.4f}; "
            f"global minimum = {GLOBAL_MINIMUM:.4f}"
        )
