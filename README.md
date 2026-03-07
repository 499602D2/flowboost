# 🏄‍♂️ FlowBoost — Multi-objective Bayesian optimization for OpenFOAM

![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12_%7C_3.13_%7C_3.14-blue)

FlowBoost is a highly configurable and extensible library for handling and optimizing OpenFOAM CFD simulations. It provides ready bindings for state-of-the-art Bayesian optimization using Meta's Ax, powered by PyTorch, and simple interfaces for using any other optimization library.

## Features
- Easy API syntax (see `examples/`)
- Ready bindings for [Meta's Ax (Adaptive Experimentation Platform)](https://ax.dev/)
  - Multi-objective, high-dimensional Bayesian optimization
  - SAASBO, GPU acceleration
- Fully hands-off cluster-native job management
- Simple interfaces for OpenFOAM cases (`flowboost.Case`)
- Use any optimization backend by implementing a few interfaces

## Examples
The `examples/` directory contains code examples for simplified real-world scenarios:

1. `aerofoilNACA0012Steady`: parameter optimization for a NACA 0012 aerofoil steady-state simulation

By default, FlowBoost uses Ax's [Service API](https://ax.dev/) as its optimization backend. In practice, any optimizer can be used, as long as it conforms to the abstract `flowboost.optimizer.Backend` base class, which the backend interfaces in `flowboost.optimizer.interfaces` implement.

## OpenFOAM case abstraction
Working with OpenFOAM cases is performed through the `flowboost.Case` abstraction, which provides a high-level API for OpenFOAM case-data and configuration access. The `Case` abstraction can be used as-is outside of optimization workflows:

```python
from flowboost import Case

# Clone tutorial to current working directory (or a specified dir)
tutorial_case = Case.from_tutorial("fluid/aerofoilNACA0012Steady")

# Dictionary read/write access
control_dict = tutorial_case.dictionary("system/controlDict")
control_dict.entry("writeInterval").set("5000")

# Access data in an evaluated case
case = Case("my/case/path")
df = case.data.simple_function_object_reader("forceCoeffsCompressible")
```

## Installation
FlowBoost requires Python 3.10 or later.

### uv (recommended)
```shell
uv add flowboost
```

### pip
```shell
pip install flowboost
```

### CPU compatibility
In order to use the standard `polars` package, your CPU should support AVX2 instructions ([and other SIMD instructions](https://github.com/pola-rs/polars/blob/78dc62851a13b87dc751a627e1e96ba1bf1549ee/py-polars/polars/_cpu_check.py)). These are typically available in Intel Broadwell/4000-series and later, and all AMD Zen-based CPUs.

If your CPU is from 2012 or earlier, you will most likely receive an illegal instruction error. This can be solved by installing the `lts-cpu` extra:

```shell
uv add flowboost[lts-cpu]
# or: pip install flowboost[lts-cpu]
```

This installs `polars-lts-cpu`, which is functionally identical but not as performant.

## Development

The project is packaged using [uv](https://docs.astral.sh/uv/). The code is linted and formatted using [Ruff](https://docs.astral.sh/ruff/).

```shell
uv sync
uv run pytest
```

### GPU acceleration
If your environment has a CUDA-compatible NVIDIA GPU, you should verify you have a recent CUDA Toolkit release. Otherwise, GPU acceleration for PyTorch will not be available. This is especially critical if you are using SAASBO for high-dimensional optimization tasks (≥20 dimensions).

```shell
nvcc -V

# Verify CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Testing
Passing the full test suite requires OpenFOAM to be installed and sourced. FlowBoost has only been tested using the [openfoam.org](https://openfoam.org/) lineage (not the ESI/openfoam.com fork), specifically version 11.

If you wish to contribute code to the project, please ensure your contribution still passes the current test coverage.

## Acknowledgments
The base functionality for FlowBoost was created as part of a mechanical engineering master's thesis at Aalto University, funded by Wärtsilä. Wärtsilä designs and manufactures marine combustion engines and energy solutions in Vaasa, Finland.
