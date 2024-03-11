# 🏄‍♂️ FlowBoost — Multi-objective Bayesian optimization for OpenFOAM
FlowBoost provides a highly configurable and extendable framework for handling and optimizing OpenFOAM CFD simulations. The framework provides ready bindings for state-of-the-art Bayesian optimization using Meta's Ax, powered by PyTorch, and easily extendable abstractions for using any other optimization library.

## Features
- Easy API syntax (see `examples/`)
- Ready bindings for [Meta's Ax (Adaptive Experimentation Platform)](https://ax.dev/)
  - Multi-objective, high-dimensional Bayesian optimization
  - SAASBO, GPU acceleration
- Fully hands-off cluster-native job management
- Simple abstractions for OpenFOAM cases (`flowboost.Case`)

## Optimization framework
FlowBoost's goal is to be minimal, unopinionated, and highly configurable. You can, for example:
- Use any optimization framework as a backend by implementing a few interfaces (`flowboost <-> your backend`)
- Submit your jobs to any evaluation environment
- Implement objective functions any way you want

## Examples
The `examples/` directory contains code examples for a few simplified real-world scenarios:

1) Very simple example based on the aachenBomb tutorial case
2) Optimization of injection timing in an internal combustion engine
3) Parameter optimization for in-situ adapative tabulation (ISAT) configuration

By default, FlowBoost uses Ax's [Service API](https://ax.dev/tutorials/gpei_hartmann_service.html) as its optimization backend. In practice, any optimizer can be used, as long as it conforms to the abstract `flowboost.optimizer.Backend` base class, which the backend interfaces in `flowboost.optimizer.interfaces` implement.

## OpenFOAM case abstraction
Working with OpenFOAM cases is performed through the `flowboost.Case` abstraction, which provides a high-level API for OpenFOAM case-data and configuration access. The `Case` abstraction can be used as-is outside of optimization workflows:

```python
from flowboost import Case

# Clone tutorial to current working directory (or a specified dir)
tutorial_case = Case.from_tutorial("multicomponentFluid/aachenBomb")

# Dictionary read/write access
control_dict = new_case.dictionary("system/controlDict")
control_dict.entry("deltaT").write(0.05)

# Access data in an evaluated case
case = Case("my/case/path")
df = case.data.simple_function_object_reader("integral_Qdot")
```

## Installation requirements
Installation can be done on any system with Python 3.10 or later. Older Python versions are not supported. There are certain caveats with regard to older CPU architectures (10+ years old), which are outlined below.

### CPU type
In order to use the standard `polars` package, your CPU should support AVX2 instructions ([and other SIMD instructions](https://github.com/pola-rs/polars/blob/78dc62851a13b87dc751a627e1e96ba1bf1549ee/py-polars/polars/_cpu_check.py)). These are typically available in Intel Broadwell/4000-series and later, and all AMD's Zen-based CPUs.

If your CPU is from 2012 or earlier, you will most likely receive an illegal instruction error. This can be solved by first uninstalling the `polars` package, and then installing `polars-lts-cpu`. The two packages are functionally identical, but the lts-version is not as performant.

## Setting up a development environment
Setting up a development environment should be done in a virtual environment. The project is packaged and managed using [Poetry](https://python-poetry.org/), meaning that you can set-up the environment using `poetry install` if you want to.

For a more traditional Conda/venv -based installation, see below. These methods use the `requirements.txt` files under `/requirements`, which are automatically generated by Poetry. The files can be re-generated using the `poetry run generate-requirements` script.

### Using venv
1. Ensure you have Python 3.10 or later installed.
2. Create a virtual environment in your project directory
3. Activate the environment and install dependencies

```shell
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements/requirements_dev.txt
```

### Using Conda
1. Ensure you have Anaconda or Miniconda installed
2. Create a Conda environment with Python 3.10 or later
   1. You can specify e.g. `python=3.12` when creating the Conda environment
3. Install dependencies from the `requirements/requirements_dev.txt` file

```shell
conda create --name flowboost --file requirements/requirements_dev.txt python=3.12
conda activate flowboost
```

## Getting started with the PyPi package
To get started with FlowBoost, install the `flowboost` Python package. A Python version of at least 3.10 is required. Additionally, you should use a virtual environment to manage the installation (Conda, Miniconda, venv, etc.):

```shell
# Ensure version ≥3.9 if using venv or global install
python3 --version

# Optional: initialize a virtual environment (venv, conda, etc.)
conda create --name flowboost python=3.11
conda activate flowboost

# Install package
pip3 install -U flowboost
```

The `flowboost` modules are now available in your environment. To get started, see documentation at `documentation_url`, and start importing modules!

## GPU acceleration
If your environment has a CUDA-compatible NVIDIA GPU, you should verify you have a recent CUDA Toolkit release. Otherwise, the GPU acceleration that PyTorch heavily exploits will not be available. This is especially critical if you are using SAASBO for high-dimensional optimization tasks (≥20 dimensions).

```shell
nvcc -V

# Verify CUDA availability
conda activate flowboost
python3
> import torch
> torch.cuda.is_available()
> exit()
```

## Unit-tests
The test suite can be run by simply installing the project dependencies outlined in `pyproject.toml`, and running `pytest`. Passing the full test suite requires you to have OpenFOAM installed and sourced. Note, that FlowBoost has only been tested using the org-lineage of OpenFOAM, more specifically version 11.

If you wish to contribute code to the project, please ensure your contribution still passes the current test coverage.

## Acknowledgments
The base functionality for FlowBoost was created as part of a mechanical engineering master's thesis at Aalto University, funded by Wärtsilä. Wärtsilä designs and manufactures marine combustion engines and energy solutions in Vaasa, Finland.