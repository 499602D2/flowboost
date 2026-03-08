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
2. `pitzDaily`: backward-facing step optimization using local Docker execution and the Pandas data backend

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

## OpenFOAM

FlowBoost uses OpenFOAM in two ways:

1. **Case setup** uses CLI tools like `foamDictionary` and `foamCloneCase` on the host machine.
2. **Simulations** run wherever the `Manager` sends them: locally (`Local`, `DockerLocal`) or on a cluster (`SGE`, `Slurm`).

The host always needs access to OpenFOAM CLI tools for case setup, even when simulations run elsewhere. On Linux, a native install works. On macOS and Windows, FlowBoost provides these tools transparently through Docker.

- **Linux**: native OpenFOAM or Docker
- **macOS**: Docker ([OrbStack](https://orbstack.dev/) recommended, Docker Desktop also works)
- **Windows**: Docker (Docker Desktop). Not tested on Windows.

On first run in Docker mode, FlowBoost builds the `flowboost/openfoam:13` image from the bundled Dockerfile. This is a one-time operation. To force a specific mode, set `FLOWBOOST_FOAM_MODE` to `native` or `docker`. To use a custom image, set `FLOWBOOST_FOAM_IMAGE`.

FlowBoost uses the [openfoam.org](https://openfoam.org/) lineage (not the ESI/openfoam.com fork) and has been tested with versions 11 and 13. The bundled Dockerfile targets OpenFOAM 13 on Ubuntu 24.04. Each OpenFOAM release is tied to a specific Ubuntu LTS, so Dockerfiles are per-version by design.

### Using Docker mode

In Docker mode, CLI tools like `foamDictionary` run inside a persistent container. This is automatic: `Case`, `Dictionary`, and other abstractions work the same way regardless of the mode (native, Docker).

When running multiple OpenFOAM commands (e.g. reading dictionaries across many cases), use the `container()` context manager to keep a single container alive for the entire block:

```python
from flowboost import Case, foam_runtime

workdir = Path("flowboost_data")

with foam_runtime().container(workdir):
    for case_dir in sorted(workdir.glob("case_*")):
        case = Case(case_dir)
        k = case.dictionary("0/k").entry("boundaryField/inlet/value").value
        # All foamDictionary calls reuse the same container
```

Without `container()`, FlowBoost auto-mounts paths as needed, which may restart the container when new paths are encountered. Pre-mounting a parent directory (like the workdir above) avoids this.

To run simulations locally in Docker, use the `DockerLocal` manager:

```python
from flowboost import Manager

manager = Manager.create(scheduler="dockerlocal", wdir=data_dir, job_limit=2)
```

Each submitted case gets its own detached container with the case directory bind-mounted. See the `pitzDaily` example for a complete Docker-based workflow.

## GPU acceleration

If your environment has a CUDA-compatible NVIDIA GPU, verify you have a recent CUDA Toolkit release. Otherwise, GPU acceleration for PyTorch will not be available. This is especially critical if you are using SAASBO for high-dimensional optimization tasks (≥20 dimensions).

```shell
nvcc -V

# Verify CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tooling, and testing guidance.

## Acknowledgments
The base functionality for FlowBoost was created as part of a mechanical engineering master's thesis at Aalto University, funded by Wärtsilä. Wärtsilä designs and manufactures marine combustion engines and energy solutions in Vaasa, Finland.
