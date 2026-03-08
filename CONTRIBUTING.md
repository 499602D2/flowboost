# Contributing

Contributions are welcome! Please ensure your changes pass the existing tests and linting before submitting a pull request.

## Setup

FlowBoost uses [uv](https://docs.astral.sh/uv/) for package and project management. uv replaces pip, virtualenv, and conda with a single tool: it manages Python versions, creates isolated environments, and installs dependencies; all significantly faster than the alternatives. The `uv.lock` lockfile ensures everyone gets exactly the same dependency versions.

To get started, [install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```shell
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install dependencies and set up pre-commit hooks:

```shell
uv sync
uv run pre-commit install
```

`uv sync` creates a virtual environment and installs everything. Use `uv run` to run commands inside it (e.g. `uv run pytest`), or activate it manually with `source .venv/bin/activate`.

## Tooling

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting. [ty](https://github.com/astral-sh/ty) provides type checking. Both run automatically via pre-commit hooks on each commit.

- **Ruff**: blocks the commit if there are lint or formatting issues (auto-fixes where possible)
- **ty**: advisory only — reports type errors but does not block commits

You can run them manually:

```shell
uv run ruff check --fix .
uv run ruff format .
uv run ty check
```

## Testing

Tests run in parallel via pytest-xdist:

```shell
uv run pytest                          # full suite
uv run pytest tests/path/to/test.py    # single file
uv run pytest -k "test_name"           # by name
uv run pytest -m "not slow"            # skip slow tests
```

### OpenFOAM

Most tests require OpenFOAM (see [README](README.md#openfoam) for setup). Tests that require a native install are marked with `@pytest.mark.native_foam_only`. See the `pitzDaily` example for Docker-based usage with `DockerLocal` and the `container()` context manager.

## CI/CD

CI runs automatically on all pull requests and pushes to `main`, testing against Python 3.10–3.14 (including lowest-direct dependency resolution on 3.10).

Releases are published to [PyPI](https://pypi.org/p/flowboost) automatically when a version tag is pushed:

```shell
git tag v<version>
git push origin v<version>
```
