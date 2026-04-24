"""OpenFOAM postProcessing data access with pluggable DataFrame backends."""

from .base import Data
from .pandas import PandasData
from .polars import PolarsData

# Backend registry — used by Data._for_backend() for per-call overrides
_BACKENDS: dict[str, type[Data]] = {
    "polars": PolarsData,
    "pandas": PandasData,
}

__all__ = ["Data", "PandasData", "PolarsData"]
