from __future__ import annotations

from typing import Any

import numpy as np

from flowboost.optimizer.search_space import Dimension


def coerce_objective_scalar(value: Any, *, label: str = "Objective output") -> float:
    """Coerce scalar-like objective data to a native Python float.

    Accepts Python numeric scalars, NumPy numeric scalars, and size-1
    ``ndarray`` / ``list`` / ``tuple`` containers. Multi-element containers are
    rejected because collapsing them would change the optimization target.
    """

    unwrapped = _unwrap_scalar_like(value, label=label)

    try:
        return Dimension._coerce(unwrapped, "float")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc


def _unwrap_scalar_like(value: Any, *, label: str) -> Any:
    if value is None:
        raise ValueError(f"{label} must be scalar-like, got None")

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"{label} must be scalar-like, got ndarray with {value.size} values"
            )
        return _unwrap_scalar_like(value.reshape(-1)[0], label=label)

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{label} must be scalar-like, got {type(value).__name__} with "
                f"{len(value)} values"
            )
        return _unwrap_scalar_like(value[0], label=label)

    if isinstance(value, np.generic):
        return value.item()

    return value
