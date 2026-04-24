from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowboost.manager.manager import Manager as Manager
    from flowboost.openfoam.case import Case as Case
    from flowboost.openfoam.data import PandasData as PandasData
    from flowboost.openfoam.data import PolarsData as PolarsData
    from flowboost.openfoam.dictionary import Dictionary as Dictionary
    from flowboost.openfoam.runtime import get_runtime as foam_runtime
    from flowboost.optimizer.objectives import (
        AggregateObjective as AggregateObjective,
        Objective as Objective,
    )
    from flowboost.optimizer.search_space import Dimension as Dimension
    from flowboost.session.session import Session as Session


def __getattr__(name):
    _imports = {
        "Case": "flowboost.openfoam.case",
        "PandasData": "flowboost.openfoam.data",
        "PolarsData": "flowboost.openfoam.data",
        "Dictionary": "flowboost.openfoam.dictionary",
        "Dimension": "flowboost.optimizer.search_space",
        "Objective": "flowboost.optimizer.objectives",
        "AggregateObjective": "flowboost.optimizer.objectives",
        "Manager": "flowboost.manager.manager",
        "Session": "flowboost.session.session",
        "foam_runtime": ("flowboost.openfoam.runtime", "get_runtime"),
    }
    if name in _imports:
        import importlib

        entry = _imports[name]
        if isinstance(entry, tuple):
            module_path, attr = entry
        else:
            module_path, attr = entry, name
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'flowboost' has no attribute {name!r}")


__all__ = [
    "AggregateObjective",
    "Case",
    "Dictionary",
    "Dimension",
    "Manager",
    "Objective",
    "PandasData",
    "PolarsData",
    "Session",
    "foam_runtime",
]
