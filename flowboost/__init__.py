def __getattr__(name):
    _imports = {
        "Case": "flowboost.openfoam.case",
        "Dictionary": "flowboost.openfoam.dictionary",
        "Dimension": "flowboost.optimizer.search_space",
        "Objective": "flowboost.optimizer.objectives",
        "AggregateObjective": "flowboost.optimizer.objectives",
        "Manager": "flowboost.manager.manager",
        "Session": "flowboost.session.session",
    }
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'flowboost' has no attribute {name!r}")


__all__ = [
    "AggregateObjective",
    "Case",
    "Dictionary",
    "Dimension",
    "Manager",
    "Objective",
    "Session",
]
