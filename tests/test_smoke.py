import flowboost


def test_public_api():
    for name in flowboost.__all__:
        attr = getattr(flowboost, name)
        assert callable(attr), f"{name} is not callable"


def test_submodule_imports():
    from flowboost.openfoam.case import Case  # noqa: F401
    from flowboost.openfoam.dictionary import Dictionary  # noqa: F401
    from flowboost.optimizer.search_space import Dimension  # noqa: F401
    from flowboost.manager.manager import Manager  # noqa: F401
    from flowboost.session.session import Session  # noqa: F401
