import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import Constraint, Objective, ScalarizedObjective
from flowboost.optimizer.search_space import Dimension

DEFAULT_OFFLOAD_RESULT_FNAME = "acquisition_result.json"


class Backend(ABC):
    def __init__(self) -> None:
        self.type: str = self.__class__.__name__
        self.objectives: list[Union[Objective, ScalarizedObjective]] = []
        self.constraints: list[Constraint] = []
        self.dimensions: list[Dimension] = []
        self.offload_acquisition: bool = False
        self.random_seed: int | None = None
        self.initialization_trials: int | None = None
        self._initialized: bool = False

    def _ensure_initialized(self, op: str) -> None:
        if not self._initialized:
            raise RuntimeError(
                f"Backend.{op}() called before initialize(). Call "
                f"backend.initialize() (or Session.start()) after setting "
                f"objectives and the search space."
            )

    @staticmethod
    def create(backend: str) -> "Backend":
        from flowboost.optimizer.interfaces.Ax import AxBackend

        match backend.lower():
            case "ax" | "axbackend":
                return AxBackend()
            case _:
                raise NotImplementedError(f"Backend '{backend}' not implemented")

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def set_search_space(self, dimensions: list[Dimension]):
        pass

    @abstractmethod
    def set_objectives(self, objectives: list[Union[Objective, ScalarizedObjective]]):
        pass

    @abstractmethod
    def set_parameter_constraints(self, constraints: list[str]):
        pass

    def set_constraints(self, constraints: Union[Constraint, list[Constraint]]):
        """Register Constraint objects: tracking metrics that bound the
        optimizer's exploration without being optimization targets.

        Accepts either a single Constraint or an iterable of Constraints.
        Rejects non-Constraint items up front so misuse surfaces here instead
        of crashing later with an AttributeError inside the evaluation loop.
        """
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        try:
            constraints_list = list(constraints)
        except TypeError as e:
            raise TypeError(
                f"set_constraints expects a Constraint or an iterable of "
                f"Constraints, got {type(constraints).__name__!r}."
            ) from e
        for i, c in enumerate(constraints_list):
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"set_constraints[{i}] must be a Constraint, got "
                    f"{type(c).__name__!r}. Use set_objectives() for "
                    f"Objective and ScalarizedObjective."
                )
        self.constraints = constraints_list

    @abstractmethod
    def attach_pending_cases(self, cases: list[Case]):
        """
        If possible in the backend, currently running cases should be
        re-attached to the model as being pending. This ensures that the
        same points are not generated twice.

        Args:
            cases (list[Case]): Currently running/pending cases
        """
        pass

    @abstractmethod
    def attach_failed_cases(self, cases: list[Case]):
        """
        Attaches failed cases to the backend, to avoid re-evaluation.

        Args:
            cases (list[Case]): List of failed cases.
        """
        pass

    def ask(self, max_cases: int) -> list[dict[Dimension, Any]]:
        self._ensure_initialized("ask")
        parametrizations = self._ask(max_cases)
        return self._post_process_suggestion_parametrizations(parametrizations)

    @abstractmethod
    def _ask(self, max_cases: int) -> Any:
        pass

    @abstractmethod
    def _post_process_suggestion_parametrizations(
        self, parametrizations: dict[int, dict[str, Any]]
    ) -> list[dict[Dimension, Any]]:
        """
        Post-process the result of Backend.ask(), returning a list of
        suggestion dictionaries (mappings of Dimension->value).

        Args:
            parametrizations (Any): Arbitrary result returned by backend

        Returns:
            list[dict[Dimension, Any]]: Post-processed suggestions
        """
        pass

    @abstractmethod
    def tell(self, cases: list[Case]):
        pass

    @abstractmethod
    def tell_and_ask(
        self, cases: list[Case], max_cases: int
    ) -> list[dict[Dimension, Any]]:
        """
        A shorthand for first updating the surrogate model with tell(), and
        then running next-point acquisition with ask().

        Args:
            cases (list[Case]): List of cases to use for model update
            max_cases (int): Limit for new cases. May not always be respected.

        Returns:
            list[dict[Dimension, Any]]: List of Dimension-keyed dictionaries, \
                one key-value pair per configured Dimension. One list entry \
                per Case.
        """
        pass

    def prepare_for_acquisition_offload(
        self, finished_cases: list[Case], pending_cases: list[Case], save_in: Path
    ) -> tuple[Path, Path]:
        """
        Prepares the optimizer backend for acquisition offload. The preparation
        entails the serialization of the model configuration, and the
        pre-computed objective function outputs and case configuration
        parametrizations.

        This ensures that the acquisition offload does not require code
        duplication, and can be achieved by simply ingressing two snapshot
        files.

        The function uses the abstract Backend.produce_state_snapshot for
        the model snapshot. However, the data snapshot here is generic,
        and can be overridden if needed.

        Args:
            finished_cases (list[Case]): List of finished, processed cases
            save_in (Path): Path to save snapshot files in

        Returns:
            tuple[Path, Path]: Paths to (model_snapshot, data_snapshot).
        """
        # Save a state snapshot
        model_snapshot = self.produce_state_snapshot(save_in)

        self._ensure_unique_case_names(
            [*finished_cases, *pending_cases],
            context="finished and pending cases",
        )

        # Construct one dictionary, keyed by names (Case isn't serializable)
        serializable = {
            "optimizer": self.__class__.__name__,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "finished_cases": {
                c.name: {
                    "parametrization": c.parametrize_configuration(self.dimensions),
                    "objectives": c.objective_function_outputs(
                        [*self.objectives, *self.constraints]
                    ),
                }
                for c in finished_cases
            },
            "pending_cases": {
                c.name: {
                    "parametrization": c.parametrize_configuration(self.dimensions)
                }
                for c in pending_cases
            },
        }

        data_snapshot = Path(save_in, "data_snapshot.json")
        with open(data_snapshot, "w") as f:
            json.dump(serializable, f)

        logging.info(
            f"Saved data snapshot (parameters + obj.f. outputs) [{data_snapshot}]"
        )
        return (model_snapshot, data_snapshot)

    @staticmethod
    def _ensure_unique_case_names(cases: list[Case], *, context: str) -> None:
        names = [case.name for case in cases]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            duplicates = ", ".join(duplicate_names)
            raise ValueError(
                f"Case names must be unique across {context}: {duplicates}"
            )

    @abstractmethod
    def produce_state_snapshot(self, save_in: Path) -> Path:
        pass

    @classmethod
    @abstractmethod
    def restore_from_state_snapshot(cls, from_file: Path) -> "Backend":
        pass

    def batch_process(self, cases: list[Case]) -> list[list[float]]:
        """Evaluate every objective and constraint on `cases` and persist
        results to metadata.

        Returns a list-of-lists shaped [num_objectives][num_successful_cases]
        of objective values. Constraint values are persisted to metadata but
        not part of the return shape (constraints aren't optimization targets).
        """
        for objective in self.objectives:
            logging.info(f"Processing objective '{objective.name}'")
            objective.batch_evaluate(cases, save_values=True)
        for constraint in self.constraints:
            logging.info(f"Processing constraint '{constraint.name}'")
            constraint.batch_evaluate(cases, save_values=True)

        successful_cases = [c for c in cases if c.success is not False]
        logging.info(f"Proceeding with {len(successful_cases)} successful case(s)")
        if not successful_cases:
            return []

        final_outputs: list[list[float]] = [
            [objective.data_for_case(c) for c in successful_cases]
            for objective in self.objectives
        ]

        for case_idx, case in enumerate(successful_cases):
            objective_results: dict[str, Any] = {}
            for obj_idx, objective in enumerate(self.objectives):
                entry: dict[str, Any] = {
                    "value": final_outputs[obj_idx][case_idx],
                    "minimize": objective.minimize,
                }
                if isinstance(objective, ScalarizedObjective):
                    entry["components"] = {
                        inner.name: inner.data_for_case(case)
                        for inner in objective.objectives
                    }
                    component_bounds: dict[str, dict[str, float]] = {}
                    for inner in objective.objectives:
                        bounds: dict[str, float] = {}
                        if inner.gte is not None:
                            bounds["gte"] = inner.gte
                        if inner.lte is not None:
                            bounds["lte"] = inner.lte
                        if bounds:
                            component_bounds[inner.name] = bounds
                    if component_bounds:
                        entry["component_bounds"] = component_bounds
                if isinstance(objective, Objective):
                    if objective.gte is not None:
                        entry["gte"] = objective.gte
                    if objective.lte is not None:
                        entry["lte"] = objective.lte
                objective_results[objective.name] = entry
            case.update_metadata(objective_results, entry_header="objective-outputs")

            if self.constraints:
                constraint_results: dict[str, Any] = {}
                for constraint in self.constraints:
                    entry: dict[str, Any] = {"value": constraint.data_for_case(case)}
                    if constraint.gte is not None:
                        entry["gte"] = constraint.gte
                    if constraint.lte is not None:
                        entry["lte"] = constraint.lte
                    constraint_results[constraint.name] = entry
                case.update_metadata(
                    constraint_results, entry_header="constraint-outputs"
                )

            logging.debug(f"Saved metric outputs to metadata for {case.name}")

        return final_outputs

    def _objective_name_to_objective(
        self, objective_name: str
    ) -> Union[Objective, ScalarizedObjective]:
        """
        Args:
            objective_name (str): Objective name to map to the corresponding \
                objective

        Raises:
            ValueError: _description_

        Returns:
            Union[Objective, ScalarizedObjective]: An Objective
        """
        for objective in self.objectives:
            if objective.name == objective_name:
                return objective

        raise ValueError(f"Requested objective '{objective_name}' not found in backend")

    def _dim_name_to_dimension(self, dim_name: str) -> Dimension:
        for dim in self.dimensions:
            if dim.name == dim_name:
                return dim

        raise ValueError(f"Requested dimension '{dim_name}' not found in backend")

    def _objective_names_are_unique(self) -> bool:
        return len(set([o.name for o in self.objectives])) == len(self.objectives)

    def _dim_names_are_unique(self) -> bool:
        return len(set([o.name for o in self.dimensions])) == len(self.dimensions)

    def _all_metric_names(self) -> list[str]:
        """Every metric name registered with the backend, across objectives
        (including inner objectives of a ScalarizedObjective, and the derived
        bound tracking metric emitted for each bounded Objective at either
        the top level or inside a ScalarizedObjective) and constraints.
        Used for cross-uniqueness validation."""
        names: list[str] = []
        for obj in self.objectives:
            if isinstance(obj, ScalarizedObjective):
                for inner in obj.objectives:
                    names.append(inner.name)
                    if inner.has_bounds:
                        names.append(inner.bound_metric_name)
            else:
                names.append(obj.name)
                if isinstance(obj, Objective) and obj.has_bounds:
                    names.append(obj.bound_metric_name)
        names.extend(c.name for c in self.constraints)
        return names

    def _verify_configuration(self):
        """Verify a few key-parts of the backend configuration.

        Raises:
            ValueError: If objectives or dimensions are not configured properly
        """
        if any([len(self.objectives) == 0, len(self.dimensions) == 0]):
            raise ValueError("Objectives or search space not defined")

        if not self._objective_names_are_unique():
            raise ValueError("All Objective names must be unique")

        if not self._dim_names_are_unique():
            raise ValueError("All Dimension names must be unique")

        # Constraint names must not clash with objective metric names — Ax
        # registers them all in the same metric namespace.
        all_names = self._all_metric_names()
        if len(set(all_names)) != len(all_names):
            duplicates = sorted({n for n in all_names if all_names.count(n) > 1})
            raise ValueError(
                f"Metric names must be unique across objectives "
                f"(including inner objectives of any ScalarizedObjective) "
                f"and constraints. Duplicates: {', '.join(duplicates)}"
            )


class OptimizationComplete(Exception):
    """Raised by a backend when the optimizer has exhausted its budget and
    no further trials can be produced. Callers (typically ``Session``) are
    expected to catch this and shut down the loop cleanly.
    """
