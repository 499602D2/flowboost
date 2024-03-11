import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import AggregateObjective, Objective
from flowboost.optimizer.search_space import Dimension

DEFAULT_OFFLOAD_RESULT_FNAME = "acquisition_result.json"


class Backend(ABC):
    def __init__(self) -> None:
        self.type: str = self.__class__.__name__
        self.objectives: list[Union[Objective, AggregateObjective]] = []
        self.dimensions: list[Dimension] = []
        self.offload_acquisition: bool = False

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
    def set_objectives(self, objectives: list[Union[Objective, AggregateObjective]]):
        pass

    @abstractmethod
    def set_parameter_constraints(self, constraints: list[str]):
        pass

    @abstractmethod
    def set_outcome_constraints(self, constraints: list[str]):
        pass

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
        parametrizations = self._ask(max_cases)
        return self._post_process_suggestion_parametrizations(parametrizations)

    @abstractmethod
    def _ask(self, max_cases: int) -> Any:
        pass

    @abstractmethod
    def _post_process_suggestion_parametrizations(
        self, parametrizations: Any
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

        # Ensure case names are unique
        if len(set([c.name for c in finished_cases])) != len(finished_cases):
            raise ValueError("All case names must be unique: cannot proceed")

        # Construct one dictionary, keyed by names (Case isn't serializable)
        serializable = {
            "optimizer": self.__class__.__name__,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "finished_cases": {
                c.name: {
                    "parametrization": c.parametrize_configuration(self.dimensions),
                    "objectives": c.objective_function_outputs(self.objectives),
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

    @abstractmethod
    def produce_state_snapshot(self, save_in: Path) -> Path:
        pass

    @classmethod
    @abstractmethod
    def restore_from_state_snapshot(cls, from_file: Path) -> "Backend":
        pass

    def batch_process(self, cases: list[Case]) -> list[list[float]]:
        """
        Process a batch of cases through all objectives, applying any necessary
        batch processing and aggregation, and return a list of lists of floats
        suitable for optimization.
        """
        all_objective_outputs = []

        # Step 1: Evaluate the objective functions
        for objective in self.objectives:
            logging.info(f"Processing objective '{objective.name}'")
            objective_outputs = objective.batch_evaluate(cases, save_values=False)
            all_objective_outputs.append(objective_outputs)

        # Step 2: Ensure that no cases were marked as failed: if they did, remove them
        # Remove None values from all_objective_outputs
        successful_cases_indices = [i for i, case in enumerate(cases) if case.success]
        all_objective_outputs = [
            [
                output
                for i, output in enumerate(objective_outputs)
                if i in successful_cases_indices
            ]
            for objective_outputs in all_objective_outputs
        ]

        # Update logging to reflect the removal of failed cases
        logging.info(
            f"Removed failed cases: proceeding with {len(successful_cases_indices)} successful case(s)"
        )

        if len(successful_cases_indices) == 0:
            return []

        # Step 3: Execute post-processing steps for only successful cases
        final_outputs = []
        for i, objective in enumerate(self.objectives):
            logging.info(f"Post-processing objective '{objective.name}' outputs.")
            # Filter cases and their outputs for successful cases only
            successful_cases = [cases[i] for i in successful_cases_indices]
            successful_outputs = all_objective_outputs[i]

            # Execute post-processing
            post_processed_outputs = objective.batch_post_process(
                successful_cases, successful_outputs, save_values=True
            )
            final_outputs.append(post_processed_outputs)

        return final_outputs

    def _objective_name_to_objective(
        self, objective_name: str
    ) -> Union[Objective, AggregateObjective]:
        """
        Args:
            objective_name (str): Objective name to map to the corresponding \
                objective

        Raises:
            ValueError: _description_

        Returns:
            Union[Objective, AggregateObjective]: An Objective
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
