import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from ax import Trial
from ax.service.ax_client import AxClient, ObjectiveProperties, TParameterization

from flowboost.openfoam.case import Case
from flowboost.optimizer.backend import Backend
from flowboost.optimizer.objectives import AggregateObjective, Objective
from flowboost.optimizer.search_space import Dimension


class AxBackend(Backend):
    def __init__(self, stateless: bool = True):
        """Craete an Ax client for Bayesian optimization.

        Args:
            stateless (bool, optional): Re-initialize client on every invokation; slow when using SAASBO. Defaults to True.
        """
        # Inherit main properties from Backend ABC
        super().__init__()

        # Main Ax state
        self.client: AxClient = AxClient()
        self.stateless: bool = stateless

        # Ax-specific options
        self.use_GPU: bool = True
        self.SAASBO: bool = False
        self.max_parallelism: Optional[int] = None
        self.initialization_trials: Optional[int] = None

        # Ax-specific constraints
        self._parameter_constraints: list[str] = []
        self._outcome_constraints: list[str] = []

        # Ax-specific features for noise
        self._SEM_by_objective: dict[str, Optional[float]] = {}
        self._trial_index_case_mapping: dict[Case, int] = {}

    def initialize(self):
        if not self.stateless:
            raise NotImplementedError("Stateful optimizer checkpoints not implemented")

        # Check objectives + dimensions
        self._verify_configuration()

        # Create a stateless optimizer client
        self.client.create_experiment(
            parameters=self._get_ax_search_space(),
            objectives=self._get_ax_objectives(),
            parameter_constraints=self._parameter_constraints,
            outcome_constraints=self._outcome_constraints,
            choose_generation_strategy_kwargs={
                "num_initialization_trials": self.initialization_trials,
                "use_saasbo": self.SAASBO,
                "verbose": True,
                "max_parallelism_override": self.max_parallelism,
            },
        )

        logging.info("Ax experiment initialized")

    def produce_state_snapshot(self, save_in: Path) -> Path:
        """
        Produces a JSON file of the client's settings and state, so that it
        can be restored.

        Args:
            save_in (Path): Directory to produce a JSON file in.

        Raises:
            ValueError: If configuration is invalid

        Returns:
            Path: Path to the saved, time-stamped file
        """
        if not self.offload_acquisition:
            raise ValueError(
                "Acquisition offload not enabled (backend.offload_acquisition)"
            )

        json_path = Path(save_in, "ax_snapshot.json")
        self.client.save_to_json_file(str(json_path))

        logging.info(f"Saved Ax state snapshot [{json_path}]")
        return json_path

    @classmethod
    def offloaded_acquisition(
        cls,
        model_snapshot: Path,
        data_snapshot: Path,
        num_trials: int,
        output_path: Path,
    ):
        # Ensure all files exist
        for p in (model_snapshot, data_snapshot):
            if not p.exists():
                raise FileNotFoundError(f"File not found [{model_snapshot}]")

        # Restore backend
        logging.info("Restoring Ax backend from state snapshot")
        ax = cls.restore_from_state_snapshot(model_snapshot)

        # Restore data
        with open(data_snapshot, "r") as json_f:
            data = json.load(json_f)

        # Ensure backend types match in data snapshot
        if data["optimizer"] != ax.type:
            raise ValueError(
                f"Optimizer mismatch in data: '{data['optimizer']}' != {ax.type}"
            )

        logging.info(f"Attaching from data snapshot ({data['created_at']})")

        # Attach finished trials to backend
        logging.info("Attaching finished trials")
        for case_name, data_dict in data["finished_cases"].items():
            _, idx = ax.client.attach_trial(
                parameters=data_dict["parametrization"], arm_name=case_name
            )

            # TODO support user's noise preference!
            raw_data = {
                obj_name: (outcome, 0.0)
                for obj_name, outcome in data_dict["objectives"].items()
            }

            ax.client.complete_trial(trial_index=idx, raw_data=raw_data)  # type: ignore

        # Attach pending trials to backend
        logging.info("Attaching pending trials")
        for case_name, data_dict in data["pending_cases"].items():
            _, idx = ax.client.attach_trial(
                parameters=data_dict["parametrization"], arm_name=case_name
            )

        # Run acquisition
        logging.info("Data loaded: running model update + acquisition")
        new_parametrizations, finished = ax.client.get_next_trials(
            max_trials=num_trials
        )

        if finished:
            # TODO add to json as a field
            logging.info("Optimization finished")

        logging.info(f"Received {len(new_parametrizations)} new trial(s) from Ax")

        snapshot = {
            "optimizer": ax.type,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "status_finished": finished,
            "parametrizations": new_parametrizations,
        }

        with open(output_path, "w") as json_f:
            json.dump(snapshot, json_f)

        logging.info(f"Wrote acquisition snapshot to file [{output_path}]")

    @classmethod
    def restore_from_state_snapshot(cls, from_file: Path) -> "AxBackend":
        """
        Restore Ax state from a json snapshot. Used for acquisition offloading.

        Args:
            from_file (Path): JSON file to restore from

        Raises:
            FileNotFoundError: If file not found

        Returns:
            AxBackend: Restored Ax client
        """
        ax = cls()
        if not from_file.exists():
            raise FileNotFoundError(f"Ax snapshot file not found: {from_file}")

        with open(from_file, "r") as json_f:
            json_d = json.load(json_f)
            ax.client.from_json_snapshot(json_d)

        logging.info("Restored Ax state from json snapshot")
        return ax

    def _re_initialize_client(self):
        """Re-initialize the Ax client in order to re-build the surrogate
        function.

        Raises:
            ValueError: If stateless mode is not active
        """
        if not self.stateless:
            raise ValueError("Re-initialize should not be called when stateless=False")
        self.client = AxClient()
        self.initialize()

    def set_objectives(self, objectives: list[Union[Objective, AggregateObjective]]):
        if isinstance(objectives, (Objective, AggregateObjective)):
            objectives = [objectives]
        self.objectives = objectives

    def set_search_space(self, dimensions: list[Dimension]):
        if isinstance(dimensions, Dimension):
            dimensions = [dimensions]

        self.dimensions = dimensions

    def set_parameter_constraints(self, constraints: list[str]):
        """Allows setting parameter constraints using string-based mathematical
        expressions for range-parameters.

        Ax does not support parameter constraints on logarithmic dimensions.

        Examples: `"x3 >= x4"` or `"-x3 + 2*x4 - 3.5*x5 >= 2"`

        See Ax's documentation for `ParameterConstraint` for additional docs.

        Args:
            constraints (list[str]): Parameter constraints to apply on optimizer.
        """
        self._parameter_constraints = constraints

    def set_outcome_constraints(self, constraints: list[str]):
        self._outcome_constraints = constraints

    def attach_pending_cases(self, cases: list[Case]):
        if cases:
            logging.info(f"Attaching {len(cases)} pending case(s) to the model")

        # Ensure all cases have been attached as trials to Ax
        self._ensure_attached(cases)

    def attach_failed_cases(self, cases: list[Case]):
        if not cases:
            return

        self._ensure_attached(cases)

        # Perform attach
        for case in cases:
            """
            "The difference between abandonment and failure is that the FAILED
            state is meant to express a possibly transient or retryable error,
            so trials in that state may be re-run and arm(s) in them may be
            resuggested by Ax models to be added to new trials."

            https://ax.dev/api/core.html#ax.core.base_trial.TrialStatus
            """
            if case not in self._trial_index_case_mapping:
                raise ValueError(
                    f"Case not in trial mapping, cannot mark failed [{case}]"
                )

            trial_idx = self._trial_index_case_mapping[case]
            trial = self.client.get_trial(trial_idx)

            if not self._can_abandon_trial(trial):
                continue

            logging.info(f"Marking as abandoned: {case}")
            self.client.abandon_trial(trial_idx)

    def tell_and_ask(
        self, cases: list[Case], max_cases: int
    ) -> list[dict[Dimension, Any]]:
        """
        A shorthand for first calling tell() and then ask().

        Args:
            cases (list[Case]): List of evaluated cases
            max_cases (int): Maximum number of cases to yield. May be less \
                during the initial Sobol' sequence seeding phase.

        Returns:
            list[dict[Dimension, Any]]: List of dictionaries, each dictionary \
                having Dimension-value kv-pairs ready to be generated into \
                a new case. One dictionary per case.
        """
        self.tell(cases)
        return self.ask(max_cases=max_cases)

    def _ask(self, max_cases: int) -> dict[int, TParameterization]:
        new_parametrizations, finished = self.client.get_next_trials(
            max_trials=max_cases
        )

        if finished:
            logging.info("🎉 Optimization finished")
            sys.exit("Optimization finished: can not proceed further")

        logging.info(f"Received {len(new_parametrizations)} new trial(s) from Ax")

        # TODO handle case where 0 trials returned and we don't want to ask for
        # new trials. Can be at the end of Sobol seeding phase.
        if len(new_parametrizations) == 0:
            raise ValueError("Cannot proceed: no trials received (TODO fix)")

        # Convert the parametrizations back to be mapped by Dimensions
        # TODO: this discards the trial ID!
        return new_parametrizations

    def tell(self, cases: list[Case]):
        """
        Inform Ax of evaluated cases, updating the surrogate function and
        preparing the optimizer for the next acquisition event.

        The cases are expected to have been already evaluated by the
        objective functions.

        Args:
            cases (list[Case]): List of evaluated cases
        """
        if self.stateless:
            self._re_initialize_client()
        else:
            raise NotImplementedError("Stateful tell() not implemented")

        # TODO handle failure criteria here

        # Next, get the objective function outputs for each case
        logging.info("Retrieving objective function outputs")
        case_objective_outputs: dict[Case, dict] = {
            c: c.objective_function_outputs(self.objectives) for c in cases
        }

        logging.info(f"Ax search space: {self._get_ax_search_space()}")

        # Ensure all cases have been attached as trials to Ax
        self._ensure_attached(cases)

        # Complete case trials
        self._complete_trials(cases, case_objective_outputs)

    def get_model_prediction(self, case: Case):
        if case not in self._trial_index_case_mapping:
            return None

        parametrization = self.client.get_trial_parameters(
            trial_index=self._trial_index_case_mapping[case]
        )

        # Get model's predictions for this parameterization
        logging.info("Getting model predictions for parameterization")

        try:
            prediction = self.client.get_model_predictions_for_parameterizations(
                parameterizations=[parametrization]
            )[0]
        except NotImplementedError:
            # Predictions are not implemented for Sobol generation phase: OK
            return None
        except Exception:
            logging.exception(
                f"Could not get model predictions for parameterization={parametrization}"
            )
            return None

        return prediction

    def _post_process_suggestion_parametrizations(
        self, parametrizations: dict[int, TParameterization]
    ) -> list[dict[Dimension, Any]]:
        """
        Converts the list of string-keyed dictionaries, mapping search space
        dimensions to values to evaluate, to be keyed by Dimension objects.

        Args:
            parametrizations (dict): Trial indices mapped to parametrization \
                dictionaries.

        Returns:
            list[dict[Dimension, Any]]: Dimension-keyed list of \
                parametrizations.
        """
        processed = []

        # For each parametrization, convert the str-keys to be Dimensions
        for trial_index, parametrization in parametrizations.items():
            new_suggestion = {}
            for str_key, val in parametrization.items():
                dim = self._dim_name_to_dimension(str_key)
                new_suggestion[dim] = val

            processed.append(new_suggestion)

        return processed

    def _ensure_attached(self, cases: list[Case]):
        """
        Attach creates new Trials from the Case objects by linking their
        points in the search space dimensions to the optimizer. Does not allow
        for cases to be attached more than once.

        Attachment can be done before the cases have been evaluated.

        Args:
            cases (list[Case]): Cases to attach as trials.
            parametrizations (dict[Case, dict], optional): Pre-computed \
                parametrizations for each case. Parametrizations are dicts, \
                mapping all search space dimension names to a value.
        """
        for case in cases:
            if case in self._trial_index_case_mapping:
                logging.info(f"Case already attached in _ensure_attached, {case}")
                continue

            # Generate parametrizations: these describe the search space for Ax
            p = case.parametrize_configuration(self.dimensions)

            logging.info(f"Attaching c={case.name}, p={p}")
            _, idx = self.client.attach_trial(parameters=p, arm_name=case.name)

            # TODO if we were to run in stateful mode, we'd stash the index
            self._trial_index_case_mapping[case] = idx

    def _can_abandon_trial(self, trial: Trial) -> bool:
        if trial is None:
            return False

        # Check if trial is already complete, failed, or abandoned
        if (
            trial.status.is_terminal
            or trial.status.is_abandoned
            or trial.status.is_failed
            or trial.status.is_completed
        ):
            logging.warning(
                f"Trial {trial} is already terminal/abandoned/failed/completed"
            )
            return False

        return True

    def _complete_trials(
        self, cases: list[Case], objective_f_outputs: dict[Case, dict]
    ):
        for case in cases:
            if case not in self._trial_index_case_mapping:
                raise ValueError(
                    f"Case not in trial index mapping, cannot complete trial [{case}]"
                )

            # Construct tuple-based result for case: see Ax TEvaluationOutcome
            raw_data = {
                obj_name: (outcome, self._SEM_by_objective.get(obj_name, 0.0))
                for obj_name, outcome in objective_f_outputs[case].items()
            }

            self.client.complete_trial(
                trial_index=self._trial_index_case_mapping[case],
                raw_data=raw_data,  # type: ignore
            )

    def _get_ax_objectives(self):
        ax_objectives = {}
        for objective in self.objectives:
            ax_objectives[objective.name] = ObjectiveProperties(
                minimize=objective.minimize, threshold=objective.threshold
            )

        return ax_objectives

    def _dim_to_Ax_parameter_dict(self, dim: Dimension):
        d = {"name": dim.name, "type": dim.type, "value_type": dim.value_type}

        match dim.type:
            case "range":
                d["bounds"] = dim.bounds
                d["log_scale"] = dim.log_scale
                d["digits"] = dim.digits
            case "choice":
                d["values"] = dim.values
                d["is_ordered"] = dim.is_ordered
            case "fixed":
                if not dim.values:
                    raise ValueError("Fixed dimension must have a specified value")

                d["value"] = dim.values[0]
            case _:
                raise ValueError(f"Dimension type '{dim.type} not supported'")

        return d

    def _get_ax_search_space(self):
        return [self._dim_to_Ax_parameter_dict(d) for d in self.dimensions]
