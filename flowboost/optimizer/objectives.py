import logging
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from flowboost.openfoam.case import Case


class Objective:
    def __init__(
        self,
        name: str,
        minimize: bool,
        objective_function: Callable,
        objective_function_kwargs: dict = {},
        normalization_step: Optional[
            Literal["min-max", "yeo-johnson", "box-cox"]
        ] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """ An optimization objective that produces a quantifiable result for
        an OpenFOAM simulation. An Objective operates on an objective function,
        usings its evaluated value for informing the optimization process.

        The objective function is expected to return something that is, ideally,
        simply a scalar value. However, as you may define an arbitrary
        post-processing step for the Objective, operating on the output vector
        the Objective produces from all data points, the output can be arbitrary.

        More specifically:
        1) Objective consumes data (pd.DataFrame) for OpenFOAM Case
        2) `objective_function` operates on the data, returning an arbitrary
        value which is associated with the Case.
        3) Any values returned by `objective_function` can be later accesses as
        a vector, composed of the outputs of Objective for each OpenFOAM case.

        Additionally, you may pass arbitrary (constant) data to the function
        you have defined. This can be a DataFrame of reference data for a
        baseline simulation, or something completely different.

        Args:
            name (str): Name or description for this objective
            minimize (bool): Optimization objective: is the objective \
                function to be minimized, maximized, or something else?
            objective_function (Callable): An objective function for  \
                computing a quantified value. Function should accept at least \
                    one argument (Case), and potentially additional kwargs.
            objective_function_kwargs (dict, optional): Arbitrary additional \
                data passed to objective_function as kwargs. Defaults to {}.
            threshold (float, optional): Optional threshold value for MOO \
                objectives (see Ax documentation). Defaults to None.
        """
        self.name: str = name
        self.id: str = str(uuid4())

        # Goal for optimization task
        self.minimize: bool = minimize

        # Objective threshold: this is used by Ax during MOO, and is not
        # leveraged by all backends.
        #
        # For any objective function to be maximized, represents the _lowest_
        # objective value that is considered valuable in the context of
        # multi-objective optimization. Applies in the opposite manner for
        # objectives that are to be minimized.
        self.threshold: Optional[float] = threshold

        # User-provided function instance performing arbitrary computation
        # on a Pandas DataFrame, and returning some comparable result that
        # can be used during the optimization process.
        self.objective_function: Callable = objective_function

        # Additional data that gets passed to the objective function. This can
        # be anything, potentially an aggregated DataFrame for a baseline
        # simulation used for comparisons, or something more esoteric.
        self.objective_function_kwargs: dict = objective_function_kwargs

        # Any post-processing steps to be completed for this objective.
        # Post-processing is performed once _all_ Case objects have had the
        # objective_function evaluated: the post processing function thus gets
        # passed a vector of all results from the Objective outputs for each
        # Case.
        #
        # The steps are defined as a tuple of (Callable, arg1, arg2, ...),
        # where the args get unpacked and automatically passed to the
        # post-processing function during evaluation.
        self._post_processing_steps: list[tuple[Callable, Any]] = []

        # Data storage for each case that gets evaluated by this objective
        # _objective_output_data is raw, pre-post-processed data.
        self._objective_output_data: dict[Case, Any] = {}
        self._post_processed_data: dict[Case, Any] = {}

        if normalization_step:
            self.add_normalization_step(method=normalization_step)

    def add_normalization_step(
        self, method: Literal["min-max", "yeo-johnson", "box-cox"]
    ):
        """Simple high-level method of integrating a normalization step to the
        objective pipeline. The methods are from `sklearn.preprocessing`.

        The normalization step is performed once all points in the dataset have
        been evaluated using this Objective function. The input is, thus, the
        array of outputs this Objective has generated.

        For custom normalization steps not supported by this function, use the
        `Objective.attach_post_processing_step()` method, which accepts any
        arbitrary Callable + kwargs.

        Args:
            method (Literal): Normalization method to use: min-max or power-transform.

        Raises:
            ValueError: If method is not implemented
        """
        match method.lower():
            case "min-max":
                normalizer = MinMaxScaler()
            case "yeo-johnson":
                normalizer = PowerTransformer(method="yeo-johnson")
            case "box-cox":
                normalizer = PowerTransformer(method="box-cox")
            case _:
                logging.warning(
                    "For custom normalization methods, use `attach_post_processing_step`"
                )
                raise ValueError(f"Unsupported normalization method '{method}'")

        self.attach_post_processing_step(
            step=ScikitNormalizationStep(normalizer).evaluate
        )

    def attach_post_processing_step(self, step: Callable, **kwargs: Optional[dict]):
        """Add a new post-processing step to this Objective. The post-processing
        steps are evaluated in order of `Objective._post_processing_steps`: thus,
        the order they are added in matters.

        A typical use-case would be a normalization step.

        Args:
            step (Callable): _description_
        """
        self._post_processing_steps.append((step, kwargs))

    def execute_post_processing_steps(
        self, cases: list[Case], outputs: list[Any], save_output: bool = True
    ) -> dict[Case, Any]:
        if len(outputs) != len(cases):
            raise ValueError(
                f"Case count != output count: cases={cases}, outputs={outputs}"
            )

        if len(outputs) == 0:
            return {}

        # Iterate over post-processing steps
        for step, kwargs in self._post_processing_steps:
            # Apply the step to the entire array of outputs
            # Consider using try-except to handle errors gracefully
            try:
                outputs = step(outputs, **kwargs)
            except Exception as e:
                raise ValueError(f"Error applying post-processing step {step}: {e}")

        # Optionally, save post-processed outputs back to a dictionary keyed by Case
        # if specific per-case post-processed data is needed for further analysis
        out_dict: dict[Case, Any] = {}
        for case, processed_output in zip(cases, outputs):
            out_dict[case] = processed_output

        if save_output:
            self._post_processed_data = out_dict

        return out_dict

    def evaluate(self, case: Case, save_value: bool = True) -> Optional[Any]:
        """
        Evaluates the objective function for a case, returning the output
        value. If the objective function returns a None-type, the value is
        not saved to the objective to avoid post-processing failures.

        Args:
            case (Case): _description_
            save_value (bool, optional): _description_. Defaults to True.

        Returns:
            Any: Value of the objective function output for a successful case: \
                None indicates an evaluation failure.
        """
        if case.success is False:
            logging.warning("Case has been marked as failed: not evaluating objective")
            return None

        if self.objective_function_kwargs:
            out = self.objective_function(case, self.objective_function_kwargs)
        else:
            out = self.objective_function(case)

        if out is None:
            # Failures are not stored in the objective
            logging.warning(f"Objective function returned a None for {case}")
            logging.warning("Marking case as failed and not storing output!")
            case.mark_failed()
            return None

        if save_value:
            self._objective_output_data[case] = out

        return out

    def data_for_case(self, case: Case, post_processed: bool = True) -> Any:
        """Reads objective output for a Case, either from before or after
        the post-processing step.

        WARN: This function does _not_ re-apply the post-processing step.

        Args:
            case (Case): _description_
            post_processed (bool, optional): _description_. Defaults to False.

        Returns:
            Any: Objective output, None if not found.
        """
        if post_processed:
            return self._post_processed_data.get(case, None)

        return self._objective_output_data.get(case, None)

    def batch_evaluate(
        self, cases: list[Case], save_values: bool = False
    ) -> list[Optional[Any]]:
        # Evaluate for each case
        return [self.evaluate(case=case, save_value=save_values) for case in cases]

    def batch_post_process(
        self, cases: list[Case], outputs: list[Any], save_values: bool = True
    ) -> list[float]:
        out_d = self.execute_post_processing_steps(
            cases=cases, outputs=outputs, save_output=save_values
        )
        return list(out_d.values())


class AggregateObjective:
    # TODO make this a super of Objective
    """An AggregateObjective serves as an aggregator for multiple Objectives, in
    scenarios where the underlying optimizer either supports only one optimization
    objective, or an otherwise finite number of them.

    This wrapper allows for multiple, typically similar, objectives to be merged
    into one, scalar value, according to some weighting rules the user specifies.

    The underlying setup is identical to that of an Objective, with the exception
    of the post-processing steps being applied on a _tuple_ of all Objective
    outputs this AggregateObjective wraps within itself.
    """

    def __init__(
        self,
        name: str,
        minimize: bool,
        objectives: list[Objective],
        threshold: float,
        weights: Optional[list[float]] = None,
    ) -> None:
        """Initialize a new, aggregated AggregateObjective.

        Args:
            objectives (list[Objective]): Objectives to aggregate
            weights (Optional[list[float]], optional): Optional weights for aggregation step. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.name: str = name
        self.minimize: bool = minimize
        self.objectives: list[Objective] = objectives
        self.weights: list[float] = (
            weights if weights is not None else [1.0] * len(objectives)
        )
        self.threshold: Optional[float] = threshold

        self._post_processing_steps: list[tuple[Callable, dict[str, Any]]] = []

        # Data storage for each case that gets evaluated by this objective
        # _objective_output_data is raw, pre-post-processed data.
        self._objective_output_data: dict[Case, Any] = {}
        self._post_processed_data: dict[Case, Any] = {}

        if len(self.weights) != len(self.objectives):
            raise ValueError("Length of weights must match number of objectives")

    def attach_post_processing_step(self, step: Callable, **kwargs: dict[str, Any]):
        """Attach a post-processing step for this AggregateObjective. Post-processing
        steps are evaluated PRIOR to the aggregation step, operating on a list
        of tuples, each tuple representing the outputs for one processed Case's
        AggregateObjective output.

        The input should thus be list[tuple[Any * len(AggregateObjective.objectives)]]

        Args:
            step (Callable): Post-processing function
        """
        self._post_processing_steps.append((step, kwargs))

    def evaluate(self, case: Case) -> tuple:
        objective_outputs = [
            obj.evaluate(case, save_value=False) for obj in self.objectives
        ]

        self._objective_output_data[case] = objective_outputs
        return tuple(objective_outputs)

    def _evaluate_batch(self, cases: list[Case]) -> list[tuple]:
        all_outputs = [self.evaluate(case) for case in cases]
        return all_outputs

    def apply_batch_post_processing(self, all_outputs: list[tuple]) -> list[tuple]:
        # Apply post-processing steps suitable for batch-level processing
        # Example: normalization across each element of the tuples
        for step, kwargs in self._post_processing_steps:
            try:
                all_outputs = [step(output, **kwargs) for output in zip(*all_outputs)]
                # Transpose back if needed
                all_outputs = list(zip(*all_outputs))
            except Exception as e:
                raise ValueError(
                    f"Error applying batch post-processing step {step}: {e}"
                )
        return all_outputs

    def aggregate_outputs(self, processed_outputs: list[tuple]) -> list[float]:
        """Aggregate the post-processed tuple outputs into scalar values."""
        aggregated_values = [
            sum(
                float(output * weight)
                for output, weight in zip(obj_outputs, self.weights)
            )
            for obj_outputs in processed_outputs
        ]
        return aggregated_values

    def batch_process(self, cases: list[Case], save_values: bool = True) -> list[float]:
        # Step 1: Evaluate all objectives for all cases
        all_outputs = self._evaluate_batch(cases)

        # Step 2: Apply batch-level post-processing
        processed_outputs = self.apply_batch_post_processing(all_outputs)

        # Step 3: Aggregate post-processed outputs into scalar values
        outputs = self.aggregate_outputs(processed_outputs)

        for case, out in zip(cases, outputs):
            self._post_processed_data[case] = out

        return outputs

    def data_for_case(self, case: Case, post_processed: bool = True):
        if post_processed:
            return self._post_processed_data.get(case, None)

        return self._objective_output_data.get(case, None)


def objective_func_output(
    case: Case,
    objective: Objective,
    post_processed: bool = False,
    re_evaluate: bool = False,
):
    """Get the objective function (Objective) output for this case. The
    default setting, `post_processed=False`, returns the raw objective
    output, prior to any post-processing steps such as normalization.

    WARN: re-evaluating the data with post_processed=True can be extremely
    slow, depending on your post-processing flow. This typically entails
    evaluating the objective function for all cases.

    Args:
        objective (Objective): _description_
        post_processed (bool, optional): _description_. Defaults to False.
        re_evaluate (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if isinstance(objective, AggregateObjective):
        raise NotImplementedError(
            "Cannot evaluate AggregateObjectives through Case methods"
        )

    if re_evaluate and post_processed:
        # TODO: objective has all data, so just re-run postproc step?
        raise NotImplementedError(
            "Cannot re-evaluate while accessing post-processed data"
        )

    if re_evaluate:
        objective.evaluate(case)

    return objective.data_for_case(case, post_processed=post_processed)


class ScikitNormalizationStep:
    """
    Class serves as a very simple wrapper around Scikit's normalization
    functions that can be directly dropped into the objective's post-processing
    steps.
    """

    def __init__(self, scikit_normalizer) -> None:
        if not hasattr(scikit_normalizer, "fit_transform"):
            raise ValueError("Normalizer does not have fit_transform method")
        self.normalizer = scikit_normalizer

    def evaluate(self, input: list) -> list:
        return list(self.normalizer.fit_transform(np.array(input).reshape(-1, 1)))
