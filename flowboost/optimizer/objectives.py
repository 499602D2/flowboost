import logging
import math
from typing import Any, Callable, Optional, Union
from uuid import uuid4

from flowboost.openfoam.case import Case
from flowboost.optimizer.scalars import coerce_objective_scalar


class _EvaluatedMetric:
    """Cache + lookup machinery shared by every class that produces a scalar
    metric value from a Case (Objective, Constraint, ScalarizedObjective).
    Subclasses must implement `evaluate()`; `metric_values_for_case()` may be
    overridden when the class contributes more than one Ax metric.
    Not part of the public API.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.id: str = str(uuid4())
        self._values: dict[Case, float] = {}

    def evaluate(self, case: Case, save_value: bool = True) -> Optional[float]:
        raise NotImplementedError(
            "_EvaluatedMetric subclasses must implement evaluate()"
        )

    def batch_evaluate(
        self, cases: list[Case], save_values: bool = True
    ) -> list[Optional[float]]:
        return [self.evaluate(case=case, save_value=save_values) for case in cases]

    def data_for_case(self, case: Case) -> Optional[float]:
        """Return the cached evaluated value for a case, or None if not seen."""
        return self._values.get(case)

    def metric_values_for_case(self, case: Case) -> dict[str, float]:
        """Return {metric_name: value} contributed for `case`. Empty dict means
        no value available (case failed or not yet evaluated)."""
        v = self.data_for_case(case)
        return {} if v is None else {self.name: v}


class _CallableMetric(_EvaluatedMetric):
    """Metric whose value comes from calling a user-supplied function on the
    Case. The single-function-call evaluator factors out the shared machinery
    between Objective and Constraint. Not part of the public API.
    """

    def __init__(
        self,
        name: str,
        objective_function: Callable,
        objective_function_kwargs: Optional[dict[str, Any]] = None,
        static_transform: Optional[Callable[[float], float]] = None,
    ) -> None:
        super().__init__(name)
        self.objective_function: Callable = objective_function
        self.objective_function_kwargs: dict[str, Any] = dict(
            objective_function_kwargs or {}
        )
        self.static_transform: Optional[Callable[[float], float]] = static_transform

    def _value_label(self) -> str:
        return f"Metric '{self.name}' output"

    def evaluate(self, case: Case, save_value: bool = True) -> Optional[float]:
        if case.success is False:
            logging.warning("Case has been marked as failed: not evaluating metric")
            return None

        if self.objective_function_kwargs:
            raw = self.objective_function(case, **self.objective_function_kwargs)
        else:
            raw = self.objective_function(case)

        if raw is None:
            logging.warning(f"Metric function returned a None for {case}")
            logging.warning("Marking case as failed and not storing output!")
            case.mark_failed()
            return None

        value = self.static_transform(raw) if self.static_transform else raw
        value = coerce_objective_scalar(value, label=self._value_label())

        if save_value:
            self._values[case] = value

        return value


def _validate_bounds(gte: Optional[float], lte: Optional[float], context: str) -> None:
    if gte is not None and not math.isfinite(gte):
        raise ValueError(
            f"{context}: gte={gte!r} must be a finite number. Use `None` to "
            f"indicate 'no lower bound' — NaN and ±inf are not accepted."
        )
    if lte is not None and not math.isfinite(lte):
        raise ValueError(
            f"{context}: lte={lte!r} must be a finite number. Use `None` to "
            f"indicate 'no upper bound' — NaN and ±inf are not accepted."
        )
    if gte is not None and lte is not None:
        if gte > lte:
            raise ValueError(
                f"{context}: gte={gte} > lte={lte} defines an empty feasible range."
            )
        if gte == lte:
            raise ValueError(
                f"{context}: gte={gte} == lte={lte} defines a single-point "
                f"feasible range, which has no meaning for a soft outcome "
                f"constraint on a continuous GP surrogate. For a tight "
                f"tolerance around X, use gte=X-eps, lte=X+eps."
            )


class Objective(_CallableMetric):
    """A named optimization objective that evaluates a Case and returns a
    scalar value.

    Outcome standardization is the optimizer backend's job (Ax wraps every
    metric in StandardizeY + BoTorch's Standardize). Don't add normalization
    here. If your measurement is naturally non-linear (log-distributed drag,
    say), pass `static_transform=math.log` — a pure float→float function,
    applied per evaluation, no fitting, no state.

    Objectives can carry a soft outcome bound via `gte`/`lte`. The bound
    applies to the same measurement the objective optimizes (e.g. "minimize
    drag, but also keep drag ≤ 0.05"). Under the hood this becomes a derived
    tracking metric named ``{name}__bound`` with an `OutcomeConstraint`
    pointing to it; Ax can't attach a constraint to an objective metric
    directly, so the derived name is the workaround. For a bound on a metric
    that isn't an optimization target, use a standalone `Constraint`.
    """

    BOUND_METRIC_SUFFIX = "__bound"

    def __init__(
        self,
        name: str,
        minimize: bool,
        objective_function: Callable,
        objective_function_kwargs: Optional[dict[str, Any]] = None,
        threshold: Optional[float] = None,
        static_transform: Optional[Callable[[float], float]] = None,
        gte: Optional[float] = None,
        lte: Optional[float] = None,
    ) -> None:
        """An optimization objective for an OpenFOAM simulation.

        Args:
            name: Identifier for the objective. Must be unique across all
                metrics in the experiment (including those wrapped in a
                ScalarizedObjective and any Constraints). If `gte` or `lte`
                is set, the derived bound metric name ``{name}__bound`` must
                also not collide with another registered metric.
            minimize: True to minimize, False to maximize.
            objective_function: Callable taking a Case (and optional kwargs)
                and returning a scalar. Returning None marks the case failed.
            objective_function_kwargs: Extra kwargs passed through to
                `objective_function` on every call.
            threshold: Optional MOO objective threshold (see Ax docs). Only
                used by backends that support multi-objective Pareto fronts.
            static_transform: Optional pure float→float transform applied to
                each scalar output. Use for domain-driven re-expressions
                (e.g. `math.log` for a log-distributed quantity). Not for
                normalization — the backend handles that.
            gte: Optional soft lower bound on this objective's value
                (absolute, not relative). Steers the optimizer away from
                regions where the bound is violated without hard-rejecting.
            lte: Optional soft upper bound. Same semantics as `gte`.
        """
        _validate_bounds(gte, lte, context=f"Objective {name!r}")

        super().__init__(
            name=name,
            objective_function=objective_function,
            objective_function_kwargs=objective_function_kwargs,
            static_transform=static_transform,
        )
        self.minimize: bool = minimize
        self.threshold: Optional[float] = threshold
        self.gte: Optional[float] = gte
        self.lte: Optional[float] = lte

    @property
    def has_bounds(self) -> bool:
        return self.gte is not None or self.lte is not None

    @property
    def bound_metric_name(self) -> str:
        """Name of the derived tracking metric that carries this objective's
        soft bound at the Ax layer. Only meaningful when `has_bounds` is True.
        """
        return f"{self.name}{self.BOUND_METRIC_SUFFIX}"

    def _value_label(self) -> str:
        return f"Objective '{self.name}' output"

    def metric_values_for_case(self, case: Case) -> dict[str, float]:
        """For a bounded Objective, emit both the objective metric and the
        derived bound metric in the output dict. Ax expects raw_data to cover
        every registered metric, including the tracking metric that holds the
        bound.
        """
        v = self.data_for_case(case)
        if v is None:
            return {}
        out = {self.name: v}
        if self.has_bounds:
            out[self.bound_metric_name] = v
        return out


class Constraint(_CallableMetric):
    """A measurable quantity used to *steer* the optimizer rather than be
    optimized for.

    A Constraint registers a tracking metric with the backend (Ax: via
    `experiment.add_tracking_metric`) and attaches a soft outcome constraint
    to it. The optimizer evaluates the constraint metric on every case and is
    discouraged from proposing regions where the bound is violated — but it
    is not a hard reject; the backend's acquisition function applies a
    penalty.

    Use this when you have a measurement that should keep the optimizer in a
    sensible region of the design space without it being one of the things
    you're trying to push up or down (e.g. "pressure must stay ≥ 45.0 while
    we minimize heat loss and heat flux").

    Bounds are absolute. Ax's underlying `OutcomeConstraint` defaults to
    relative bounds (interpreted as a percentage relative to a baseline arm),
    which would silently change the meaning of every `gte`/`lte` users supply
    here; flowboost intentionally inverts that default. Relative bounds are
    not currently exposed — open an issue if you need them.
    """

    def __init__(
        self,
        name: str,
        objective_function: Callable,
        gte: Optional[float] = None,
        lte: Optional[float] = None,
        objective_function_kwargs: Optional[dict[str, Any]] = None,
        static_transform: Optional[Callable[[float], float]] = None,
    ) -> None:
        """A constraint-only metric.

        Args:
            name: Identifier for the constraint metric. Must be unique across
                all metrics in the experiment.
            objective_function: Callable taking a Case (and optional kwargs)
                and returning the constraint metric's scalar value.
            gte: Optional lower bound (absolute, not relative). At least one
                of `gte` or `lte` must be supplied — a Constraint with no
                bound has no effect.
            lte: Optional upper bound (absolute, not relative).
            objective_function_kwargs: Extra kwargs passed to the function.
            static_transform: Optional pure float→float transform applied to
                each output before the bound is checked.
        """
        if gte is None and lte is None:
            raise ValueError(
                f"Constraint {name!r} requires at least one of `gte` or `lte`. "
                f"Without a bound, a constraint has no effect on the optimizer."
            )
        _validate_bounds(gte, lte, context=f"Constraint {name!r}")

        super().__init__(
            name=name,
            objective_function=objective_function,
            objective_function_kwargs=objective_function_kwargs,
            static_transform=static_transform,
        )
        self.gte: Optional[float] = gte
        self.lte: Optional[float] = lte

    def _value_label(self) -> str:
        return f"Constraint '{self.name}' output"


class ScalarizedObjective(_EvaluatedMetric):
    """A weighted sum of multiple Objectives, optimized as a single scalar.

    Direction is encoded by signed weights: a negative weight flips the
    contribution of an inner objective relative to the formula. The outer
    `minimize` flag then says whether to minimize or maximize the resulting
    sum.

    Example: maximize lift-to-drag ratio
        lift = Objective(name="lift", minimize=False, ...)
        drag = Objective(name="drag", minimize=True,  ...)
        ratio = ScalarizedObjective(
            name="LiftToDrag", minimize=False,
            objectives=[lift, drag], weights=[0.7, -0.3],
        )

    With a backend that supports it (e.g. AxBackend), each inner objective
    becomes its own metric in the surrogate, gets standardized independently,
    and the weighted sum is computed at the acquisition step. With backends
    that don't, scalarization happens at the flowboost layer.
    """

    def __init__(
        self,
        name: str,
        minimize: bool,
        objectives: list[Objective],
        weights: Optional[list[float]] = None,
    ) -> None:
        if not objectives:
            raise ValueError("ScalarizedObjective requires at least one objective")

        for i, obj in enumerate(objectives):
            if isinstance(obj, ScalarizedObjective):
                raise TypeError(
                    f"ScalarizedObjective.objectives[{i}] is itself a "
                    f"ScalarizedObjective. Nested scalarization is not "
                    f"supported — flatten into one ScalarizedObjective with "
                    f"the combined inner objectives and weights."
                )
            if not isinstance(obj, Objective):
                raise TypeError(
                    f"ScalarizedObjective.objectives[{i}] must be an "
                    f"Objective, got {type(obj).__name__!r}. Constraints and "
                    f"other metric types cannot be scalarized."
                )

        names = [o.name for o in objectives]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"ScalarizedObjective has duplicate inner objective names: "
                f"{duplicates}. Each inner metric must be uniquely named — "
                f"Ax registers each as its own metric on the experiment. "
                f"(If you intended two distinct terms, give them different "
                f"names; if you passed the same Objective instance twice, "
                f"drop the duplicate.)"
            )

        if any(o.threshold is not None for o in objectives):
            raise ValueError(
                "Inner Objective.threshold is not honored when wrapped in a "
                "ScalarizedObjective: scalarization collapses to a single "
                "objective so per-metric Pareto thresholds don't apply. Drop "
                "the threshold or use plain Objectives in MOO mode instead."
            )

        super().__init__(name)
        self.minimize: bool = minimize
        self.objectives: list[Objective] = objectives
        self.weights: list[float] = (
            list(weights) if weights is not None else [1.0] * len(objectives)
        )

        if len(self.weights) != len(self.objectives):
            raise ValueError("Length of weights must match number of objectives")

        for i, w in enumerate(self.weights):
            if not math.isfinite(w):
                raise ValueError(
                    f"ScalarizedObjective weight at index {i} is not finite "
                    f"({w!r}). Weights must be finite floats — NaN and ±inf "
                    f"are rejected."
                )

        if all(w == 0.0 for w in self.weights):
            raise ValueError(
                "ScalarizedObjective has no non-zero weights; the "
                "scalarization has no signal. Drop the wrapper, or set at "
                "least one non-zero weight."
            )

        # Each inner objective declared a direction (minimize or not). The
        # weight's sign in a scalarization implies a direction too: with
        # outer minimize=M, a positive weight means "this term's direction
        # equals M", a negative weight means "opposite of M". Reject when
        # those disagree — the optimization is internally inconsistent and
        # Ax would otherwise reject it deep in the surrogate setup. Weight
        # zero is a direction-less term; warn and skip the direction check
        # so users can temporarily mute a contribution during tuning.
        for obj, w in zip(self.objectives, self.weights):
            if w == 0.0:
                logging.warning(
                    f"ScalarizedObjective {self.name!r}: inner objective "
                    f"{obj.name!r} has weight 0 — its contribution is "
                    f"disabled. Set a non-zero weight to re-enable."
                )
                continue
            implied_minimize = self.minimize if w > 0 else not self.minimize
            if obj.minimize != implied_minimize:
                obj_dir = "minimize" if obj.minimize else "maximize"
                implied_dir = "minimize" if implied_minimize else "maximize"
                raise ValueError(
                    f"Inconsistent direction for inner objective {obj.name!r}: "
                    f"weight={w:+g} with outer minimize={self.minimize} implies "
                    f"'{implied_dir}', but the objective declares '{obj_dir}'. "
                    f"Flip the sign of the weight or the inner objective's "
                    f"minimize flag."
                )

        # Threshold is single-objective-mode only (no Pareto front to bound).
        # Kept as a None attribute so call sites that read it don't break.
        self.threshold: Optional[float] = None

    def evaluate(self, case: Case, save_value: bool = True) -> Optional[float]:
        """Evaluate every inner objective and return the signed weighted sum.

        Inner evaluation results are cached on each inner Objective (so
        `inner.data_for_case(case)` works after this call). If any inner
        returns None, the whole scalarization is None.
        """
        if case.success is False:
            logging.warning("Case has been marked as failed: not evaluating objective")
            return None

        inner_values = [
            obj.evaluate(case, save_value=save_value) for obj in self.objectives
        ]
        if any(v is None for v in inner_values):
            return None

        scalar = sum(w * v for w, v in zip(self.weights, inner_values))
        scalar = coerce_objective_scalar(
            scalar, label=f"ScalarizedObjective '{self.name}' output"
        )

        if save_value:
            self._values[case] = scalar

        return scalar

    def metric_values_for_case(self, case: Case) -> dict[str, float]:
        """Return per-inner metric values for each inner objective, including
        the derived bound alias ``{name}__bound`` for any inner carrying
        `gte`/`lte`. Delegates to each inner's own `metric_values_for_case`
        so the emit logic for bounds lives in one place.

        These are the per-metric values backends with native scalarization
        (e.g. Ax) expect: Ax does the weighted sum itself at the acquisition
        step, and reads bound tracking metrics separately for any
        OutcomeConstraints. Overrides the base implementation, which would
        return the scalar under the ScalarizedObjective's own name.
        """
        out: dict[str, float] = {}
        for obj in self.objectives:
            inner_values = obj.metric_values_for_case(case)
            if not inner_values:
                return {}
            out.update(inner_values)
        return out


def objective_func_output(
    case: Case,
    objective: Union[Objective, "ScalarizedObjective"],
    re_evaluate: bool = False,
) -> Optional[float]:
    """Read the cached output of `objective` for `case`.

    For ScalarizedObjective this returns the scalarized value. To inspect
    individual inner contributions, iterate `objective.objectives` and call
    `inner.data_for_case(case)` on each.
    """
    if re_evaluate:
        objective.evaluate(case)
    return objective.data_for_case(case)
