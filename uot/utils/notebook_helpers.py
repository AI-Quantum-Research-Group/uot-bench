from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

from uot.data.measure import BaseMeasure
from uot.problems.problem_generator import ProblemGenerator
from uot.problems.barycenter_problem import BarycenterProblem


def one_problem(generator: ProblemGenerator, **kwargs) -> Any:
    """
    Convenience wrapper to grab a single problem instance from a generator.
    """
    return next(generator.generate(**kwargs))


def stack_measure_weights(
        measures: Iterable[BaseMeasure],
        include_zeros: bool = True,
) -> jnp.ndarray:
    """
    Convert a list of measures into a stacked (M, N) weight array.
    """
    weights = []
    for measure in measures:
        if hasattr(measure, "for_grid_solver"):
            _, w = measure.for_grid_solver()
            w = jnp.asarray(w).reshape(-1)
        elif hasattr(measure, "weights_nd"):
            w = jnp.asarray(measure.weights_nd).reshape(-1)
        else:
            _, w = measure.to_discrete(include_zeros=include_zeros)
            w = jnp.asarray(w).reshape(-1)
        weights.append(w)
    return jnp.stack(weights, axis=0)


def barycenter_inputs(
        problem: BarycenterProblem,
        *,
        support_mode: str = "problem",
        shared_mode: str = "union",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
        return_support: bool = False,
) -> (tuple[list[BaseMeasure],
      jnp.ndarray,
      Any, jnp.ndarray]):
    """
    Extract (measures, lambdas, cost, stacked weights) from a barycenter problem.
    """
    measures = problem.get_marginals()
    if support_mode == "shared":
        support, meas_array, cost, lambdas = problem.shared_support_inputs(
            mode=shared_mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        if return_support:
            return measures, jnp.asarray(lambdas), cost, jnp.asarray(meas_array), support
        return measures, jnp.asarray(lambdas), cost, jnp.asarray(meas_array)

    if support_mode != "problem":
        raise ValueError("support_mode must be 'problem' or 'shared'")

    lambdas = jnp.asarray(problem.lambdas())
    cost = problem.get_costs()[0]
    meas_array = stack_measure_weights(measures, include_zeros=include_zeros)
    if return_support:
        return measures, lambdas, cost, meas_array, None
    return measures, lambdas, cost, meas_array
