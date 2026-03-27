from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp

from uot.data.measure import BaseMeasure, GridMeasure
from uot.problems.barycenter_problem import BarycenterProblem
from uot.problems.problem_generator import ProblemGenerator


def one_problem(generator: ProblemGenerator, **kwargs) -> Any:
    """Compatibility wrapper around `ProblemGenerator.one`."""
    return generator.one(**kwargs)


def stack_measure_weights(
    measures: Iterable[BaseMeasure],
    include_zeros: bool = True,
) -> jnp.ndarray:
    weights = []
    for measure in measures:
        if isinstance(measure, GridMeasure):
            _, w = measure.as_grid()
            w = jnp.asarray(w).reshape(-1)
        else:
            _, w = measure.as_point_cloud(include_zeros=include_zeros)
            w = jnp.asarray(w).reshape(-1)
        weights.append(w)
    return jnp.stack(weights, axis=0)


def barycenter_inputs(
    problem: BarycenterProblem,
    *,
    support_mode: str = "problem",
    shared_mode: str = "same",
    include_zeros: bool = True,
    atol: float = 0.0,
    rtol: float = 0.0,
    return_support: bool = False,
):
    measures = problem.solver_inputs(include_cost=False).marginals
    if support_mode == "shared":
        inputs = problem.point_cloud_inputs(
            shared_support=shared_mode,
            include_cost=True,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        if return_support:
            return measures, jnp.asarray(inputs.lambdas), inputs.cost, jnp.asarray(inputs.weights), inputs.support
        return measures, jnp.asarray(inputs.lambdas), inputs.cost, jnp.asarray(inputs.weights)

    if support_mode != "problem":
        raise ValueError("support_mode must be 'problem' or 'shared'")

    inputs = problem.solver_inputs(include_cost=True)
    cost = inputs.costs[0] if inputs.costs else None
    meas_array = stack_measure_weights(measures, include_zeros=include_zeros)
    if return_support:
        return measures, jnp.asarray(inputs.lambdas), cost, meas_array, None
    return measures, jnp.asarray(inputs.lambdas), cost, meas_array


def two_marginal_inputs(
    problem,
    *,
    include_cost: bool = True,
    include_zeros: bool = True,
):
    inputs = problem.point_cloud_inputs(
        shared_support="same",
        include_cost=include_cost,
        include_zeros=include_zeros,
    )
    return inputs.support, inputs.weights, inputs.cost
