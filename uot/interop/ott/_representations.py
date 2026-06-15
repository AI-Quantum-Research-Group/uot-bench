"""OTT-JAX representation builders for the benchmarking pipeline.

Registers ``"ott_linear"``, ``"ott_quadratic"``, ``"ott_barycenter"``, and
``"ott_gw_barycenter"`` representation kinds so that OTT solver wrappers can
receive pre-built OTT problem objects from the runner, outside the timed solve
region.

Called once from :mod:`uot.interop.ott.__init__` as a side-effect import.
"""

from __future__ import annotations

from typing import Any

from uot.experiments.representations import register_representation
from uot.interop.ott._costs import cost_fn_for_name
from uot.interop.ott._problems import (
    measure_to_pointcloud,
    two_marginal_to_linear_problem,
    two_marginal_to_quadratic_problem,
)
from uot.problems.base_problem import Problem


def _build_ott_linear(
    problem: Problem,
    *,
    epsilon: float = 1e-2,
    scale_cost: float | str = 1.0,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    batch_size: int | None = None,
    **_: Any,
) -> Any:
    """Build an OTT LinearProblem for two-marginal / Sinkhorn problems."""
    marginals = problem.get_marginals()
    mu, nu = marginals[0], marginals[1]
    return two_marginal_to_linear_problem(
        mu, nu,
        cost_name=problem.cost_name,
        scale_cost=scale_cost,
        batch_size=batch_size,
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
    )


def _build_ott_quadratic(
    problem: Problem,
    *,
    epsilon: float = 1e-2,
    scale_cost: float | str = 1.0,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    fused_penalty: float = 0.0,
    gw_unbalanced_correction: bool = True,
    ranks: int | tuple[int, ...] = -1,
    tolerances: float | tuple[float, ...] = 1e-2,
    **_: Any,
) -> Any:
    """Build an OTT QuadraticProblem (GW / Fused-GW) for two-marginal problems."""
    marginals = problem.get_marginals()
    mu, nu = marginals[0], marginals[1]
    return two_marginal_to_quadratic_problem(
        mu, nu,
        cost_name=problem.cost_name,
        scale_cost=scale_cost,
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
        fused_penalty=fused_penalty,
        gw_unbalanced_correction=gw_unbalanced_correction,
        ranks=ranks,
        tolerances=tolerances,
    )


def _build_ott_barycenter(
    problem: Problem,
    *,
    epsilon: float = 1e-2,
    **_: Any,
) -> Any:
    """Build an OTT FreeBarycenterProblem for discrete Wasserstein barycenter."""
    import jax.numpy as jnp
    from ott.problems.linear.barycenter_problem import FreeBarycenterProblem

    marginals = problem.get_marginals()
    ys = []
    bs = []
    for m in marginals:
        pts, wts = m.as_point_cloud()
        ys.append(jnp.asarray(pts))
        bs.append(jnp.asarray(wts))

    cost_fn = cost_fn_for_name(problem.cost_name)
    return FreeBarycenterProblem(
        y=jnp.stack(ys),
        b=jnp.stack(bs),
        epsilon=epsilon,
        cost_fn=cost_fn,
    )


def _build_ott_gw_barycenter(
    problem: Problem,
    *,
    epsilon: float = 1e-2,
    scale_cost: float | str = 1.0,
    **_: Any,
) -> Any:
    """Build an OTT GWBarycenterProblem for Gromov–Wasserstein barycenter."""
    import jax.numpy as jnp
    from ott.problems.quadratic.gw_barycenter import GWBarycenterProblem

    marginals = problem.get_marginals()
    geoms = [
        measure_to_pointcloud(m, cost_name=problem.cost_name, scale_cost=scale_cost, epsilon=epsilon)
        for m in marginals
    ]
    weights = jnp.asarray([m.as_point_cloud()[1] for m in marginals])
    return GWBarycenterProblem(geoms=geoms, b=weights)


def _register_ott_representations() -> None:
    """Register all OTT-JAX representation kinds (called once at import time)."""
    register_representation("ott_linear", _build_ott_linear)
    register_representation("ott_quadratic", _build_ott_quadratic)
    register_representation("ott_barycenter", _build_ott_barycenter)
    register_representation("ott_gw_barycenter", _build_ott_gw_barycenter)
