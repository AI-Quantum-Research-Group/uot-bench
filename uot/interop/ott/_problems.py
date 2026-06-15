"""Convert uot measures and problems into OTT-JAX geometry/problem objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from uot.data.measure import BaseMeasure, GridMeasure, PointCloudMeasure
from uot.interop.ott._costs import cost_fn_for_name

if TYPE_CHECKING:
    from ott.geometry.geometry import Geometry
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.problems.quadratic.quadratic_problem import QuadraticProblem


def measure_to_pointcloud(
    mu: BaseMeasure,
    *,
    cost_name: str | None = None,
    scale_cost: float | str = 1.0,
    batch_size: int | None = None,
    epsilon: float = 1e-2,
) -> "Geometry":
    """Convert a uot measure to an OTT PointCloud geometry (self-cost).

    Returns a ``PointCloud(mu.points, mu.points, ...)`` for use as an
    intra-space geometry in GW problems.
    """
    from ott.geometry.pointcloud import PointCloud

    pts, _ = mu.as_point_cloud()
    pts = jnp.asarray(pts)
    cost_fn = cost_fn_for_name(cost_name)
    return PointCloud(
        pts,
        pts,
        cost_fn=cost_fn,
        scale_cost=scale_cost,
        batch_size=batch_size,
        epsilon=epsilon,
    )


def two_measures_to_pointcloud(
    mu: BaseMeasure,
    nu: BaseMeasure,
    *,
    cost_name: str | None = None,
    scale_cost: float | str = 1.0,
    batch_size: int | None = None,
    epsilon: float = 1e-2,
) -> "Geometry":
    """Convert a pair of uot measures to an OTT PointCloud geometry."""
    from ott.geometry.pointcloud import PointCloud

    x, _ = mu.as_point_cloud()
    y, _ = nu.as_point_cloud()
    x, y = jnp.asarray(x), jnp.asarray(y)
    cost_fn = cost_fn_for_name(cost_name)
    return PointCloud(
        x,
        y,
        cost_fn=cost_fn,
        scale_cost=scale_cost,
        batch_size=batch_size,
        epsilon=epsilon,
    )


def measures_to_grid(
    mu: GridMeasure,
    nu: GridMeasure,
    *,
    epsilon: float = 1e-2,
) -> "Geometry":
    """Convert a pair of GridMeasures to an OTT Grid geometry."""
    from ott.geometry.grid import Grid

    mu.check_compatible(nu)
    axes, _ = mu.as_grid()
    return Grid(x=axes, epsilon=epsilon)


def two_marginal_to_linear_problem(
    mu: BaseMeasure,
    nu: BaseMeasure,
    *,
    cost_name: str | None = None,
    scale_cost: float | str = 1.0,
    batch_size: int | None = None,
    epsilon: float = 1e-2,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
) -> "LinearProblem":
    """Build an OTT LinearProblem from two uot BaseMeasures."""
    from ott.problems.linear.linear_problem import LinearProblem

    geom = two_measures_to_pointcloud(
        mu, nu,
        cost_name=cost_name,
        scale_cost=scale_cost,
        batch_size=batch_size,
        epsilon=epsilon,
    )
    a = jnp.asarray(mu.weights if hasattr(mu, "weights") else mu.as_point_cloud()[1])
    b = jnp.asarray(nu.weights if hasattr(nu, "weights") else nu.as_point_cloud()[1])
    return LinearProblem(geom, a=a, b=b, tau_a=tau_a, tau_b=tau_b)


def two_marginal_to_quadratic_problem(
    mu: BaseMeasure,
    nu: BaseMeasure,
    *,
    cost_name: str | None = None,
    scale_cost: float | str = 1.0,
    epsilon: float = 1e-2,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    fused_penalty: float = 0.0,
    gw_unbalanced_correction: bool = True,
    ranks: int | tuple[int, ...] = -1,
    tolerances: float | tuple[float, ...] = 1e-2,
) -> "QuadraticProblem":
    """Build an OTT QuadraticProblem (Gromov–Wasserstein) from two measures.

    The intra-space geometries are built from each measure's self-cost.
    If ``fused_penalty > 0``, the cross-space geometry is also included.
    """
    from ott.problems.quadratic.quadratic_problem import QuadraticProblem

    geom_xx = measure_to_pointcloud(
        mu, cost_name=cost_name, scale_cost=scale_cost, epsilon=epsilon,
    )
    geom_yy = measure_to_pointcloud(
        nu, cost_name=cost_name, scale_cost=scale_cost, epsilon=epsilon,
    )
    geom_xy = None
    if fused_penalty > 0.0:
        geom_xy = two_measures_to_pointcloud(
            mu, nu, cost_name=cost_name, scale_cost=scale_cost, epsilon=epsilon,
        )

    a = jnp.asarray(mu.weights if hasattr(mu, "weights") else mu.as_point_cloud()[1])
    b = jnp.asarray(nu.weights if hasattr(nu, "weights") else nu.as_point_cloud()[1])

    return QuadraticProblem(
        geom_xx=geom_xx,
        geom_yy=geom_yy,
        geom_xy=geom_xy,
        a=a,
        b=b,
        fused_penalty=fused_penalty,
        tau_a=tau_a,
        tau_b=tau_b,
        gw_unbalanced_correction=gw_unbalanced_correction,
        ranks=ranks,
        tolerances=tolerances,
    )
