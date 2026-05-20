from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import jax

from gpu_tracker.tracker import Tracker

from uot.data.measure import BaseMeasure
from uot.problems.base_problem import Problem
from uot.solvers.base_solver import BaseSolver, SolverOutput
from uot.utils.types import ArrayLike


def _wait_jax_finish(result: dict[str, Any]) -> dict[str, Any]:
    """Block until all JAX arrays in *result* are ready."""
    return jax.tree.map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        result,
    )


def _require(result: dict[str, Any], required: set[str]) -> None:
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Solver returned no {missing!r} fields")


def measure_time(
    prob: Problem,
    instance: BaseSolver,
    marginals: Sequence[BaseMeasure],
    costs: Sequence[ArrayLike],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *instance* and return ``{"time": ms}``."""
    start_time = time.perf_counter()
    solution = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    return {"time": (time.perf_counter() - start_time) * 1000}


def measure_time_and_output(
    prob: Problem,
    instance: BaseSolver,
    marginals: Sequence[BaseMeasure],
    costs: Sequence[ArrayLike],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *instance* and return timing + all solver output fields."""
    start_time = time.perf_counter()
    solution: SolverOutput = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(solution)
    metrics: dict[str, Any] = {"time": (time.perf_counter() - start_time) * 1000}
    metrics.update(solution)
    return metrics


def measure_solution_precision(
    prob: Problem,
    instance: BaseSolver,
    marginals: Sequence[BaseMeasure],
    costs: Sequence[ArrayLike],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *instance* and return relative cost error vs. exact solution."""
    result: SolverOutput = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(result)
    _require(result, {"cost"})
    exact = prob.get_exact_cost()  # type: ignore[attr-defined]
    return {"cost_rerr": abs(exact - result["cost"]) / exact}


def measure_with_gpu_tracker(
    prob: Problem,
    instance: BaseSolver,
    marginals: Sequence[BaseMeasure],
    costs: Sequence[ArrayLike],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *instance* inside a GPU/CPU resource tracker and return detailed metrics."""
    import jax.numpy as jnp

    with Tracker(
        sleep_time=0.1,
        gpu_ram_unit="megabytes",
        time_unit="seconds",
    ) as gt:
        start_time = time.perf_counter()
        metrics: dict[str, Any] = instance.solve(marginals=marginals, costs=costs, **kwargs)
        _wait_jax_finish(metrics)
        finish_time = time.perf_counter()

        if instance.__class__.__name__ == "BackNForthSqEuclideanSolver":
            axes_mu = marginals[0].as_grid(backend="jax", dtype=jnp.float64)[0]
            grids = jnp.meshgrid(*axes_mu, indexing="ij")
            X = jnp.stack(grids, axis=-1)
            mu_nd = marginals[0].as_grid(backend="jax", dtype=jnp.float64)[1]
            nu_nd = marginals[1].as_grid(backend="jax", dtype=jnp.float64)[1]
            extra = instance._extra_metrics(
                mu_nd=mu_nd,
                nu_nd=nu_nd,
                axes_mu=axes_mu,
                X=X,
                psi=-metrics["v_final"],
                T=metrics["monge_map"],
            )
            metrics.update(extra)

        metrics.pop("transport_plan", None)
        metrics.pop("u_final", None)
        metrics.pop("v_final", None)

    usage = gt.resource_usage
    peak_gpu_ram = usage.max_gpu_ram
    gpu_utilization = usage.gpu_utilization
    peak_ram = usage.max_ram
    cpu_utilization = usage.cpu_utilization
    time_counter = finish_time - start_time

    metrics.pop("monge_map", None)

    _require(metrics, {"cost"})
    metrics.update(
        {
            "cost": metrics.get("cost", None),
            "time_unit": usage.compute_time.unit,
            "time": usage.compute_time.time,
            "time_counter": time_counter,
            "gpu_mem_unit": peak_gpu_ram.unit,
            "peak_gpu_mem": peak_gpu_ram.main,
            "combined_peak_gpu_ram": peak_gpu_ram.combined,
            "peak_gpu_util_pct": gpu_utilization.gpu_percentages.max_hardware_percent,
            "mean_gpu_util_pct": gpu_utilization.gpu_percentages.mean_hardware_percent,
            "peak_ram_MiB": peak_ram.main.private_rss,
            "combined_peak_ram_MiB": peak_ram.combined.private_rss,
            "max_cpu_util_pct": cpu_utilization.main.max_hardware_percent,
            "mean_cpu_util_pct": cpu_utilization.main.mean_hardware_percent,
        }
    )
    return metrics
