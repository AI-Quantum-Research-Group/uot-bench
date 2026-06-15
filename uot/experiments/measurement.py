from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Required, TypedDict, cast

import jax

from uot.problems.base_problem import Problem
from uot.problems.protocols import HasExactCost
from uot.solvers.base_solver import BaseSolver, SolverOutput


class MeasurementTime(TypedDict):
    time: float


class MeasurementTimeAndOutput(TypedDict, total=False):
    cost: Required[jax.Array | float]
    time: Required[float]
    transport_plan: jax.Array
    coupling: jax.Array
    iterations: int
    converged: bool
    error: float | jax.Array
    u_final: jax.Array
    v_final: jax.Array
    potentials: tuple[jax.Array, jax.Array]
    monge_map: jax.Array


class MeasurementPrecision(TypedDict):
    cost_rerr: float


class MeasurementGPU(TypedDict, total=False):
    cost: jax.Array | float
    time: float
    time_unit: str
    time_counter: float
    gpu_mem_unit: str
    peak_gpu_mem: float
    combined_peak_gpu_ram: float
    peak_gpu_util_pct: float
    mean_gpu_util_pct: float
    peak_ram_MiB: float
    combined_peak_ram_MiB: float
    max_cpu_util_pct: float
    mean_cpu_util_pct: float
    iterations: int
    error: float | jax.Array


def _wait_jax_finish(result: Mapping[str, Any]) -> None:
    """Block until all JAX arrays in *result* are ready."""
    jax.tree.map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        result,
    )


def _require(result: Mapping[str, Any], required: set[str]) -> None:
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"Solver returned no {missing!r} fields")


def invoke_solver(instance: BaseSolver, view: Any, **kwargs: Any) -> SolverOutput:
    """Call *instance*.solve with the correct convention for its ``input_kind``.

    For the default ``"marginals_costs"`` kind, unpacks ``view.marginals`` and
    ``view.costs`` as keyword arguments — preserving the existing native-solver
    contract byte-for-byte.

    For any other kind, passes *view* as the sole positional argument so that
    solvers with pre-built backend representations can receive them directly.
    """
    kind = getattr(instance, "input_kind", "marginals_costs")
    if kind == "marginals_costs":
        return instance.solve(marginals=view.marginals, costs=view.costs, **kwargs)
    return instance.solve(view, **kwargs)


def measure_time(
    prob: Problem,
    instance: BaseSolver,
    view: Any,
    **kwargs: Any,
) -> MeasurementTime:
    """Run *instance* and return ``{"time": ms}``."""
    start_time = time.perf_counter()
    solution = invoke_solver(instance, view, **kwargs)
    _wait_jax_finish(solution)
    return {"time": (time.perf_counter() - start_time) * 1000}


def measure_time_and_output(
    prob: Problem,
    instance: BaseSolver,
    view: Any,
    **kwargs: Any,
) -> MeasurementTimeAndOutput:
    """Run *instance* and return timing + all solver output fields."""
    start_time = time.perf_counter()
    solution: SolverOutput = invoke_solver(instance, view, **kwargs)
    _wait_jax_finish(solution)
    metrics: dict[str, Any] = {"time": (time.perf_counter() - start_time) * 1000}
    metrics.update(solution)
    return metrics  # type: ignore[return-value]


def measure_solution_precision(
    prob: Problem,
    instance: BaseSolver,
    view: Any,
    **kwargs: Any,
) -> MeasurementPrecision:
    """Run *instance* and return relative cost error vs. exact solution."""
    result: SolverOutput = invoke_solver(instance, view, **kwargs)
    _wait_jax_finish(result)
    _require(result, {"cost"})
    if not isinstance(prob, HasExactCost):
        raise TypeError(f"{type(prob).__name__} does not implement get_exact_cost()")
    exact = prob.get_exact_cost()
    return {"cost_rerr": float(abs(exact - result["cost"]) / exact)}


def measure_with_gpu_tracker(
    prob: Problem,
    instance: BaseSolver,
    view: Any,
    **kwargs: Any,
) -> MeasurementGPU:
    """Run *instance* inside a GPU/CPU resource tracker and return detailed metrics."""
    import jax.numpy as jnp
    from gpu_tracker.tracker import Tracker as _Tracker  # type: ignore[import-untyped]

    with _Tracker(
        sleep_time=0.1,
        gpu_ram_unit="megabytes",
        time_unit="seconds",
    ) as gt:
        start_time = time.perf_counter()
        metrics: dict[str, Any] = cast(dict[str, Any], invoke_solver(instance, view, **kwargs))
        _wait_jax_finish(metrics)
        finish_time = time.perf_counter()

        if instance.__class__.__name__ == "BackNForthSqEuclideanSolver":
            marginals = view.marginals  # view is SolverInputs for marginals_costs kind
            axes_mu = marginals[0].as_grid(backend="jax", dtype=jnp.float64)[0]  # type: ignore[attr-defined]
            grids = jnp.meshgrid(*axes_mu, indexing="ij")
            X = jnp.stack(grids, axis=-1)
            mu_nd = marginals[0].as_grid(backend="jax", dtype=jnp.float64)[1]  # type: ignore[attr-defined]
            nu_nd = marginals[1].as_grid(backend="jax", dtype=jnp.float64)[1]  # type: ignore[attr-defined]
            extra = instance._extra_metrics(  # type: ignore[attr-defined]
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

    assert gt.resource_usage is not None, "Tracker did not record resource usage"
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
    return metrics  # type: ignore[return-value]
