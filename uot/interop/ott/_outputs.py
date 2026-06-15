"""Translate OTT-JAX output objects into uot SolverOutput TypedDicts."""

from __future__ import annotations

import jax.numpy as jnp

from uot.solvers.base_solver import SolverOutput


def _last_finite(arr) -> float:
    if arr is None:
        return float("nan")
    arr = jnp.asarray(arr).ravel()
    finite = arr[jnp.isfinite(arr)]
    return float(finite[-1]) if finite.size > 0 else float("nan")


def from_sinkhorn_output(out) -> SolverOutput:
    """Translate a ``SinkhornOutput`` to a uot ``SolverOutput``.

    Uses ``out.primal_cost`` (= <C, P>) as the transport cost, which matches
    what uot-bench's native solvers store in the ``cost`` field.
    """
    f, g = out.f, out.g
    result: SolverOutput = {
        "cost": jnp.asarray(float(out.primal_cost)),
        "transport_plan": out.matrix,
        "u_final": jnp.asarray(f),
        "v_final": jnp.asarray(g),
        "potentials": (jnp.asarray(f), jnp.asarray(g)),
        "iterations": int(out.n_iters),
        "converged": bool(out.converged),
        "error": _last_finite(out.errors),
    }
    return result


def from_lr_sinkhorn_output(out) -> SolverOutput:
    """Translate a ``LRSinkhornOutput`` to a uot ``SolverOutput``.

    The coupling Q diag(1/g) R^T is NOT materialised; factors are stored in
    ``low_rank_plan``.  ``primal_cost`` (= <C, Q diag(1/g) R^T>) is used as
    the transport cost.
    """
    return {
        "cost": jnp.asarray(float(out.primal_cost)),
        "low_rank_plan": (
            jnp.asarray(out.q),
            jnp.asarray(out.r),
            jnp.asarray(out.g),
        ),
        "iterations": int(out.n_iters),
        "converged": bool(out.converged),
        "error": _last_finite(out.errors),
    }


def _as_iter(v) -> list:
    """Normalise a possibly-scalar OTT field into a list (None -> [])."""
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return [v]


def from_gw_output(out) -> SolverOutput:
    """Translate a ``GWOutput`` to a uot ``SolverOutput``."""
    # OTT pads ``out.costs`` with -1.0 for outer iterations that never ran, so
    # the last entry is usually the -1 sentinel rather than the final GW cost.
    costs = getattr(out, "costs", None)
    valid_costs = jnp.asarray([]) if costs is None else jnp.asarray(costs).ravel()
    valid_costs = valid_costs[valid_costs != -1]
    last_cost = float(valid_costs[-1]) if valid_costs.size > 0 else float("nan")

    # GW reports n_iters == -1; fall back to the number of executed outer steps.
    n_iters = getattr(out, "n_iters", -1)
    if n_iters is None or n_iters < 0:
        iterations = int(valid_costs.size)
    else:
        iterations = int(n_iters)

    result: SolverOutput = {
        "cost": jnp.asarray(last_cost),
        "iterations": iterations,
        "converged": bool(getattr(out, "converged", False)),
        "error": _last_finite(getattr(out, "errors", None)),
    }
    matrix = getattr(out, "matrix", None)
    if matrix is not None:
        result["transport_plan"] = matrix
    return result


def from_sinkhorn_divergence_output(divergence_val, out) -> SolverOutput:
    """Translate sinkhorn_divergence output to a uot ``SolverOutput``."""
    converged = _as_iter(getattr(out, "converged", None))
    n_iters = _as_iter(getattr(out, "n_iters", None))
    errors = getattr(out, "errors", None)
    if errors is not None:
        try:
            last_errors = errors[-1]
        except (TypeError, IndexError, KeyError):
            last_errors = errors
    else:
        last_errors = None
    return {
        "cost": jnp.asarray(float(divergence_val)),
        "converged": bool(all(converged)) if converged else False,
        "iterations": int(sum(n_iters)) if n_iters else 0,
        "error": _last_finite(last_errors),
    }


def from_discrete_barycenter_output(out) -> SolverOutput:
    """Translate a barycenter output to a uot ``SolverOutput``."""
    bar = getattr(out, "x", None)
    cost = float(jnp.sum(jnp.asarray(out.costs))) if hasattr(out, "costs") else float("nan")
    result: SolverOutput = {
        "cost": jnp.asarray(cost),
        "converged": bool(getattr(out, "converged", True)),
    }
    if bar is not None:
        result["barycenter"] = jnp.asarray(bar)  # type: ignore[typeddict-unknown-key]
    return result
