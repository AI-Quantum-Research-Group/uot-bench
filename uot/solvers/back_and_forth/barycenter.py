from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
from functools import reduce

import numpy as np
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.central_gradient_nd import _central_gradient_nd

from .method import backnforth_sqeuclidean_nd
from .pushforward import adaptive_pushforward_nd


def backnforth_barycenter_sqeuclidean_nd(
    weights: Sequence[float],
    measures_weights: Sequence[jnp.ndarray],
    coordinates: Sequence[jnp.ndarray],
    barycenter_init: jnp.ndarray | None = None,
    outer_maxiter: int = 15,
    stopping_tol: float = 5e-4,
    relaxation: float = 1.0,
    transport_stepsize: float = 1.0,
    transport_maxiter: int = 500,
    transport_tol: float = 1e-3,
    transport_error_metric: str = 'h1_psi_relative',
    pushforward_fn=adaptive_pushforward_nd,
):
    if barycenter_init is None:
        barycenter_init = reduce(lambda a, b: a + b, measures_weights)  # arithmetic mean
    barycenter_density = jnp.clip(barycenter_init, 0.0)
    barycenter_density = barycenter_density / jnp.maximum(
        barycenter_density.sum(), jnp.finfo(barycenter_density.dtype).eps
    )

    history = []
    pushforward_path = []
    debug_snapshots = []
    pair_debug_info = []
    relaxation = float(relaxation)
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1].")

    for outer_iter in range(outer_maxiter):
        iteration_snapshot = {"mu": np.asarray(barycenter_density), "nus": [np.asarray(w) for w in measures_weights]}
        iteration_pairs = []
        psi_accum = jnp.zeros_like(barycenter_density)
        max_transport_error = 0.0
        max_marginal_error = 0.0

        for lambd, weights in zip(weights, measures_weights):
            call_debug = {"mu": np.asarray(barycenter_density), "nu": np.asarray(weights)}
            (
                iterations,
                _,
                psi,
                _,
                rho_mu,
                _,
                _,
                _,
            ) = backnforth_sqeuclidean_nd(
                mu=barycenter_density,
                nu=weights,
                coordinates=coordinates,
                stepsize=transport_stepsize,
                maxiterations=transport_maxiter,
                tolerance=transport_tol,
                progressbar=False,
                pushforward_fn=pushforward_fn,
                error_metric=transport_error_metric,
            )
            print(f"Internal: bfm converged after {iterations} iterations")
            psi_accum = psi_accum + lambd * psi
            call_debug["pushforward"] = np.asarray(rho_mu)
            iteration_pairs.append(call_debug)
            pair_error = jnp.sum(jnp.abs(rho_mu - weights))
            marginal_error = jnp.sum(jnp.power(rho_mu - weights, 2))
            max_transport_error = max(float(pair_error), max_transport_error)
            max_marginal_error = max(float(marginal_error), max_marginal_error)

        grad_residual = _central_gradient_nd(psi_accum)
        residual = float(jnp.max(jnp.abs(grad_residual)))

        pushed_density, _ = pushforward_fn(barycenter_density, -psi_accum)
        pushforward_path.append(np.asarray(pushed_density))
        barycenter_density = (
            (1.0 - relaxation) * barycenter_density + relaxation * pushed_density
        )
        barycenter_density = jnp.clip(barycenter_density, 0.0)
        barycenter_density = barycenter_density / jnp.maximum(
            barycenter_density.sum(), jnp.finfo(barycenter_density.dtype).eps
        )

        iteration_snapshot["output"] = np.asarray(barycenter_density)
        debug_snapshots.append(iteration_snapshot)

        pair_debug_info.append(iteration_pairs)

        history.append(
            {
                "outer_iter": outer_iter,
                "residual": residual,
                "max_transport_error": max_transport_error,
                "max_marginal_error": max_marginal_error,
            }
        )
        if residual < stopping_tol:
            break
    diagnostics = {
        "history": history,
        "iterations": len(history),
        "final_residual": history[-1]["residual"] if history else 0.0,
        "max_transport_error": history[-1]["max_transport_error"] if history else 0.0,
        "max_marginal_error": history[-1]["max_marginal_error"] if history else 0.0,
        "pushforward_path": pushforward_path,
        "debug_snapshots": debug_snapshots,
        "pair_debug_info": pair_debug_info,
    }
    return barycenter_density, diagnostics
