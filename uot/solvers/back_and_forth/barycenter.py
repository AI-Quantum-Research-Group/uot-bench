from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
from functools import reduce

import numpy as np
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.utils.central_gradient_nd import _central_gradient_nd

from .method import backnforth_sqeuclidean_nd

from functools import partial
from typing import Callable, Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax

# --- your project imports (kept as-is) ---
from uot.utils.central_gradient_nd import _central_gradient_nd
from .method import backnforth_sqeuclidean_nd
# from .pushforward import adaptive_pushforward_nd
from .forward_pushforward import cic_pushforward_nd


def _stack_measures(measures_weights):
    """Stack a sequence of measures into a single array.

    Parameters
    ----------
    measures_weights : jnp.ndarray | Sequence[jnp.ndarray]
        Either a pre-stacked array of shape (J, *gridshape) or a sequence of
        arrays each with shape (*gridshape).

    Returns
    -------
    jnp.ndarray
        A stacked array of shape (J, *gridshape).
    """
    if isinstance(measures_weights, jnp.ndarray):
        return measures_weights
    return jnp.stack(list(measures_weights), axis=0)


@partial(
    jax.jit,
    static_argnames=(
        "outer_maxiter",
        "transport_maxiter",
        "pushforward_fn",
        "transport_error_metric",
    ),
)
def backnforth_barycenter_sqeuclidean_nd_jax(
    weights: jnp.ndarray,                 # (J,)
    measures: jnp.ndarray,                # (J, *gridshape)
    coordinates: Any,                     # pytree (e.g. tuple/list of coord arrays)
    barycenter_init: Optional[jnp.ndarray] = None,  # (*gridshape,)
    outer_maxiter: int = 15,
    stopping_tol: float = 5e-4,
    relaxation: float = 1.0,
    transport_stepsize: float = 1.0,
    transport_maxiter: int = 500,
    transport_tol: float = 1e-3,
    transport_error_metric: str = "h1_psi_relative",
    pushforward_fn: Optional[Callable] = cic_pushforward_nd,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute a Wasserstein barycenter with a JAX-jitted back-and-forth solver.

    This function runs an outer fixed-point loop to update the barycenter
    ``mu`` and, at each iteration, solves J transport problems in parallel
    (one per input measure) via ``jax.vmap``. The outer loop is implemented
    with ``lax.while_loop`` to be JIT-friendly and uses the PHI potential to
    define the stopping residual while the PSI potential is used for the
    pushforward update (as in the original implementation).

    Parameters
    ----------
    weights : jnp.ndarray
        Barycenter weights of shape (J,). These are normalized internally to
        sum to 1 with numerical safeguards.
    measures : jnp.ndarray
        Stacked input measures of shape (J, *gridshape).
    coordinates : Any
        Coordinate pytree passed to the transport solver
        ``backnforth_sqeuclidean_nd`` (e.g. tuple/list of coordinate arrays).
    barycenter_init : jnp.ndarray | None, optional
        Optional initialization for the barycenter with shape (*gridshape).
        If ``None``, the arithmetic mean of ``measures`` is used.
    outer_maxiter : int, default=15
        Maximum number of outer barycenter iterations.
    stopping_tol : float, default=5e-4
        Threshold on the maximum absolute gradient of the aggregated PHI
        potential used to stop the outer loop.
    relaxation : float, default=1.0
        Relaxation factor in (0, 1] for the barycenter update.
    transport_stepsize : float, default=1.0
        Step size passed to ``backnforth_sqeuclidean_nd``.
    transport_maxiter : int, default=500
        Maximum iterations for each transport solve.
    transport_tol : float, default=1e-3
        Tolerance for each transport solve.
    transport_error_metric : str, default="h1_psi_relative"
        Error metric name forwarded to ``backnforth_sqeuclidean_nd``.
    pushforward_fn : Callable | None, default=cic_pushforward_nd
        Pushforward function used to update the barycenter with the aggregated
        PSI potential. Must accept ``(mu, potential)`` and return a tuple
        ``(pushed_density, aux)``.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        A tuple ``(mu, diagnostics)`` where:
        - ``mu`` is the final barycenter density, shape (*gridshape)
        - ``diagnostics`` contains:
          - ``iterations``: scalar number of outer iterations
          - ``final_residual``: scalar residual at termination
          - ``residual_hist``: array of shape (outer_maxiter,)
          - ``max_transport_error_hist``: array of shape (outer_maxiter,)
          - ``max_marginal_error_hist``: array of shape (outer_maxiter,)

    Notes
    -----
    - ``backnforth_sqeuclidean_nd`` must be JAX-traceable and is assumed to
      return a tuple where ``out[1]`` is ``phi``, ``out[2]`` is ``psi``, and
      ``out[4]`` is ``rho_mu``. If the solver output changes, update indices.
    - ``outer_maxiter``, ``transport_maxiter``, ``pushforward_fn``, and
      ``transport_error_metric`` are static arguments in the JIT signature.
    """

    if pushforward_fn is None:
        raise ValueError("pushforward_fn must be provided (e.g. adaptive_pushforward_nd).")

    # normalize weights
    weights = jnp.asarray(weights, dtype=measures.dtype)
    weights = weights / jnp.maximum(weights.sum(), jnp.finfo(weights.dtype).eps)

    # init barycenter
    if barycenter_init is None:
        barycenter_init = measures.mean(axis=0)  # arithmetic mean across J
    mu0 = jnp.clip(barycenter_init, 0.0)
    mu0 = mu0 / jnp.maximum(mu0.sum(), jnp.finfo(mu0.dtype).eps)

    # relaxation in (0,1]
    relaxation = jnp.asarray(relaxation, dtype=mu0.dtype)
    relaxation = jnp.clip(relaxation, jnp.asarray(1e-12, mu0.dtype), jnp.asarray(1.0, mu0.dtype))

    # --- per-pair solve: (mu, nu) -> (phi, psi, rho_mu, l1_err, l2_err)
    def _pair_solve(mu, nu):
        out = backnforth_sqeuclidean_nd(
            mu=mu,
            nu=nu,
            coordinates=coordinates,
            stepsize=transport_stepsize,
            maxiterations=transport_maxiter,
            tolerance=transport_tol,
            progressbar=False,
            pushforward_fn=pushforward_fn,
            error_metric=transport_error_metric,
        )

        # IMPORTANT: adjust indices if your solver returns in different positions
        phi = out[1]     # was "_" in your original destructuring
        psi = out[2]
        rho_mu = out[4]

        l1_err = jnp.sum(jnp.abs(rho_mu - nu))
        l2_err = jnp.sum(jnp.square(rho_mu - nu))
        return phi, psi, rho_mu, l1_err, l2_err

    # vectorize across measures: mu is shared (None), nu varies along axis 0
    vmapped_pair_solve = jax.vmap(
        _pair_solve,
        in_axes=(None, 0),
        out_axes=(0, 0, 0, 0, 0),
    )

    # fixed-size diagnostic buffers (jit-friendly)
    residual_hist = jnp.zeros((outer_maxiter,), dtype=mu0.dtype)
    max_transport_err_hist = jnp.zeros((outer_maxiter,), dtype=mu0.dtype)
    max_marginal_err_hist = jnp.zeros((outer_maxiter,), dtype=mu0.dtype)

    init_residual = jnp.asarray(jnp.inf, dtype=mu0.dtype)
    carry0 = (0, mu0, init_residual, residual_hist, max_transport_err_hist, max_marginal_err_hist)

    def cond_fn(carry):
        i, _, residual, *_ = carry
        return jnp.logical_and(i < outer_maxiter, residual > stopping_tol)

    def body_fn(carry):
        i, mu, _, residual_hist, max_transport_hist, max_marginal_hist = carry

        # parallel transport solves
        phis, psis, rhos_mu, l1_errs, l2_errs = vmapped_pair_solve(mu, measures)

        # broadcast weights to field shape
        # psis has shape (J, *gridshape), so add singleton dims
        w = weights.reshape((weights.shape[0],) + (1,) * (psis.ndim - 1))

        # accumulate both potentials
        phi_accum = jnp.sum(w * phis, axis=0)
        psi_accum = jnp.sum(w * psis, axis=0)

        # residual computed with PHI
        grad_residual = _central_gradient_nd(phi_accum)
        residual = jnp.max(jnp.abs(grad_residual))

        # errors
        max_transport_error = jnp.max(l1_errs)
        max_marginal_error = jnp.max(l2_errs)

        # pushforward uses PSI (unchanged)
        pushed_density, _ = pushforward_fn(mu, -psi_accum)
        mu_new = (1.0 - relaxation) * mu + relaxation * pushed_density
        mu_new = jnp.clip(mu_new, 0.0)
        mu_new = mu_new / jnp.maximum(mu_new.sum(), jnp.finfo(mu_new.dtype).eps)

        # write histories
        residual_hist = residual_hist.at[i].set(residual)
        max_transport_hist = max_transport_hist.at[i].set(max_transport_error)
        max_marginal_hist = max_marginal_hist.at[i].set(max_marginal_error)

        return (i + 1, mu_new, residual, residual_hist, max_transport_hist, max_marginal_hist)

    i_fin, mu_fin, residual_fin, residual_hist, max_transport_hist, max_marginal_hist = lax.while_loop(
        cond_fn, body_fn, carry0
    )

    diagnostics = {
        "iterations": i_fin,                         # scalar int
        "final_residual": residual_fin,              # scalar
        "residual_hist": residual_hist,              # (outer_maxiter,)
        "max_transport_error_hist": max_transport_hist,  # (outer_maxiter,)
        "max_marginal_error_hist": max_marginal_hist,    # (outer_maxiter,)
    }
    return mu_fin, diagnostics


def backnforth_barycenter_sqeuclidean_nd_optimized(
    weights,
    measures_weights,
    coordinates,
    barycenter_init=None,
    outer_maxiter: int = 15,
    stopping_tol: float = 5e-4,
    relaxation: float = 1.0,
    transport_stepsize: float = 1.0,
    transport_maxiter: int = 500,
    transport_tol: float = 1e-3,
    transport_error_metric: str = "h1_psi_relative",
    pushforward_fn: Optional[Callable] = cic_pushforward_nd,
):
    """Convenience wrapper for the JAX barycenter solver.

    This function mirrors the original API and allows ``measures_weights`` to
    be either a stacked array or a sequence of arrays. It forwards all solver
    parameters to ``backnforth_barycenter_sqeuclidean_nd_jax``.

    Parameters
    ----------
    weights : array-like
        Barycenter weights of shape (J,).
    measures_weights : jnp.ndarray | Sequence[jnp.ndarray]
        Input measures as either a stacked array of shape (J, *gridshape) or
        a sequence of arrays each with shape (*gridshape).
    coordinates : Any
        Coordinate pytree for the transport solver.
    barycenter_init : jnp.ndarray | None, optional
        Optional initialization for the barycenter with shape (*gridshape).
    outer_maxiter : int, default=15
        Maximum number of outer iterations.
    stopping_tol : float, default=5e-4
        Outer loop stopping tolerance.
    relaxation : float, default=1.0
        Relaxation factor in (0, 1] for the barycenter update.
    transport_stepsize : float, default=1.0
        Step size for each transport solve.
    transport_maxiter : int, default=500
        Maximum iterations for each transport solve.
    transport_tol : float, default=1e-3
        Tolerance for each transport solve.
    transport_error_metric : str, default="h1_psi_relative"
        Error metric name forwarded to ``backnforth_sqeuclidean_nd``.
    pushforward_fn : Callable | None, default=cic_pushforward_nd
        Pushforward function used for the barycenter update.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        The final barycenter density and diagnostics dictionary, matching
        ``backnforth_barycenter_sqeuclidean_nd_jax``.
    """
    measures = _stack_measures(measures_weights)
    weights = jnp.asarray(weights, dtype=measures.dtype)

    mu, diag = backnforth_barycenter_sqeuclidean_nd_jax(
        weights=weights,
        measures=measures,
        coordinates=coordinates,
        barycenter_init=barycenter_init,
        outer_maxiter=outer_maxiter,
        stopping_tol=stopping_tol,
        relaxation=relaxation,
        transport_stepsize=transport_stepsize,
        transport_maxiter=transport_maxiter,
        transport_tol=transport_tol,
        transport_error_metric=transport_error_metric,
        pushforward_fn=pushforward_fn,
    )
    return mu, diag
