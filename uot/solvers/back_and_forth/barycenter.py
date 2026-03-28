from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from uot.utils.central_gradient_nd import _central_gradient_nd

from .method import CTransformFn, PushforwardFn, backnforth_sqeuclidean_nd
from .forward_pushforward import cic_pushforward_nd
from .pushforward import adaptive_pushforward_nd
from .c_transform import c_transform_quadratic_fast
from .monge_map import (
    monge_map_from_psi_nd,
    monge_map_cic_from_psi_nd,
    monge_map_adaptive_from_psi_nd,
)
from .solver import BackNForthSqEuclideanSolver


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


def _resolve_monge_map_fn(pushforward_fn: Callable) -> Callable:
    if pushforward_fn is adaptive_pushforward_nd:
        return monge_map_adaptive_from_psi_nd
    if pushforward_fn is cic_pushforward_nd:
        return monge_map_cic_from_psi_nd
    return monge_map_from_psi_nd


@partial(
    jax.jit,
    static_argnames=(
        "outer_maxiter",
        "transport_maxiter",
        "pushforward_fn",
        "c_transform_fn",
        "transport_error_metric",
        "return_monge_maps",
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
    pushforward_fn: PushforwardFn = cic_pushforward_nd,
    c_transform_fn: CTransformFn = c_transform_quadratic_fast,
    return_monge_maps: bool = False,
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
    return_monge_maps : bool, default=False
        If True, compute and return the per-measure Monge maps for the final
        barycenter in the diagnostics dictionary (key ``"monge_maps"``).

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
        - ``monge_maps`` (optional): array of shape (J, *gridshape, d)

    Notes
    -----
    - ``backnforth_sqeuclidean_nd`` must be JAX-traceable and is assumed to
      return a tuple where ``out[1]`` is ``phi``, ``out[2]`` is ``psi``, and
      ``out[4]`` is ``rho_mu``. If the solver output changes, update indices.
    - ``outer_maxiter``, ``transport_maxiter``, ``pushforward_fn``, and
      ``transport_error_metric`` are static arguments in the JIT signature.
    """

    if not callable(pushforward_fn):
        raise TypeError("pushforward_fn must be callable in backnforth_barycenter_sqeuclidean_nd_jax")
    if not callable(c_transform_fn):
        raise TypeError("c_transform_fn must be callable in backnforth_barycenter_sqeuclidean_nd_jax")

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
            c_transform_fn=c_transform_fn,
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

    if return_monge_maps:
        # Re-solve transports from the final barycenter to get final psi's.
        _, psis_fin, _, _, _ = vmapped_pair_solve(mu_fin, measures)
        monge_map_fn = _resolve_monge_map_fn(pushforward_fn)
        monge_maps = jax.vmap(lambda psi: monge_map_fn(psi=-psi), in_axes=0)(psis_fin)
        monge_maps = jnp.moveaxis(monge_maps, 1, -1)  # (J, *gridshape, d)
        diagnostics["monge_maps"] = monge_maps
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
    pushforward_fn: PushforwardFn | str = cic_pushforward_nd,
    c_transform_fn: CTransformFn | str = c_transform_quadratic_fast,
    return_monge_maps: bool = False,
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
    return_monge_maps : bool, default=False
        If True, include per-measure Monge maps in the diagnostics output.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        The final barycenter density and diagnostics dictionary, matching
        ``backnforth_barycenter_sqeuclidean_nd_jax``.
    """
    measures = _stack_measures(measures_weights)
    weights = jnp.asarray(weights, dtype=measures.dtype)
    resolved_pushforward_fn, _ = BackNForthSqEuclideanSolver._resolve_pushforward_fn(pushforward_fn)
    resolved_c_transform_fn, _ = BackNForthSqEuclideanSolver._resolve_c_transform_fn(c_transform_fn)

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
        pushforward_fn=resolved_pushforward_fn,
        c_transform_fn=resolved_c_transform_fn,
        return_monge_maps=return_monge_maps,
    )
    return mu, diag


def _density_mismatch(a: jnp.ndarray, b: jnp.ndarray, norm: str = "l2") -> jnp.ndarray:
    diff = a - b
    if norm == "l1":
        return jnp.sum(jnp.abs(diff))
    if norm == "l2":
        return jnp.linalg.norm(diff.reshape(-1))
    raise ValueError(f"Unsupported mismatch norm '{norm}'. Use 'l1' or 'l2'.")


def build_gibbs_nu(
    phi_list: Sequence[jnp.ndarray],
    lambda_list: Sequence[float] | jnp.ndarray,
    gamma: float,
    *,
    cell_volume: Optional[float] = None,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Build the Gibbs barycenter iterate from weighted phi-potentials."""
    if len(phi_list) == 0:
        raise ValueError("phi_list must contain at least one potential array.")
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive.")

    phi_stack = jnp.stack([jnp.asarray(phi) for phi in phi_list], axis=0)
    weights = jnp.asarray(lambda_list, dtype=phi_stack.dtype)
    if weights.ndim != 1 or weights.shape[0] != phi_stack.shape[0]:
        raise ValueError(
            f"lambda_list must have shape ({phi_stack.shape[0]},), got {weights.shape}."
        )
    weights = weights / jnp.maximum(weights.sum(), jnp.asarray(eps, dtype=weights.dtype))

    weighted_phi = jnp.tensordot(weights, phi_stack, axes=(0, 0))
    V = weighted_phi / jnp.asarray(gamma, dtype=phi_stack.dtype)
    logits = -V
    logits = logits - jnp.max(logits)
    nu_unnorm = jnp.exp(logits)

    if cell_volume is not None:
        nu_unnorm = nu_unnorm * jnp.asarray(cell_volume, dtype=nu_unnorm.dtype)

    normalizer = jnp.maximum(nu_unnorm.sum(), jnp.asarray(eps, dtype=nu_unnorm.dtype))
    return nu_unnorm / normalizer


def solve_barycenter_back_and_forth(
    mu_list: Sequence[jnp.ndarray],
    lambda_list: Sequence[float] | jnp.ndarray,
    gamma: float,
    params: Optional[dict] = None,
):
    """
    Entropy-regularized barycenter outer loop using existing two-marginal back-and-forth pieces.

    Returns
    -------
    tuple
        (nu, phi_list, psi_list, diagnostics)
    """
    params = {} if params is None else dict(params)
    if len(mu_list) == 0:
        raise ValueError("mu_list must contain at least one marginal.")

    mus = [jnp.asarray(mu) for mu in mu_list]
    shape0 = mus[0].shape
    for idx, mu in enumerate(mus):
        if mu.shape != shape0:
            raise ValueError(
                f"All marginals must share one grid shape. mu_list[0]={shape0}, "
                f"mu_list[{idx}]={mu.shape}."
            )

    coordinates = params.get("coordinates")
    if coordinates is None:
        raise ValueError("params['coordinates'] is required.")

    formulation = str(params.get("formulation", "phi")).lower()
    if formulation not in {"phi", "psi"}:
        raise ValueError("params['formulation'] must be 'phi' or 'psi'.")

    num_outer_iters = int(params.get("num_outer_iters", 10))
    eta = jnp.asarray(params.get("eta", 1.0), dtype=mus[0].dtype)
    outer_tol = params.get("outer_tol")
    mismatch_norm = str(params.get("mismatch_norm", "l2")).lower()
    log_every = max(int(params.get("log_every", 1)), 1)
    verbose = bool(params.get("verbose", False))
    cell_volume = params.get("cell_volume")

    resolved_c_transform_fn, _ = BackNForthSqEuclideanSolver._resolve_c_transform_fn(
        params.get("c_transform_fn", c_transform_quadratic_fast)
    )
    two_marginal_solver = params.get("two_marginal_solver", backnforth_sqeuclidean_nd)
    pushforward_fn, _ = BackNForthSqEuclideanSolver._resolve_pushforward_fn(
        params.get("pushforward_fn", adaptive_pushforward_nd)
    )

    apply_h1_inverse = params.get("apply_h1_inverse")
    if apply_h1_inverse is None:
        def _identity(rhs):
            return rhs
        apply_h1_inverse = _identity

    solver_kwargs = dict(params.get("two_marginal_params", {}))
    stepsize = solver_kwargs.pop("stepsize", params.get("stepsize", 1.0))
    maxiterations = int(solver_kwargs.pop("maxiterations", params.get("maxiterations", 500)))
    tolerance = solver_kwargs.pop("tolerance", params.get("tolerance", 1e-3))
    stepsize_lower_bound = solver_kwargs.pop(
        "stepsize_lower_bound",
        params.get("stepsize_lower_bound", 0.01),
    )
    error_metric = solver_kwargs.pop(
        "error_metric",
        params.get("error_metric", "h1_psi_relative"),
    )
    progressbar = bool(solver_kwargs.pop("progressbar", False))

    init_phi_list = params.get("init_phi_list")
    if init_phi_list is None:
        phi_list = [jnp.zeros_like(mu) for mu in mus]
    else:
        if len(init_phi_list) != len(mus):
            raise ValueError("init_phi_list must match len(mu_list).")
        phi_list = [jnp.asarray(phi) for phi in init_phi_list]

    init_psi_list = params.get("init_psi_list")
    if init_psi_list is None:
        psi_list = [jnp.zeros_like(mu) for mu in mus]
    else:
        if len(init_psi_list) != len(mus):
            raise ValueError("init_psi_list must match len(mu_list).")
        psi_list = [jnp.asarray(psi) for psi in init_psi_list]

    weights = jnp.asarray(lambda_list, dtype=mus[0].dtype)
    if weights.ndim != 1 or weights.shape[0] != len(mus):
        raise ValueError(f"lambda_list must have shape ({len(mus)},), got {weights.shape}.")
    weights = weights / jnp.maximum(weights.sum(), jnp.finfo(weights.dtype).eps)

    mass_hist = []
    mismatch_hist = []
    max_mismatch_hist = []
    inner_iter_hist = []
    nu = build_gibbs_nu(phi_list, weights, gamma, cell_volume=cell_volume)
    nu_push_list = [jnp.zeros_like(mus[0]) for _ in mus]

    for outer_idx in range(num_outer_iters):
        if formulation == "phi":
            psi_list = [resolved_c_transform_fn(phi, coordinates) for phi in phi_list]
        else:
            # Keep the driver interface formulation-aware; psi-variant is left for later.
            phi_list = [resolved_c_transform_fn(psi, coordinates) for psi in psi_list]
            raise NotImplementedError(
                "psi-formulation is not implemented yet; use params['formulation']='phi'."
            )

        nu = build_gibbs_nu(phi_list, weights, gamma, cell_volume=cell_volume)

        mismatches = []
        inner_iters = []
        for i, mu in enumerate(mus):
            # Reuse existing two-marginal back-and-forth solver unchanged.
            pair_result = two_marginal_solver(
                mu=mu,
                nu=nu,
                coordinates=coordinates,
                stepsize=stepsize / jnp.maximum(mu.max(), nu.max()),
                maxiterations=maxiterations,
                tolerance=tolerance,
                progressbar=progressbar,
                pushforward_fn=pushforward_fn,
                c_transform_fn=resolved_c_transform_fn,
                stepsize_lower_bound=stepsize_lower_bound,
                error_metric=error_metric,
                **solver_kwargs,
            )

            inner_iters.append(int(pair_result[0]))
            phi_i = pair_result[1]
            psi_i = pair_result[2]
            phi_list[i] = phi_i
            psi_list[i] = psi_i

            # Reuse the existing pushforward implementation directly.
            nu_i, _ = pushforward_fn(mu, -psi_i)
            nu_push_list[i] = nu_i
            rhs_i = weights[i] * (nu_i - nu)
            phi_update = jnp.asarray(apply_h1_inverse(rhs_i), dtype=phi_i.dtype)
            phi_list[i] = phi_i + eta * phi_update

            mismatches.append(float(_density_mismatch(nu_i, nu, norm=mismatch_norm)))

        mass_nu = float(jnp.sum(nu))
        max_mismatch = max(mismatches) if mismatches else 0.0
        mass_hist.append(mass_nu)
        mismatch_hist.append(mismatches)
        max_mismatch_hist.append(max_mismatch)
        inner_iter_hist.append(inner_iters)

        if verbose and (outer_idx % log_every == 0):
            mismatch_str = ", ".join(f"{m:.3e}" for m in mismatches)
            print(
                f"[outer {outer_idx + 1:03d}] mass(nu)={mass_nu:.8f} "
                f"||nu_i-nu||=({mismatch_str}) max={max_mismatch:.3e}"
            )

        if outer_tol is not None and max_mismatch <= float(outer_tol):
            break

    diagnostics = {
        "iterations": len(mass_hist),
        "mass_nu": jnp.asarray(mass_hist, dtype=mus[0].dtype),
        "nu_mismatch": jnp.asarray(mismatch_hist, dtype=mus[0].dtype),
        "max_nu_mismatch": jnp.asarray(max_mismatch_hist, dtype=mus[0].dtype),
        "inner_iterations": jnp.asarray(inner_iter_hist, dtype=jnp.int32),
        "nu_pushforwards": jnp.stack(nu_push_list, axis=0),
        "formulation": formulation,
    }
    return nu, phi_list, psi_list, diagnostics
