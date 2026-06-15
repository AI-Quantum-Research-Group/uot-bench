from __future__ import annotations

from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax import lax


def _normalize(v, eps=1e-30):
    return v / jnp.maximum(jnp.sum(v), eps)


def _normalize_measures(measures, eps=1e-30):
    z = jnp.maximum(
        jnp.sum(measures, axis=tuple(range(1, measures.ndim)), keepdims=True), eps
    )
    return measures / z


def _check_uniform_axis_1d(x, tol=1e-6):
    dx = x[1] - x[0]
    ok = jnp.all(jnp.abs((x[1:] - x[:-1]) - dx) <= tol * jnp.maximum(1.0, jnp.abs(dx)))
    return dx, ok


def _gaussian_kernel_offsets_axis(x, reg):
    """
    1D Gaussian kernel on offsets for one axis:
        K(i,j) = exp(-0.5 * (x_i - x_j)^2 / reg)

    Returns kernel vector of length 2n-1.
    """
    n = x.shape[0]
    dx, ok = _check_uniform_axis_1d(x)
    dx = jnp.asarray(dx, dtype=x.dtype)

    offsets = dx * jnp.arange(-(n - 1), n, dtype=x.dtype)
    kernel = jnp.exp(-0.5 * (offsets**2) / reg)
    return kernel, ok


def _conv_same_1d_batched(signal, kernel):
    """
    signal: (B, L)
    kernel: (W,)
    output: (B, L)
    """
    lhs = signal[:, None, :]  # (B, 1, L)
    rhs = kernel[None, None, :]  # (1, 1, W)
    out = lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding="SAME",
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return out[:, 0, :]


def _apply_conv_along_axis(field, kernel, axis):
    """
    Apply 1D convolution along a chosen spatial axis.

    field shape: (..., n_axis, ...)
    returns same shape.
    """
    axis = int(axis)
    field_moved = jnp.moveaxis(field, axis, -1)
    orig_shape = field_moved.shape
    L = orig_shape[-1]
    batch = prod(orig_shape[:-1])

    field_flat = field_moved.reshape((batch, L))
    out_flat = _conv_same_1d_batched(field_flat, kernel)
    out = out_flat.reshape(orig_shape)
    out = jnp.moveaxis(out, -1, axis)
    return out


def _make_apply_K_separable(coordinates, reg):
    """
    Build separable Gaussian operator K on a common tensor-product grid.

    coordinates: tuple/list of 1D axes
    """
    kernels = []
    oks = []
    for x in coordinates:
        k, ok = _gaussian_kernel_offsets_axis(jnp.asarray(x), reg)
        kernels.append(k)
        oks.append(ok)

    def apply_K(v):
        # v shape: (*grid_shape)
        out = v
        for ax, ker in enumerate(kernels):
            out = _apply_conv_along_axis(out, ker, ax)
        return out

    grid_ok = jnp.all(jnp.asarray(oks))
    return apply_K, grid_ok


def _apply_K_to_stack(stack, apply_K):
    """
    stack shape: (J, *grid_shape)
    apply K to each slice.
    """
    return jax.vmap(apply_K, in_axes=0, out_axes=0)(stack)


def _geom_weighted_mean(stack, lambdas, eps=1e-30):
    """
    stack shape: (J, *grid_shape)
    lambdas shape: (J,)
    returns:
        exp(sum_j lambda_j log(stack_j))
    """
    lam = lambdas / jnp.maximum(jnp.sum(lambdas), eps)
    log_stack = jnp.log(jnp.maximum(stack, eps))
    return jnp.exp(jnp.tensordot(lam, log_stack, axes=(0, 0)))


def _barycenter_error_scaling_domain(measures, u, v, b, apply_K):
    """
    Common-grid scaling-domain residuals.

    measures: (J, *grid_shape)
    u, v:     (J, *grid_shape)
    b:        (*grid_shape)
    """
    Kv = _apply_K_to_stack(v, apply_K)
    Ku = _apply_K_to_stack(u, apply_K)

    a_hat = u * Kv
    b_hat = v * Ku

    spatial_axes = tuple(range(1, measures.ndim))
    err_a_per_j = jnp.sum((a_hat - measures) ** 2, axis=spatial_axes)
    err_b_per_j = jnp.sum((b_hat - b[None, ...]) ** 2, axis=spatial_axes)

    err_a = jnp.max(err_a_per_j)
    err_b = jnp.max(err_b_per_j)
    err = jnp.maximum(err_a, err_b)

    return err, {
        "err_a": err_a,
        "err_b": err_b,
    }


@partial(
    jax.jit,
    static_argnames=(
        "reg",
        "maxiter",
        "return_diagnostics",
        "error_check_every",
    ),
)
def barycenter_sinkhorn_conv_nd(
    measures: jnp.ndarray,
    coordinates,
    lambdas: jnp.ndarray,
    reg: float = 1e-3,
    tol: float = 1e-4,
    maxiter: int = 100,
    return_diagnostics: bool = False,
    error_check_every: int = 20,
):
    """
    Convolutional d-dimensional Sinkhorn barycenter on a common regular grid.

    Parameters
    ----------
    measures : array, shape (J, *grid_shape)
        Input marginals on the same tensor-product grid.
    coordinates : tuple/list of 1D arrays
        One axis array per dimension.
    lambdas : array, shape (J,)
        Nonnegative barycenter weights.
    reg : float
        Entropic regularization parameter.
    tol : float
        Stopping tolerance on the scaling-domain residual.
    maxiter : int
        Maximum number of Sinkhorn iterations.
    return_diagnostics : bool
        Whether to return internal arrays.
    error_check_every : int
        Compute expensive residual only every this many iterations.

    Returns
    -------
    b : array, shape (*grid_shape)
        Barycenter.
    diagnostics : dict
    """
    measures = _normalize_measures(jnp.asarray(measures))
    lambdas = jnp.asarray(lambdas, dtype=measures.dtype)
    lambdas = lambdas / jnp.maximum(jnp.sum(lambdas), 1e-30)

    J = measures.shape[0]
    grid_shape = measures.shape[1:]

    if len(coordinates) != len(grid_shape):
        raise ValueError(
            "len(coordinates) must match the number of spatial dimensions."
        )
    for ax, x in enumerate(coordinates):
        if x.shape[0] != grid_shape[ax]:
            raise ValueError(
                f"Axis {ax} has length {x.shape[0]}, expected {grid_shape[ax]}."
            )

    apply_K, grid_ok = _make_apply_K_separable(coordinates, reg)

    # initialize
    b = _normalize(jnp.mean(measures, axis=0))
    u = jnp.ones_like(measures)
    v = jnp.ones_like(measures)

    errors = jnp.full((maxiter,), jnp.nan, dtype=measures.dtype)
    err0 = jnp.asarray(jnp.inf, dtype=measures.dtype)

    def cond_fn(state):
        i, u, v, b, err, errors = state
        return jnp.logical_and(i < maxiter, err > tol)

    def body_fn(state):
        i, u, v, b, err, errors = state

        Kv = _apply_K_to_stack(v, apply_K)
        u = measures / jnp.maximum(Kv, 1e-30)

        Ku = _apply_K_to_stack(u, apply_K)
        b = _geom_weighted_mean(Ku, lambdas)
        b = _normalize(b)

        v = b[None, ...] / jnp.maximum(Ku, 1e-30)

        def compute_err(_):
            new_err, _details = _barycenter_error_scaling_domain(
                measures, u, v, b, apply_K
            )
            return new_err

        err = lax.cond(
            (i % error_check_every) == 0,
            compute_err,
            lambda _: err,
            operand=None,
        )

        errors = errors.at[i].set(err)
        return (i + 1, u, v, b, err, errors)

    init_state = (jnp.asarray(0), u, v, b, err0, errors)
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    iterations, u, v, b, err, errors = final_state

    diagnostics = {
        "iterations": iterations,
        "error": err,
        "grid_is_uniform": grid_ok,
        **(
            {
                "u": u,
                "v": v,
                "b": b,
                "errors": errors,
            }
            if return_diagnostics
            else {}
        ),
    }
    return b, diagnostics
