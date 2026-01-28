import jax.numpy as jnp
from jax import lax
from jax import jit
from jax.scipy.special import logsumexp


@jit(static_argnames=[
    'reg',
    'maxiter',
    'return_diagnostics',
])
def barycenter_sinkhorn(
        measures: jnp.ndarray,
        cost: jnp.ndarray,
        lambdas: jnp.ndarray,
        reg: float = 1e-3,
        tol: float = 1e-4,
        maxiter: int = 100,
        return_diagnostics: bool = False,
        error_check_every: int = 20,
):
    lam = (lambdas / lambdas.sum())[:, None]   # (J,1)
    # clip measures for log computations to be a little bit safer in practice
    log_measures = jnp.log(jnp.maximum(measures, 1e-30))
    lnb = jnp.zeros_like(measures[0])
    lnK = -cost / reg

    def cond_fn(state):
        i, lnus, lnvs, lnb, err, errors = state
        return jnp.logical_and(i < maxiter, err > tol)

    def body_fn(state):
        i, lnus, lnvs, lnb, err, errors = state

        # update each f^j
        ln_Kv = logsumexp(
            lnvs[:, None, :] + lnK[None, :, :],
            axis=2,
        )
        lnus = log_measures - ln_Kv
        # compute log(K^T u^j) in a stable way
        ln_Ktu = logsumexp(
            lnus[:, :, None] + lnK[None, :, :],
            axis=1,
        )
        # update barycenter log(b)
        lnb = jnp.sum(lam * ln_Ktu, axis=0)
        lnb = lnb - logsumexp(lnb)   # now renormalize
        # update each g^j
        lnvs = lnb[None, :] - ln_Ktu

        def compute_err(_):
            new_err, _details = marginal_error_from_log(log_measures,
                                                        lnK, lnus, lnvs, lnb)
            return new_err
        err = lax.cond(
            (i % error_check_every) == 0,
            compute_err,
            lambda _: err,
            operand=None,
        )
        errors = errors.at[i].set(err)
        return (i+1, lnus, lnvs, lnb, err, errors)

    lnus = jnp.zeros_like(measures)
    lnvs = jnp.zeros_like(measures)
    errors = jnp.full((maxiter,), jnp.nan)
    err0 = jnp.asarray(jnp.inf)
    init_state = (jnp.asarray(0), lnus, lnvs, lnb, err0, errors)
    final_state = lax.while_loop(cond_fn, body_fn, init_state)
    iterations, lnus, lnvs, lnb, err, errors = final_state
    b = jnp.exp(lnb)
    b /= b.sum()
    diagnostics = {
            'iterations': iterations,
            'error': err,
            **({
                'ln_u': lnus,
                'ln_v': lnvs,
                'ln_b': lnb,
                'errors': errors,
                } if return_diagnostics else {}),
            }
    return b, diagnostics


def marginal_error_from_log(
    log_measures: jnp.ndarray,  # (J, N) = log(a_j)
    lnK: jnp.ndarray,           # (N, N) = -C/reg
    lnus: jnp.ndarray,          # (J, N) = log(u_j)
    lnvs: jnp.ndarray,          # (J, N) = log(v_j)
    lnb: jnp.ndarray,           # (N,)   = log(b)
    eps: float = 1e-30,
):
    # ln(K v_j) and ln(K^T u_j)
    lnKv = logsumexp(lnvs[:, None, :] + lnK[None, :, :], axis=2)    # (J, N)
    lnKtu = logsumexp(lnus[:, :, None] + lnK[None, :, :], axis=1)   # (J, N)

    # implied marginals in normal domain
    a_hat = jnp.exp(lnus + lnKv)                 # (J, N) should match exp(log_measures)
    b_hat = jnp.exp(lnvs + lnKtu)                # (J, N) should match exp(lnb)[None,:]

    a = jnp.exp(log_measures)                    # (J, N)
    b = jnp.exp(lnb)                             # (N,)

    # # L1 errors per measure
    # err_a_per_j = jnp.sum(jnp.abs(a_hat - a), axis=1)           # (J,)
    # err_b_per_j = jnp.sum(jnp.abs(b_hat - b[None, :]), axis=1)  # (J,)
    # L2 errors per measure
    err_a_per_j = jnp.sum(jnp.power(a_hat - a, 2), axis=1)           # (J,)
    err_b_per_j = jnp.sum(jnp.power(b_hat - b[None, :], 2), axis=1)  # (J,)

    err_a = jnp.max(err_a_per_j)
    err_b = jnp.max(err_b_per_j)
    err = jnp.maximum(err_a, err_b)

    return err, {"err_a": err_a, "err_b": err_b}
