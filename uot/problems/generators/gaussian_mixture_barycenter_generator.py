import numpy as np
import jax
import jax.numpy as jnp

from collections.abc import Callable, Iterator
from typing import Literal

from uot.problems.barycenter_problem import BarycenterProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.generate_nd_grid import generate_nd_grid, compute_cell_volume
from uot.utils.generator_helpers import (
    get_gmm_pdf as get_gmm_pdf_jax,
    generate_gmm_coefficients,
    build_gmm_pdf,
    build_gmm_pdf_scipy,
    sample_gmm_params_wishart,
    get_axes,
)
from uot.utils.build_measure import _build_measure
from uot.utils.types import ArrayLike


# Keep defaults consistent with GaussianMixtureGenerator.
MEAN_FROM_BORDERS_COEF = 0.2
VARIANCE_LOWER_BOUND_COEF = 0.001
VARIANCE_UPPER_BOUND_COEF = 0.01


class GaussianMixtureBarycenterGenerator(ProblemGenerator):
    """
    Generate barycenter problems with N Gaussian (or Gaussian-mixture) marginals
    discretized on a shared Cartesian grid.

    The sampling and discretization logic mirrors GaussianMixtureGenerator:
    mixture parameters are sampled per marginal, the pdf is evaluated on a grid,
    and the resulting weights are normalized (with optional cell-volume scaling).
    """

    def __init__(
        self,
        name: str,
        dim: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
        num_components: int = 1,
        num_marginals: int | None = None,
        use_jax: bool = True,
        seed: int = 42,
        wishart_df: int | None = None,
        wishart_scale: np.ndarray | None = None,
        mean_from_borders_coef: float = MEAN_FROM_BORDERS_COEF,
        variance_lower_bound_coef: float = VARIANCE_LOWER_BOUND_COEF,
        variance_upper_bound_coef: float = VARIANCE_UPPER_BOUND_COEF,
        measure_mode: str = "grid",
        cell_discretization: str = "cell-centered",
        analytic_when_possible: bool = True,
        analytic_mode: Literal["auto", "1d", "commuting", "two_marginal"] = "auto",
    ) -> None:
        super().__init__()
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2 or 3")
        if num_components < 1:
            raise ValueError("num_components must be >= 1")
        self._name = name
        self._dim = dim
        self._num_components = num_components
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._num_marginals = num_marginals
        self._use_jax = use_jax
        self._measure_mode = measure_mode
        self._mean_from_borders_coef = mean_from_borders_coef
        self._variance_lower_bound_coef = variance_lower_bound_coef
        self._variance_upper_bound_coef = variance_upper_bound_coef
        self._wishart_df = wishart_df if wishart_df is not None else dim + 1
        self._wishart_scale = wishart_scale if wishart_scale is not None else np.eye(
            dim)
        self.cell_discretization = cell_discretization
        self._analytic_when_possible = analytic_when_possible
        self._analytic_mode = analytic_mode
        if self._use_jax:
            self._key = jax.random.PRNGKey(seed)
        else:
            self._rng = np.random.default_rng(seed)

    def _sample_weights_jax(
        self,
        mean_bounds: tuple[float, float],
        variance_bounds: tuple[float, float],
        return_params: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        if return_params:
            self._key, means, covs = generate_gmm_coefficients(
                key=self._key,
                dim=self._dim,
                num_components=self._num_components,
                mean_bounds=mean_bounds,
                variance_bounds=variance_bounds,
            )
            weights = jnp.ones((self._num_components,)) / self._num_components
            pdf = build_gmm_pdf(means, covs, weights)
            w = pdf(self._points)
            return w / jnp.sum(w), (means, covs, weights)

        pdf, self._key = get_gmm_pdf_jax(
            key=self._key,
            dim=self._dim,
            num_components=self._num_components,
            mean_bounds=mean_bounds,
            variance_bounds=variance_bounds,
        )
        w = pdf(self._points)
        return w / jnp.sum(w)

    def _rescale_covariances(
        self,
        covs_arr: np.ndarray,
        variance_bounds: tuple[float, float],
    ) -> np.ndarray:
        low, high = variance_bounds
        if low > 0 and high > 0:
            log_low = np.log(low)
            log_high = np.log(high)
            target_vars = np.exp(
                self._rng.uniform(log_low, log_high, size=covs_arr.shape[0])
            )
        else:
            target_vars = self._rng.uniform(low, high, size=covs_arr.shape[0])

        for k in range(covs_arr.shape[0]):
            avg_var = np.trace(covs_arr[k]) / covs_arr.shape[1]
            if avg_var <= 0:
                continue
            covs_arr[k] = covs_arr[k] * (target_vars[k] / avg_var)
        return covs_arr

    def _sample_weights_np(
        self,
        mean_bounds: tuple[float, float],
        variance_bounds: tuple[float, float],
        return_params: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        means_arr, covs_arr, weights = sample_gmm_params_wishart(
            K=self._num_components,
            d=self._dim,
            mean_bounds=mean_bounds,
            wishart_df=self._wishart_df,
            wishart_scale=self._wishart_scale,
            rng=self._rng,
        )
        covs_arr = self._rescale_covariances(covs_arr, variance_bounds)
        pdf = build_gmm_pdf_scipy(means_arr, covs_arr, weights)
        w = pdf(np.asarray(self._points))
        if return_params:
            return w / np.sum(w), (means_arr, covs_arr, weights)
        return w / np.sum(w)

    @staticmethod
    def _spd_sqrtm(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, eps, None)
        out = (vecs * np.sqrt(vals)) @ vecs.T
        return out.astype(mat.dtype, copy=False)

    @staticmethod
    def _gaussian_geodesic_cov(
        cov0: np.ndarray, cov1: np.ndarray, t: float, eps: float = 1e-12
    ) -> np.ndarray:
        """
        Closed-form W2 geodesic covariance between two Gaussians.
        Σ_t = ((1-t)I + t A) Σ0 ((1-t)I + t A)^T,
        A = Σ0^{-1/2} (Σ0^{1/2} Σ1 Σ0^{1/2})^{1/2} Σ0^{-1/2}.
        """
        sqrt0 = GaussianMixtureBarycenterGenerator._spd_sqrtm(cov0, eps)
        invsqrt0 = np.linalg.inv(sqrt0)
        middle = sqrt0 @ cov1 @ sqrt0
        middle_sqrt = GaussianMixtureBarycenterGenerator._spd_sqrtm(middle, eps)
        A = invsqrt0 @ middle_sqrt @ invsqrt0
        At = (1.0 - t) * np.eye(cov0.shape[0], dtype=cov0.dtype) + t * A
        cov_t = At @ cov0 @ At.T
        cov_t = 0.5 * (cov_t + cov_t.T)
        return cov_t

    def generate(
        self,
        *,
        num_marginals: int | None = None,
    ) -> Iterator[BarycenterProblem]:
        if num_marginals is None:
            num_marginals = self._num_marginals
        if num_marginals is None:
            raise ValueError("num_marginals must be provided")
        if num_marginals < 1:
            raise ValueError("num_marginals must be >= 1")

        axes = get_axes(
            self._dim,
            self._borders,
            self._n_points,
            cell_discretization=self.cell_discretization,
            use_jax=self._use_jax,
        )
        self._points = generate_nd_grid(axes, use_jax=self._use_jax)
        cell_volume = compute_cell_volume(axes, use_jax=self._use_jax)

        mean_bounds = (
            self._borders[0]
            + (self._borders[1] - self._borders[0]) * self._mean_from_borders_coef,
            self._borders[1]
            - (self._borders[1] - self._borders[0]) * self._mean_from_borders_coef,
        )
        variance_bounds = (
            abs(self._borders[1]) * self._variance_lower_bound_coef,
            abs(self._borders[1]) * self._variance_upper_bound_coef,
        )

        sampler = self._sample_weights_jax if self._use_jax else self._sample_weights_np

        xp = jnp if self._use_jax else np
        lambdas = xp.ones((num_marginals,))
        lambdas = lambdas / lambdas.sum()

        for _ in range(self._num_datasets):
            measures = []
            sampled_means = []
            sampled_covs = []
            for _ in range(num_marginals):
                want_params = (
                    self._analytic_when_possible and self._num_components == 1
                )
                res = sampler(mean_bounds, variance_bounds, return_params=want_params)
                if want_params:
                    w, (m_i, cov_i, _alpha) = res
                    sampled_means.append(np.asarray(m_i[0]))
                    sampled_covs.append(np.asarray(cov_i[0]))
                else:
                    w = res
                if self.cell_discretization == "cell-centered":
                    w = w * cell_volume
                w = w / w.sum()
                measures.append(
                    _build_measure(
                        self._points,
                        w,
                        axes,
                        self._measure_mode,
                        self._use_jax,
                    )
                )

            problem = BarycenterProblem(
                name=self._name,
                measures=measures,
                lambdas=lambdas,
                cost_fn=self._cost_fn,
            )

            if (
                self._analytic_when_possible
                and self._num_components == 1
                and len(sampled_means) == num_marginals
            ):
                try:
                    mean_star, cov_star = compute_gaussian_barycenter_analytic(
                        means=np.stack(sampled_means),
                        covs=np.stack(sampled_covs),
                        weights=np.asarray(lambdas),
                        mode=self._analytic_mode,
                    )
                    problem.analytic_barycenter = {
                        "mean": mean_star,
                        "cov": cov_star,
                        "mode": self._analytic_mode,
                    }
                except Exception as exc:  # keep generation robust
                    problem.analytic_barycenter_error = str(exc)

            yield problem


def _is_commuting_family(
    covs: np.ndarray, atol: float, rtol: float
) -> tuple[bool, float]:
    """
    Cheap approximate check whether all matrices in covs commute pairwise.
    Returns (is_commuting, max_commutator_norm).
    """
    n = covs.shape[0]
    max_comm = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            AB = covs[i] @ covs[j]
            BA = covs[j] @ covs[i]
            comm = AB - BA
            comm_norm = np.linalg.norm(comm, ord="fro")
            denom = max(
                np.linalg.norm(AB, ord="fro"),
                np.linalg.norm(BA, ord="fro"),
                1e-15,
            )
            bound = atol + rtol * denom
            max_comm = max(max_comm, comm_norm)
            if comm_norm > bound:
                return False, max_comm
    return True, max_comm


def _analytic_barycenter_commuting(
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Commuting SPD case: diagonalize once, average sqrt eigenvalues."""
    _, evecs = np.linalg.eigh(covs[0])
    diag_covs = np.einsum("ij,njk,kl->nil", evecs.T, covs, evecs)  # n,d,d

    for D in diag_covs:
        off = D - np.diag(np.diag(D))
        off_norm = np.linalg.norm(off, ord="fro")
        denom = max(np.linalg.norm(D, ord="fro"), 1e-15)
        if off_norm > atol + rtol * denom:
            raise ValueError(
                f"analytic barycenter (commuting) rejected: off-diagonal energy {off_norm:.3e} exceeds {atol + rtol * denom:.3e}"
            )

    diag_entries = diag_covs.diagonal(axis1=1, axis2=2)  # (n, d)
    sqrt_diag = np.sqrt(np.clip(diag_entries, 0.0, None))
    diag_star = np.square(weights @ sqrt_diag)
    cov_star = evecs @ np.diag(diag_star) @ evecs.T
    cov_star = 0.5 * (cov_star + cov_star.T)
    mean_star = weights @ means
    return mean_star, cov_star


def compute_gaussian_barycenter_analytic(
    means: ArrayLike,
    covs: ArrayLike,
    weights: ArrayLike,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
    mode: Literal["auto", "1d", "two_marginal", "commuting"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic Wasserstein-2 barycenter of Gaussian marginals in special cases.

    Supports:
      - 1D scalar variances: s_* = sum λ_i s_i, Σ_* = s_*^2.
      - Two marginals: closed-form W2 geodesic at t = λ_2.
      - Commuting covariances (simultaneously diagonalizable).

    Raises ValueError if the requested mode is not applicable or numerical
    checks fail. General SPD fixed-point iteration is intentionally not
    implemented here.
    """
    means = np.asarray(means)
    covs = np.asarray(covs)
    weights = np.asarray(weights)
    dtype = covs.dtype

    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError("covs must have shape (n, d, d)")
    n, d, _ = covs.shape
    if means.shape != (n, d):
        raise ValueError(f"means must have shape {(n, d)}, got {means.shape}")
    if weights.shape[0] != n:
        raise ValueError(f"weights length {weights.shape[0]} != n={n}")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    weights = weights / weights.sum()

    def _case_1d():
        s = np.sqrt(np.clip(covs.reshape(n, -1), 0.0, None)).reshape(n)
        s_star = np.dot(weights, s)
        var_star = s_star ** 2
        mean_star = np.dot(weights, means).reshape(1)
        return mean_star, var_star.reshape(1, 1)

    def _case_two():
        t = float(weights[1])
        mean_star = (1.0 - t) * means[0] + t * means[1]
        cov_star = GaussianMixtureBarycenterGenerator._gaussian_geodesic_cov(
            covs[0], covs[1], t
        )
        return mean_star, cov_star

    def _case_commuting():
        ok, max_comm = _is_commuting_family(covs, atol, rtol)
        if not ok:
            raise ValueError(
                f"analytic barycenter not available: non-commuting covariances, max commutator Frobenius norm {max_comm:.3e}"
            )
        mean_star, cov_star = _analytic_barycenter_commuting(
            means, covs, weights, atol=atol, rtol=rtol
        )
        return mean_star, cov_star

    if mode == "1d":
        if d != 1:
            raise ValueError("mode '1d' requires dimension 1")
        m, c = _case_1d()
        return m.astype(dtype, copy=False), c.astype(dtype, copy=False)
    if mode == "two_marginal":
        if n != 2:
            raise ValueError("mode 'two_marginal' requires exactly two marginals")
        m, c = _case_two()
        return m.astype(dtype, copy=False), c.astype(dtype, copy=False)
    if mode == "commuting":
        m, c = _case_commuting()
        return m.astype(dtype, copy=False), c.astype(dtype, copy=False)

    # auto: 1D -> two-marginal -> commuting
    if d == 1:
        m, c = _case_1d()
        return m.astype(dtype, copy=False), c.astype(dtype, copy=False)
    if n == 2:
        m, c = _case_two()
        return m.astype(dtype, copy=False), c.astype(dtype, copy=False)
    m, c = _case_commuting()
    return m.astype(dtype, copy=False), c.astype(dtype, copy=False)
