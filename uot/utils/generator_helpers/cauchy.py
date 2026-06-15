import numpy as np
import jax.numpy as jnp

from scipy.special import gamma

from uot.utils.types import ArrayLike


def generate_random_covariance(
    dim: int,
    rng: np.random.Generator,
    scale_bounds: tuple[float, float] = (0.5, 2.0)
) -> np.ndarray:
    """
    Generate a random symmetric positive-definite covariance matrix of size (dim, dim).

    Returns:
        A (dim x dim) positive-definite numpy array.
    """
    if dim < 1:
        raise ValueError("dim must be a positive integer")

    # 1) random Gaussian matrix
    A = rng.standard_normal(size=(dim, dim))

    # 2) orthogonal basis via QR
    Q, _ = np.linalg.qr(A)

    # 3) eigenvalues in given bounds
    low, high = scale_bounds
    if low <= 0 or high <= low:
        raise ValueError("scale_bounds must be (low, high) with 0 < low < high")
    eigenvalues = rng.uniform(low=low, high=high, size=(dim,))

    # 4) build covariance
    cov = (Q * eigenvalues) @ Q.T
    return cov


def generate_cauchy_parameters(
    dim: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator
):
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    mean_start, mean_end = mean_bounds

    mean_start = np.full(dim, mean_start)
    mean_end = np.full(dim, mean_end)

    mean = mean_start + rng.uniform() * (mean_end - mean_start)

    return mean, generate_random_covariance(dim=dim, rng=rng)


def get_cauchy_pdf(
    dim: int,
    mean_bounds: tuple[float, float],
    rng: np.random.Generator,
    use_jax: bool = False,
):

    mean, cov = generate_cauchy_parameters(
        dim=dim, mean_bounds=mean_bounds, rng=rng)

    d = dim

    cov_inv_j = jnp.linalg.inv(cov) if use_jax else jnp.zeros((d, d))
    cov_inv_np = np.linalg.inv(cov) if not use_jax else np.zeros((d, d))

    def pdf_fn(X: ArrayLike):
        if use_jax:
            arr = jnp.asarray(X)
            if arr.ndim != 2 or arr.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")
            diff = arr - mean
            qf = jnp.einsum("nd,de,ne->n", diff, cov_inv_j, diff)
            numerator = gamma((d + 1) / 2)
            denominator = (
                gamma(0.5) * (np.pi ** (d / 2)) *
                (jnp.linalg.det(cov) ** 0.5) *
                (1 + qf) ** ((d + 1) / 2)
            )
            return numerator / denominator
        else:
            arr2 = np.asarray(X)
            if arr2.ndim != 2 or arr2.shape[1] != d:
                raise ValueError(f"Input to pdf_fn must be shape (N, {d}).")
            diff2 = arr2 - mean
            qf2 = np.einsum("nd,de,ne->n", diff2, cov_inv_np, diff2)
            numerator = gamma((d + 1) / 2)
            denominator = (
                gamma(0.5) * (np.pi ** (d / 2)) *
                (np.linalg.det(cov) ** 0.5) *
                (1 + qf2) ** ((d + 1) / 2)
            )
            return numerator / denominator

    return pdf_fn
