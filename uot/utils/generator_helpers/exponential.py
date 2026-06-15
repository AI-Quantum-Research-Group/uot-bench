import numpy as np
import jax.numpy as jnp


from uot.utils.types import ArrayLike


def get_exponential_pdf(
    scale_bounds: tuple[float, float],
    rng: np.random.Generator,
    use_jax: bool = False,
):

    scale_start, scale_end = scale_bounds
    scale = scale_start + rng.uniform() * (scale_end - scale_start)

    def pdf_fn(X: ArrayLike):
        if use_jax:
            arr = jnp.asarray(X)
            if arr.ndim != 2 or arr.shape[1] != 1:
                raise ValueError("Input to pdf_fn must be shape (N, 1).")
            return jnp.where(arr >= 0, scale * np.exp(-scale * arr), 0)
        else:
            arr2 = np.asarray(X)
            if arr2.ndim != 2 or arr2.shape[1] != 1:
                raise ValueError("Input to pdf_fn must be shape (N, 1).")
            return np.where(arr2 >= 0, scale * np.exp(-scale * arr2), 0)

    return pdf_fn
