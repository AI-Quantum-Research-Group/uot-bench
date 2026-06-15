from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import jax.numpy as jnp

from uot.utils.types import ArrayLike


def cost_euclid_squared(
    X: ArrayLike,
    Y: ArrayLike,
) -> ArrayLike:
    """
    Compute squared‐Euclidean cost matrix C where
      C[i,j] = ||X[i,:] - Y[j,:]||^2.
    """
    # X: (n, d) and Y: (m, d). Broadcasting gives (n, m, d).
    diffs = X[:, None, :] - Y[None, :, :]   # shape (n, m, d)
    return (diffs ** 2).sum(axis=2)         # shape (n, m)


def cost_euclid(
    X: ArrayLike,
    Y: ArrayLike,
    use_jax: bool = False
) -> ArrayLike:
    """
    Compute Euclidean cost matrix C where
      C[i,j] = ||X[i,:] - Y[j,:]||₂.

    If use_jax=True and JAX is installed, uses jax.numpy.sqrt; otherwise uses np.sqrt.
    """
    lib = jnp if use_jax else np

    diffs = X[:, None, :] - Y[None, :, :]   # shape (n, m, d)
    sq = (diffs ** 2).sum(axis=2)           # shape (n, m)
    return lib.sqrt(sq)                     # shape (n, m)


def cost_manhattan(
    X: ArrayLike,
    Y: ArrayLike,
    use_jax: bool = False
) -> ArrayLike:
    """
    Compute Manhattan (ℓ₁) cost matrix C where
      C[i,j] = sum_k |X[i,k] - Y[j,k]|.

    If use_jax=True and JAX is installed, uses jax.numpy.abs; otherwise uses np.abs.
    """
    lib = jnp if use_jax else np

    diffs = lib.abs(X[:, None, :] - Y[None, :, :])  # shape (n, m, d)
    return diffs.sum(axis=2)                        # shape (n, m)


def cost_cosine(
    X: ArrayLike,
    Y: ArrayLike,
    use_jax: bool = False,
    eps: float = 1e-8
) -> ArrayLike:
    """
    Compute cosine‐distance matrix C where
      C[i,j] = 1 - (X[i] · Y[j]) / (||X[i]|| * ||Y[j]||).

    If use_jax=True and JAX is installed, uses jax.numpy.linalg.norm and jax.numpy.dot;
    otherwise uses NumPy’s equivalents. In either case, returns an (n, m) array.
    """
    lib = jnp if use_jax else np

    # Dot products (n, d) @ (d, m) → (n, m)
    dots = X @ Y.T                                    # (n, m)

    # Compute norms: X_norms (n,), Y_norms (m,)
    X_norms = lib.linalg.norm(X, axis=1)             # (n,)
    Y_norms = lib.linalg.norm(Y, axis=1)             # (m,)

    # Avoid division by zero
    inv_norms = 1.0 / (X_norms[:, None] * Y_norms[None, :] + eps)

    cos_sim = dots * inv_norms                        # (n, m)
    return 1.0 - cos_sim                              # (n, m)


@dataclass(frozen=True)
class _CostEntry:
    uot_fn: Callable
    ott_factory: Optional[Callable]  # None when ott-jax is not installed


def _build_ott_factories() -> dict[str, Callable]:
    try:
        from ott.geometry.costs import SqEuclidean, Euclidean, PNormP, Cosine
        return {
            "cost_euclid_squared": lambda: SqEuclidean(),
            "cost_euclid": lambda: Euclidean(),
            "cost_manhattan": lambda: PNormP(p=1),
            "cost_cosine": lambda: Cosine(),
        }
    except ImportError:
        return {}


_OTT_FACTORIES = _build_ott_factories()

COST_REGISTRY: dict[str, _CostEntry] = {
    "cost_euclid_squared": _CostEntry(
        uot_fn=cost_euclid_squared,
        ott_factory=_OTT_FACTORIES.get("cost_euclid_squared"),
    ),
    "cost_euclid": _CostEntry(
        uot_fn=cost_euclid,
        ott_factory=_OTT_FACTORIES.get("cost_euclid"),
    ),
    "cost_manhattan": _CostEntry(
        uot_fn=cost_manhattan,
        ott_factory=_OTT_FACTORIES.get("cost_manhattan"),
    ),
    "cost_cosine": _CostEntry(
        uot_fn=cost_cosine,
        ott_factory=_OTT_FACTORIES.get("cost_cosine"),
    ),
}
