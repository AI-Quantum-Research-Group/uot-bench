"""Map uot cost-function names to OTT-JAX CostFn instances."""

from __future__ import annotations


def _require_ott():
    try:
        import ott  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ott-jax is required for uot.interop.ott. "
            "Install it with: pip install uot-bench[ott]"
        ) from e


def cost_fn_for_name(cost_name: str | None):
    """Return an OTT CostFn for the given uot cost-function name.

    Falls back to SqEuclidean for unknown names.
    """
    _require_ott()
    from ott.geometry.costs import SqEuclidean, Euclidean, PNormP, Cosine

    _MAP = {
        "cost_euclid_squared": SqEuclidean,
        "cost_euclid": Euclidean,
        "cost_manhattan": lambda: PNormP(p=1),
        "cost_cosine": Cosine,
    }

    factory = _MAP.get(cost_name or "cost_euclid_squared", SqEuclidean)
    return factory() if isinstance(factory, type) else factory()
