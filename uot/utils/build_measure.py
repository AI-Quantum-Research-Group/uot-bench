import numpy as np
from jax import numpy as jnp
from uot.data.measure import GridMeasure, PointCloudMeasure


def _build_measure(points, weights, axes, mode: str, use_jax: bool):
    """
    mode: 'grid' | 'point_cloud' | 'auto'
    - 'grid': return GridMeasure (reshape weights to ND)
    - 'point_cloud': return PointCloudMeasure (keep (N,d) + (N,))
    - 'auto': if points come from a tensor grid built by axes, prefer GridMeasure
    """
    normalized_mode = "point_cloud" if mode == "discrete" else mode
    if normalized_mode not in ("grid", "point_cloud", "auto"):
        raise ValueError("measure_mode must be 'grid', 'point_cloud', or 'auto'")

    shape = tuple((ax.shape[0] if hasattr(ax, "shape") else len(ax)) for ax in axes)

    xp = jnp if use_jax else np
    weights = xp.asarray(weights)

    if normalized_mode == "point_cloud":
        return PointCloudMeasure(points=points, weights=weights)

    # 'grid' or 'auto' -> emit GridMeasure
    weights_nd = weights.reshape(shape)  # sampler already evaluated on the grid ordering
    return GridMeasure(axes=axes, weights_nd=weights_nd, name="", normalize=False)
