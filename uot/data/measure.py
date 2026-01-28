import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence
from uot.utils.types import ArrayLike


def _is_jax_array(x) -> bool:
    return isinstance(x, jax.Array)


def _any_jax(seq: Sequence) -> bool:
    return any(_is_jax_array(x) for x in seq)


def _as_2d(points) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _snap_points(points, *, atol: float = 0.0, rtol: float = 0.0) -> np.ndarray:
    arr = _as_2d(points)
    if atol <= 0 and rtol <= 0:
        return arr
    scale = atol
    if scale <= 0:
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = rtol * max_abs
    if scale <= 0:
        return arr
    return np.round(arr / scale) * scale


def _row_view(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    return arr.view([("", arr.dtype)] * arr.shape[1]).reshape(-1)


def _align_weights(
    points,
    weights,
    support,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ArrayLike:
    points_np = _snap_points(points, atol=atol, rtol=rtol)
    support_np = _snap_points(support, atol=atol, rtol=rtol)

    if points_np.shape[1] != support_np.shape[1]:
        raise ValueError(
            f"Support dimension mismatch: {points_np.shape[1]} vs {support_np.shape[1]}"
        )

    weights_np = np.asarray(weights)
    if weights_np.shape[0] != points_np.shape[0]:
        raise ValueError(
            f"weights length {weights_np.shape[0]} does not match points {points_np.shape[0]}"
        )

    support_view = _row_view(support_np)
    points_view = _row_view(points_np)

    order = np.argsort(support_view)
    sorted_support = support_view[order]
    idx = np.searchsorted(sorted_support, points_view)
    valid = (idx < sorted_support.size)
    idx = idx[valid]
    points_view = points_view[valid]
    weights_np = weights_np[valid]

    matches = sorted_support[idx] == points_view
    support_idx = order[idx[matches]]

    aligned = np.zeros((support_np.shape[0],), dtype=weights_np.dtype)
    np.add.at(aligned, support_idx, weights_np[matches])

    want_jax = _is_jax_array(support) or _is_jax_array(weights) or _is_jax_array(points)
    return jnp.asarray(aligned) if want_jax else aligned


class BaseMeasure(ABC):
    @abstractmethod
    def to_discrete(self, include_zeros: bool = True) -> tuple[ArrayLike, ArrayLike]:
        """Return (points, weights) that approximate this measure as the discrete one"""
        ...

    @abstractmethod
    def support(self, include_zeros: bool = True) -> ArrayLike:
        """Return support points for this measure."""
        ...

    @abstractmethod
    def weights_on(
        self,
        support: ArrayLike,
        *,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        """Return weights aligned to a provided support."""
        ...


class DiscreteMeasure(BaseMeasure):
    def __init__(self, points: ArrayLike, weights: ArrayLike, name: str = ""):
        self._points = points
        self._weights = weights
        self.name = name

    def get_jax(self) -> 'DiscreteMeasure':
        return DiscreteMeasure(
            points=jnp.asarray(self._points),
            weights=jnp.asarray(self._weights),
            name=self.name,
        )

    def to_discrete(self, include_zeros: bool = True):
        if include_zeros:
            return self._points, self._weights
        mask = self._weights > 0
        return self._points[mask], self._weights[mask]

    def support(self, include_zeros: bool = True) -> ArrayLike:
        points, _ = self.to_discrete(include_zeros=include_zeros)
        return points

    def weights_on(
        self,
        support: ArrayLike,
        *,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        points, weights = self.to_discrete(include_zeros=include_zeros)
        return _align_weights(points, weights, support, atol=atol, rtol=rtol)


class GridMeasure:
    def __init__(self,
                 axes: List[ArrayLike],
                 weights_nd: ArrayLike,
                 name: str = "",
                 normalize: bool = True):
        # Pick backend: JAX if any input is JAX, else NumPy
        use_jax = _any_jax(axes) or _is_jax_array(weights_nd)
        xp = jnp if use_jax else np

        self._axes = [xp.asarray(ax) for ax in axes]
        self._weights_nd = xp.asarray(weights_nd)
        self.name = name

        # Shape & monotonicity checks
        if len(self._axes) != self._weights_nd.ndim:
            raise ValueError(f"axes len {len(self._axes)} != weights_nd.ndim {self._weights_nd.ndim}")
        for i, ax in enumerate(self._axes):
            if ax.ndim != 1:
                raise ValueError(f"Axis {i} must be 1D, got {ax.shape}")
            if ax.shape[0] != self._weights_nd.shape[i]:
                raise ValueError(f"Axis {i} length {ax.shape[0]} != weights dim {self._weights_nd.shape[i]}")
            if xp.any(ax[1:] < ax[:-1]):
                raise ValueError(f"Axis {i} must be sorted ascending")

        if xp.any(~xp.isfinite(self._weights_nd)):
            raise ValueError("weights_nd contains non-finite values")
        if xp.any(self._weights_nd < 0):
            raise ValueError("weights_nd must be nonnegative")

        if normalize:
            total = self._weights_nd.sum()
            self._weights_nd = xp.where(total > 0, self._weights_nd / total, self._weights_nd)

    # === NEW: solver-ready output for grid solvers ===
    def for_grid_solver(self,
                        *,
                        backend: str = "auto",        # "auto" | "jax" | "numpy"
                        dtype=None,                   # e.g. jnp.float64
                        device: jax.Device | None = None,
                        normalize: bool = False
                        ) -> Tuple[List[ArrayLike], ArrayLike]:
        """Return ([axis0,...,axis_{d-1}], weights_nd) in a form ready for grid solvers.
           - Axes are IJ-ordering (match meshgrid(indexing='ij')).
           - Backend/dtype/device honored without host<->device copies.
        """
        want_jax = (backend == "jax") or (backend == "auto" and (_any_jax(self._axes) or _is_jax_array(self._weights_nd)))
        xp = jnp if want_jax else np

        axes = [xp.asarray(ax, dtype=dtype) if dtype is not None else xp.asarray(ax) for ax in self._axes]
        W = xp.asarray(self._weights_nd, dtype=dtype) if dtype is not None else xp.asarray(self._weights_nd)

        if normalize:
            total = W.sum()
            W = xp.where(total > 0, W / total, W) if xp is jnp else (W / total if total > 0 else W)

        if xp is jnp and device is not None:
            axes = [jax.device_put(ax, device=device) for ax in axes]
            W = jax.device_put(W, device=device)

        return axes, W

    # Useful when you must flatten back to a point cloud
    def to_discrete(self, include_zeros: bool = False):
        xp = jnp if _any_jax(self._axes) or _is_jax_array(self._weights_nd) else np
        mesh = xp.meshgrid(*self._axes, indexing="ij")
        points = xp.stack([m.reshape(-1) for m in mesh], axis=-1)   # (N,d)
        weights = self._weights_nd.reshape(-1)                      # (N,)
        if include_zeros:
            return points, weights
        mask = weights > 0
        return points[mask], weights[mask]

    def get_jax(self) -> 'GridMeasure':
        if all(_is_jax_array(ax) for ax in self._axes) and _is_jax_array(self._weights_nd):
            return self
        return GridMeasure(
            axes=[jnp.asarray(ax) for ax in self._axes],
            weights_nd=jnp.asarray(self._weights_nd),
            name=self.name,
            normalize=False,
        )

    def support(self, include_zeros: bool = True) -> ArrayLike:
        points, _ = self.to_discrete(include_zeros=include_zeros)
        return points

    def weights_on(
        self,
        support: ArrayLike,
        *,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        points, weights = self.to_discrete(include_zeros=include_zeros)
        return _align_weights(points, weights, support, atol=atol, rtol=rtol)

    # Small utilities
    @property
    def axes(self) -> List[ArrayLike]:
        return self._axes

    @property
    def weights_nd(self) -> ArrayLike:
        return self._weights_nd

    def check_compatible(self, other: 'GridMeasure', *, atol=1e-8, rtol=1e-7):
        if len(self._axes) != len(other._axes):
            raise ValueError("Grid dimensionality mismatch")
        for i, (a, b) in enumerate(zip(self._axes, other._axes)):
            if not (a.shape == b.shape):
                raise ValueError(f"Axis {i} length mismatch: {a.shape[0]} vs {b.shape[0]}")
            xp = jnp if _is_jax_array(a) or _is_jax_array(b) else np
            if not xp.allclose(a, b, atol=atol, rtol=rtol):
                raise ValueError(f"Axis {i} values differ beyond tolerances")
