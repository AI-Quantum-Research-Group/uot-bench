from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

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


@dataclass(frozen=True)
class PreparedAlignmentSupport:
    snapped_support: np.ndarray
    support_view: np.ndarray
    order: np.ndarray
    sorted_support: np.ndarray


def _prepare_alignment_support(
    support,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> PreparedAlignmentSupport:
    support_np = _snap_points(support, atol=atol, rtol=rtol)
    support_view = _row_view(support_np)
    order = np.argsort(support_view)
    return PreparedAlignmentSupport(
        snapped_support=support_np,
        support_view=support_view,
        order=order,
        sorted_support=support_view[order],
    )


def _align_weights_prepared(
    points,
    weights,
    prepared_support: PreparedAlignmentSupport,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ArrayLike:
    points_np = _snap_points(points, atol=atol, rtol=rtol)
    support_np = prepared_support.snapped_support

    if points_np.shape[1] != support_np.shape[1]:
        raise ValueError(
            f"Support dimension mismatch: {points_np.shape[1]} vs {support_np.shape[1]}"
        )

    weights_np = np.asarray(weights)
    if weights_np.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights_np.shape}")
    if weights_np.shape[0] != points_np.shape[0]:
        raise ValueError(
            f"weights length {weights_np.shape[0]} does not match points {points_np.shape[0]}"
        )

    points_view = _row_view(points_np)
    idx = np.searchsorted(prepared_support.sorted_support, points_view)
    valid = idx < prepared_support.sorted_support.size
    idx = idx[valid]
    points_view = points_view[valid]
    weights_np = weights_np[valid]

    matches = prepared_support.sorted_support[idx] == points_view
    support_idx = prepared_support.order[idx[matches]]

    aligned = np.zeros((support_np.shape[0],), dtype=weights_np.dtype)
    np.add.at(aligned, support_idx, weights_np[matches])

    want_jax = _is_jax_array(weights) or _is_jax_array(points)
    return jnp.asarray(aligned) if want_jax else aligned


def _supports_match(
    left,
    right,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> bool:
    if left is right:
        return True
    left_np = _snap_points(left, atol=atol, rtol=rtol)
    right_np = _snap_points(right, atol=atol, rtol=rtol)
    if left_np.shape != right_np.shape:
        return False
    if atol <= 0 and rtol <= 0:
        return np.array_equal(left_np, right_np)
    return np.allclose(left_np, right_np, atol=atol, rtol=rtol)


def _align_weights(
    points,
    weights,
    support,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ArrayLike:
    prepared_support = _prepare_alignment_support(support, atol=atol, rtol=rtol)
    aligned = _align_weights_prepared(
        points,
        weights,
        prepared_support,
        atol=atol,
        rtol=rtol,
    )
    want_jax = _is_jax_array(support) or _is_jax_array(weights) or _is_jax_array(points)
    return jnp.asarray(aligned) if want_jax else aligned


class BaseMeasure(ABC):
    kind: str

    @abstractmethod
    def as_point_cloud(self, include_zeros: bool = True) -> tuple[ArrayLike, ArrayLike]:
        """Return `(points, weights)` in point-cloud form."""

    @abstractmethod
    def get_jax(self) -> "BaseMeasure":
        """Return the same measure represented with JAX arrays."""

    def to_discrete(self, include_zeros: bool = True) -> tuple[ArrayLike, ArrayLike]:
        return self.as_point_cloud(include_zeros=include_zeros)

    def support(self, include_zeros: bool = True) -> ArrayLike:
        points, _ = self.as_point_cloud(include_zeros=include_zeros)
        return points

    def weights_on(
        self,
        support: ArrayLike,
        *,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        points, weights = self.as_point_cloud(include_zeros=include_zeros)
        return _align_weights(points, weights, support, atol=atol, rtol=rtol)


class PointCloudMeasure(BaseMeasure):
    kind = "point_cloud"

    def __init__(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        name: str = "",
        normalize: bool = False,
    ):
        use_jax = _is_jax_array(points) or _is_jax_array(weights)
        xp = jnp if use_jax else np

        points_arr = xp.asarray(points)
        weights_arr = xp.asarray(weights)

        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(-1, 1)
        if points_arr.ndim != 2:
            raise ValueError(f"points must be 1D or 2D, got shape {points_arr.shape}")
        if weights_arr.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {weights_arr.shape}")
        if points_arr.shape[0] != weights_arr.shape[0]:
            raise ValueError(
                f"points length {points_arr.shape[0]} does not match weights length {weights_arr.shape[0]}"
            )
        if xp.any(~xp.isfinite(weights_arr)):
            raise ValueError("weights contain non-finite values")
        if xp.any(weights_arr < 0):
            raise ValueError("weights must be nonnegative")
        if xp.any(~xp.isfinite(points_arr)):
            raise ValueError("points contain non-finite values")

        if normalize:
            total = weights_arr.sum()
            weights_arr = xp.where(total > 0, weights_arr / total, weights_arr)

        self._points = points_arr
        self._weights = weights_arr
        self.name = name

    def get_jax(self) -> "PointCloudMeasure":
        if _is_jax_array(self._points) and _is_jax_array(self._weights):
            return self
        return PointCloudMeasure(
            points=jnp.asarray(self._points),
            weights=jnp.asarray(self._weights),
            name=self.name,
        )

    def as_point_cloud(self, include_zeros: bool = True) -> tuple[ArrayLike, ArrayLike]:
        if include_zeros:
            return self._points, self._weights
        mask = self._weights > 0
        return self._points[mask], self._weights[mask]

    @property
    def points(self) -> ArrayLike:
        return self._points

    @property
    def weights(self) -> ArrayLike:
        return self._weights


class DiscreteMeasure(PointCloudMeasure):
    """Compatibility alias for old serialized artifacts and imports."""


class GridMeasure(BaseMeasure):
    kind = "grid"

    def __init__(
        self,
        axes: list[ArrayLike],
        weights_nd: ArrayLike,
        name: str = "",
        normalize: bool = True,
    ):
        use_jax = _any_jax(axes) or _is_jax_array(weights_nd)
        xp = jnp if use_jax else np

        self._axes = [xp.asarray(ax) for ax in axes]
        self._weights_nd = xp.asarray(weights_nd)
        self.name = name

        if len(self._axes) != self._weights_nd.ndim:
            raise ValueError(
                f"axes len {len(self._axes)} != weights_nd.ndim {self._weights_nd.ndim}"
            )
        for i, ax in enumerate(self._axes):
            if ax.ndim != 1:
                raise ValueError(f"Axis {i} must be 1D, got {ax.shape}")
            if ax.shape[0] != self._weights_nd.shape[i]:
                raise ValueError(
                    f"Axis {i} length {ax.shape[0]} != weights dim {self._weights_nd.shape[i]}"
                )
            if xp.any(ax[1:] < ax[:-1]):
                raise ValueError(f"Axis {i} must be sorted ascending")

        if xp.any(~xp.isfinite(self._weights_nd)):
            raise ValueError("weights_nd contains non-finite values")
        if xp.any(self._weights_nd < 0):
            raise ValueError("weights_nd must be nonnegative")

        if normalize:
            total = self._weights_nd.sum()
            self._weights_nd = xp.where(
                total > 0,
                self._weights_nd / total,
                self._weights_nd,
            )

    def as_grid(
        self,
        *,
        backend: str = "auto",
        dtype=None,
        device: jax.Device | None = None,
        normalize: bool = False,
    ) -> tuple[list[ArrayLike], ArrayLike]:
        want_jax = (backend == "jax") or (
            backend == "auto" and (_any_jax(self._axes) or _is_jax_array(self._weights_nd))
        )
        xp = jnp if want_jax else np

        axes = [
            xp.asarray(ax, dtype=dtype) if dtype is not None else xp.asarray(ax)
            for ax in self._axes
        ]
        weights = (
            xp.asarray(self._weights_nd, dtype=dtype)
            if dtype is not None
            else xp.asarray(self._weights_nd)
        )

        if normalize:
            total = weights.sum()
            weights = xp.where(total > 0, weights / total, weights)

        if xp is jnp and device is not None:
            axes = [jax.device_put(ax, device=device) for ax in axes]
            weights = jax.device_put(weights, device=device)

        return axes, weights

    def for_grid_solver(
        self,
        *,
        backend: str = "auto",
        dtype=None,
        device: jax.Device | None = None,
        normalize: bool = False,
    ) -> tuple[list[ArrayLike], ArrayLike]:
        return self.as_grid(
            backend=backend,
            dtype=dtype,
            device=device,
            normalize=normalize,
        )

    def as_point_cloud(self, include_zeros: bool = True) -> tuple[ArrayLike, ArrayLike]:
        xp = jnp if _any_jax(self._axes) or _is_jax_array(self._weights_nd) else np
        mesh = xp.meshgrid(*self._axes, indexing="ij")
        points = xp.stack([m.reshape(-1) for m in mesh], axis=-1)
        weights = self._weights_nd.reshape(-1)
        if include_zeros:
            return points, weights
        mask = weights > 0
        return points[mask], weights[mask]

    def get_jax(self) -> "GridMeasure":
        if all(_is_jax_array(ax) for ax in self._axes) and _is_jax_array(self._weights_nd):
            return self
        return GridMeasure(
            axes=[jnp.asarray(ax) for ax in self._axes],
            weights_nd=jnp.asarray(self._weights_nd),
            name=self.name,
            normalize=False,
        )

    @property
    def axes(self) -> list[ArrayLike]:
        return self._axes

    @property
    def weights_nd(self) -> ArrayLike:
        return self._weights_nd

    def check_compatible(self, other: "GridMeasure", *, atol=1e-8, rtol=1e-7):
        if len(self._axes) != len(other._axes):
            raise ValueError("Grid dimensionality mismatch")
        for i, (a, b) in enumerate(zip(self._axes, other._axes)):
            if a.shape != b.shape:
                raise ValueError(f"Axis {i} length mismatch: {a.shape[0]} vs {b.shape[0]}")
            xp = jnp if _is_jax_array(a) or _is_jax_array(b) else np
            if not xp.allclose(a, b, atol=atol, rtol=rtol):
                raise ValueError(f"Axis {i} values differ beyond tolerances")
