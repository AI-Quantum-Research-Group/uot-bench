from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from uot.data.measure import (
    BaseMeasure,
    GridMeasure,
    PreparedAlignmentSupport,
    _align_weights_prepared,
    _prepare_alignment_support,
    _row_view,
    _snap_points,
    _supports_match,
)
from uot.utils.costs import cost_euclid_squared
from uot.utils.types import ArrayLike, Backend, ShareMode


def _qualified_name(obj: Callable | None) -> str | None:
    if obj is None:
        return None
    return f"{obj.__module__}.{getattr(obj, '__name__', type(obj).__name__)}"


def is_squared_euclidean_cost_fn(cost_fn: Callable | None) -> bool:
    if cost_fn is None:
        return False
    return cost_fn is cost_euclid_squared or _qualified_name(cost_fn) == _qualified_name(
        cost_euclid_squared
    )


@dataclass(frozen=True)
class SolverInputs:
    marginals: list[BaseMeasure]
    costs: list[ArrayLike]
    lambdas: ArrayLike | None
    cost_fn: Callable | None
    cost_name: str | None
    is_squared_euclidean: bool


@dataclass(frozen=True)
class PointCloudInputs:
    support: ArrayLike
    weights: ArrayLike
    cost: ArrayLike | None
    lambdas: ArrayLike | None
    cost_fn: Callable | None
    cost_name: str | None
    is_squared_euclidean: bool


@dataclass(frozen=True)
class GridInputs:
    axes: list[ArrayLike]
    weights: ArrayLike
    cost: ArrayLike | None
    lambdas: ArrayLike | None
    cost_fn: Callable | None
    cost_name: str | None
    is_squared_euclidean: bool


class Problem(ABC):
    """Base class for all optimal-transport problems.

    Subclass this to define a custom problem and plug it into an
    :class:`~uot.experiments.Experiment` or :func:`~uot.experiments.run_pipeline`.

    **Minimal subclass example**::

        import jax.numpy as jnp
        from uot import Problem
        from uot.data import PointCloudMeasure

        class MyProblem(Problem):
            def __init__(self, name, mu, nu, cost_fn):
                super().__init__(name, [mu, nu], [cost_fn])

            def get_marginals(self):
                return self.measures

            def get_costs(self):
                X, _ = self.measures[0].as_point_cloud()
                return [self.cost_fns[0](X, X)]

            def to_dict(self):
                return {"dataset": self.name}

            def free_memory(self):
                self._cost_cache = [None] * len(self.cost_fns)

    **Abstract methods** (must be implemented):

    - :meth:`get_marginals` — return list of :class:`~uot.data.BaseMeasure`
    - :meth:`get_costs` — return list of cost matrices as JAX arrays
    - :meth:`to_dict` — return a flat ``dict`` of metadata for DataFrame rows
    - :meth:`free_memory` — release any cached arrays

    **Optional overrides**:

    - :meth:`get_lambdas` — return barycenter weights (default: ``None``)
    """

    def __init__(
        self,
        name: str,
        measures: list[BaseMeasure],
        cost_fns: list[Callable[[ArrayLike, ArrayLike], ArrayLike]],
    ):
        super().__init__()
        if len(measures) < 2:
            raise ValueError("Need at least two marginals")
        self.name = name
        self.measures = measures
        self.cost_fns = cost_fns
        self._cost_cache = [None] * len(cost_fns)
        self._shared_support_cache: dict[tuple[str, bool, float, float], ArrayLike] = {}
        self._prepared_support_cache: dict[tuple[str, bool, float, float], PreparedAlignmentSupport] = {}
        self._aligned_weights_cache: dict[tuple[str, bool, float, float], ArrayLike] = {}
        self._view_cache: dict = {}
        self.__hash = None

    def __repr__(self):
        space_size = "x".join(
            str(marginal.as_point_cloud()[0].size)
            for marginal in self.get_marginals()
        )
        cost_name = self.cost_name or "unknown_cost"
        return f"<{self.__class__.__name__}[{self.name}] {space_size} with ({cost_name})>"

    @property
    def cost_fn(self) -> Callable | None:
        return self.cost_fns[0] if self.cost_fns else None

    @property
    def cost_name(self) -> str | None:
        fn = self.cost_fn
        return getattr(fn, "__name__", None) if fn is not None else None

    @property
    def is_squared_euclidean(self) -> bool:
        return is_squared_euclidean_cost_fn(self.cost_fn)

    def key(self) -> str:
        if self.__hash is None:
            blob = pickle.dumps(self, protocol=4)
            self.__hash = hashlib.sha1(blob).hexdigest()
        return self.__hash

    def __hash__(self) -> int:
        hex_key = self.key()[:16]
        return int(hex_key, 16)

    @abstractmethod
    def get_marginals(self) -> list[BaseMeasure]:
        """Return the marginal measures for this problem."""

    @abstractmethod
    def get_costs(self) -> list[ArrayLike]:
        """Return the cost matrices (as JAX arrays) for this problem."""

    def get_lambdas(self) -> ArrayLike | None:
        """Return barycenter weights, or ``None`` for two-marginal problems."""
        return None

    def post_solve_hooks(self) -> list:
        """Return problem-level post-solve hooks.

        Override to attach domain-specific hooks (e.g. image reconstruction)
        that should run after every solve call on this problem.  Each entry
        must match the :class:`~uot.experiments.hooks.PostSolveHook` protocol.
        """
        return []

    def solver_inputs(self, include_cost: bool = True) -> SolverInputs:
        return SolverInputs(
            marginals=self.get_marginals(),
            costs=self.get_costs() if include_cost else [],
            lambdas=self.get_lambdas(),
            cost_fn=self.cost_fn,
            cost_name=self.cost_name,
            is_squared_euclidean=self.is_squared_euclidean,
        )

    def shared_support(
        self,
        *,
        mode: ShareMode = "same",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> ArrayLike:
        key = self._shared_support_cache_key(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        if key not in self._shared_support_cache:
            self._shared_support_cache[key] = self._compute_shared_support(
                mode=mode,
                include_zeros=include_zeros,
                atol=atol,
                rtol=rtol,
            )
        return self._shared_support_cache[key]

    def weights_on_shared_support(
        self,
        *,
        mode: ShareMode = "same",
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[ArrayLike, ArrayLike]:
        key = self._shared_support_cache_key(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        support = self.shared_support(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        if key not in self._aligned_weights_cache:
            self._aligned_weights_cache[key] = self._compute_weights_on_shared_support(
                support=support,
                mode=mode,
                include_zeros=include_zeros,
                atol=atol,
                rtol=rtol,
            )
        return support, self._aligned_weights_cache[key]

    def _cost_on_support(self, support: ArrayLike) -> ArrayLike | None:
        if not self.cost_fns:
            return None
        if len(self.cost_fns) != 1:
            raise NotImplementedError(
                "Shared point-cloud cost only supports a single cost function."
            )
        return self.cost_fns[0](support, support)

    def point_cloud_inputs(
        self,
        *,
        shared_support: ShareMode = "same",
        include_cost: bool = True,
        include_zeros: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> PointCloudInputs:
        support, weights = self.weights_on_shared_support(
            mode=shared_support,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        cost = self._cost_on_support(support) if include_cost else None
        return PointCloudInputs(
            support=support,
            weights=weights,
            cost=cost,
            lambdas=self.get_lambdas(),
            cost_fn=self.cost_fn,
            cost_name=self.cost_name,
            is_squared_euclidean=self.is_squared_euclidean,
        )

    def grid_inputs(
        self,
        *,
        include_cost: bool = False,
        backend: Backend = "auto",
        dtype=None,
        device: jax.Device | None = None,  # type: ignore[name-defined]
    ) -> GridInputs:
        marginals = self.get_marginals()
        if not marginals:
            raise ValueError("Problem has no marginals.")
        if not all(isinstance(m, GridMeasure) for m in marginals):
            raise TypeError("grid_inputs requires all marginals to be GridMeasure instances.")

        grid_marginals = cast(list[GridMeasure], marginals)
        reference = grid_marginals[0]
        for marginal in grid_marginals[1:]:
            reference.check_compatible(marginal)

        axes, weights0 = reference.as_grid(
            backend=backend,
            dtype=dtype,
            device=device,  # type: ignore[arg-type]
        )
        xp = jnp if any(isinstance(ax, jax.Array) for ax in axes) or isinstance(weights0, jax.Array) else np
        weights = [weights0]
        for marginal in grid_marginals[1:]:
            _, weight_nd = marginal.as_grid(
                backend=backend,
                dtype=dtype,
                device=device,  # type: ignore[arg-type]
            )
            weights.append(weight_nd)

        cost = None
        if include_cost:
            costs = self.get_costs()
            cost = costs[0] if costs else None
            if cost is not None:
                cost = xp.asarray(cost, dtype=dtype) if dtype is not None else xp.asarray(cost)
                if xp is jnp and device is not None:
                    cost = jax.device_put(cost, device=device)

        return GridInputs(
            axes=axes,
            weights=xp.stack([xp.asarray(w) for w in weights], axis=0),
            cost=cost,
            lambdas=self.get_lambdas(),
            cost_fn=self.cost_fn,
            cost_name=self.cost_name,
            is_squared_euclidean=self.is_squared_euclidean,
        )

    def _shared_support_cache_key(
        self,
        *,
        mode: ShareMode,
        include_zeros: bool,
        atol: float,
        rtol: float,
    ) -> tuple[str, bool, float, float]:
        return (mode, bool(include_zeros), float(atol), float(rtol))

    def _shared_support_error(self) -> ValueError:
        return ValueError(
            "Marginals do not share the same support. "
            "Use mode='union' or mode='intersection' explicitly."
        )

    def _compute_shared_support(
        self,
        *,
        mode: ShareMode,
        include_zeros: bool,
        atol: float,
        rtol: float,
    ) -> ArrayLike:
        marginals = self.get_marginals()
        if not marginals:
            return np.zeros((0, 0))

        first = marginals[0]
        first_support = first.support(include_zeros=include_zeros)
        want_jax = isinstance(first_support, jax.Array)

        if mode == "first":
            return first_support

        if mode == "same":
            if include_zeros and all(isinstance(m, GridMeasure) for m in marginals):
                reference = cast(GridMeasure, first)
                for marginal in cast(list[GridMeasure], marginals)[1:]:
                    try:
                        reference.check_compatible(marginal, atol=atol, rtol=rtol)
                    except ValueError as exc:
                        raise self._shared_support_error() from exc
                return first_support

            for marginal in marginals[1:]:
                support = marginal.support(include_zeros=include_zeros)
                want_jax = want_jax or isinstance(support, jax.Array)
                if not _supports_match(first_support, support, atol=atol, rtol=rtol):
                    raise self._shared_support_error()
            return jnp.asarray(first_support) if want_jax else np.asarray(first_support)

        supports = [first_support]
        for marginal in marginals[1:]:
            support = marginal.support(include_zeros=include_zeros)
            supports.append(support)
            want_jax = want_jax or isinstance(support, jax.Array)

        supports_np = [_snap_points(s, atol=atol, rtol=rtol) for s in supports]
        if mode == "union":
            stacked = np.concatenate(supports_np, axis=0)
            view = _row_view(stacked)
            _, idx = np.unique(view, return_index=True)
            support = stacked[np.sort(idx)]
        elif mode == "intersection":
            support = supports_np[0]
            for other in supports_np[1:]:
                mask = np.isin(_row_view(support), _row_view(other))
                support = support[mask]
        else:
            raise ValueError("mode must be 'same', 'union', 'intersection', or 'first'")

        return jnp.asarray(support) if want_jax else support

    def _prepared_shared_support(
        self,
        *,
        mode: ShareMode,
        include_zeros: bool,
        atol: float,
        rtol: float,
    ):
        key = self._shared_support_cache_key(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        if key not in self._prepared_support_cache:
            self._prepared_support_cache[key] = _prepare_alignment_support(
                self.shared_support(
                    mode=mode,
                    include_zeros=include_zeros,
                    atol=atol,
                    rtol=rtol,
                ),
                atol=atol,
                rtol=rtol,
            )
        return self._prepared_support_cache[key]

    def _stack_weights(self, support: ArrayLike, weights: list[ArrayLike]) -> ArrayLike:
        want_jax = isinstance(support, jax.Array) or any(
            isinstance(weight, jax.Array) for weight in weights
        )
        xp = jnp if want_jax else np
        return xp.stack([xp.asarray(weight) for weight in weights], axis=0)

    def _compute_weights_on_shared_support(
        self,
        *,
        support: ArrayLike,
        mode: ShareMode,
        include_zeros: bool,
        atol: float,
        rtol: float,
    ) -> ArrayLike:
        marginals = self.get_marginals()
        first = marginals[0]

        if mode == "same":
            if include_zeros and all(isinstance(m, GridMeasure) for m in marginals):
                grid_marginals_same = cast(list[GridMeasure], marginals)
                reference_same = grid_marginals_same[0]
                _, first_weights = reference_same.as_grid()
                direct_weights = [first_weights.reshape(-1)]
                for marginal in grid_marginals_same[1:]:
                    try:
                        reference_same.check_compatible(marginal, atol=atol, rtol=rtol)
                    except ValueError as exc:
                        raise self._shared_support_error() from exc
                    _, other_weights = marginal.as_grid()
                    direct_weights.append(other_weights.reshape(-1))
                return self._stack_weights(support, direct_weights)

            points, weights = first.as_point_cloud(include_zeros=include_zeros)
            direct_weights = [weights]
            for marginal in marginals[1:]:
                other_points, other_weights = marginal.as_point_cloud(
                    include_zeros=include_zeros
                )
                if not _supports_match(points, other_points, atol=atol, rtol=rtol):
                    raise self._shared_support_error()
                direct_weights.append(other_weights)
            return self._stack_weights(support, direct_weights)

        if mode == "first" and include_zeros and all(
            isinstance(m, GridMeasure) for m in marginals
        ):
            grid_marginals_first = cast(list[GridMeasure], marginals)
            reference_first = grid_marginals_first[0]
            direct_weights = []
            compatible = True
            for marginal in grid_marginals_first:
                try:
                    reference_first.check_compatible(marginal, atol=atol, rtol=rtol)
                except ValueError:
                    compatible = False
                    break
                _, grid_weights = marginal.as_grid()
                direct_weights.append(grid_weights.reshape(-1))
            if compatible:
                return self._stack_weights(support, direct_weights)

        prepared_support = self._prepared_shared_support(
            mode=mode,
            include_zeros=include_zeros,
            atol=atol,
            rtol=rtol,
        )
        aligned_weights = []
        for marginal in marginals:
            points, weights = marginal.as_point_cloud(include_zeros=include_zeros)
            aligned_weights.append(
                _align_weights_prepared(
                    points,
                    weights,
                    prepared_support,
                    atol=atol,
                    rtol=rtol,
                )
            )
        return self._stack_weights(support, aligned_weights)

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a flat dict of metadata to be stored as a DataFrame row."""

    @abstractmethod
    def free_memory(self) -> None:
        """Release any large cached arrays (cost matrices, etc.)."""


import warnings as _warnings


def __getattr__(name: str):
    if name == "MarginalProblem":
        _warnings.warn(
            "MarginalProblem is deprecated and will be removed in a future release. "
            "Use Problem instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Problem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Keep a direct reference so existing `from uot.problems.base_problem import MarginalProblem`
# continues to work (triggers __getattr__ which emits the warning).
MarginalProblem = Problem
