"""Input-representation registry for the benchmarking pipeline.

A *representation* is a prepared view of a :class:`~uot.problems.base_problem.Problem`
that a solver can consume directly.  The default ``"marginals_costs"`` kind returns
a :class:`~uot.problems.base_problem.SolverInputs` dataclass — the existing contract
that all native solvers expect.

Third-party backends (e.g. OTT-JAX) register additional kinds so they can receive
pre-built geometry/problem objects without performing repeated translation inside the
timed ``solve()`` call.

Usage::

    from uot.experiments.representations import register_representation, build_representation

    # register a new kind (done once at import time by the backend package)
    register_representation("my_backend", lambda problem, **opts: _build(problem, **opts))

    # build a view (done by the runner, outside the timed region)
    view = build_representation(problem, "my_backend", epsilon=1e-2)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from uot.problems.base_problem import GridInputs, PointCloudInputs, Problem, SolverInputs


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_representation(kind: str, builder: Callable[..., Any]) -> None:
    """Register *builder* under *kind*.

    *builder* signature: ``(problem: Problem, **opts) -> view``.
    Unknown ``**opts`` keys must be accepted and ignored (use ``**_``).
    """
    _REGISTRY[kind] = builder


def build_representation(problem: Problem, kind: str, **opts: Any) -> Any:
    """Build (and cache) a representation view of *problem*.

    The result is cached on ``problem._view_cache`` keyed by ``(kind, sorted opts)``
    so re-runs (folds / replays) on the same object reuse the prepared view.

    Raises :class:`ValueError` if *kind* is not registered.
    """
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown representation kind {kind!r}.  "
            f"Registered: {sorted(_REGISTRY)}"
        )
    cache_key = (kind, _opts_key(opts))
    cache = problem._view_cache  # type: ignore[attr-defined]
    if cache_key not in cache:
        cache[cache_key] = _REGISTRY[kind](problem, **opts)
    return cache[cache_key]


def _opts_key(opts: dict[str, Any]) -> tuple:
    return tuple(sorted((k, _as_hashable(v)) for k, v in opts.items()))


def _as_hashable(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


# ── Built-in kind builders ────────────────────────────────────────────────────


def _build_marginals_costs(
    problem: Problem,
    *,
    include_cost: bool = True,
    **_: Any,
) -> SolverInputs:
    return problem.solver_inputs(include_cost=include_cost)


def _build_point_cloud(
    problem: Problem,
    *,
    shared_support: str = "same",
    include_cost: bool = True,
    include_zeros: bool = True,
    **_: Any,
) -> PointCloudInputs:
    return problem.point_cloud_inputs(
        shared_support=shared_support,  # type: ignore[arg-type]
        include_cost=include_cost,
        include_zeros=include_zeros,
    )


def _build_grid(
    problem: Problem,
    *,
    include_cost: bool = False,
    **_: Any,
) -> GridInputs:
    return problem.grid_inputs(include_cost=include_cost)


register_representation("marginals_costs", _build_marginals_costs)
register_representation("point_cloud", _build_point_cloud)
register_representation("grid", _build_grid)
