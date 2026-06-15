"""BaseSolver wrappers for OTT-JAX solvers.

Each class is a drop-in replacement for a uot-bench native solver.
Install ``ott-jax`` (``pip install uot-bench[ott]``) to use these.

Usage in YAML configs::

    solvers:
      ott-sinkhorn:
        solver: uot.interop.ott.OTTSinkhornSolver
        jit: false
        param-grid: sinkhorn

**Representation negotiation**: all OTT wrappers declare a non-default
``input_kind`` so the runner pre-builds the OTT problem object *outside* the
timed solve region.  Each ``solve`` method therefore receives the pre-built
OTT problem as its first positional argument instead of ``(marginals, costs)``.
"""

from __future__ import annotations

from typing import Any, Literal

from uot.solvers.base_solver import BaseSolver, SolverOutput
from uot.interop.ott import _outputs


def _require_ott():
    try:
        import ott  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ott-jax is required for uot.interop.ott solvers. "
            "Install it with: pip install uot-bench[ott]"
        ) from e


class OTTSinkhornSolver(BaseSolver):
    """OTT-JAX Sinkhorn solver wrapped as a uot BaseSolver.

    Hyperparameters passed here configure the OTT Sinkhorn solver object
    (structural params: lse_mode, threshold, max_iterations, …).
    Per-run parameters (epsilon) are baked into the pre-built LinearProblem
    geometry by the runner before the timed solve.
    """

    requires_squared_euclidean = False
    input_kind = "ott_linear"

    def __init__(
        self,
        lse_mode: bool = True,
        threshold: float = 1e-3,
        norm_error: int = 1,
        inner_iterations: int = 10,
        min_iterations: int = 0,
        max_iterations: int = 2000,
        momentum: Any = None,
        anderson: Any = None,
        parallel_dual_updates: bool = False,
        recenter_potentials: bool = False,
        use_danskin: bool | None = None,
        implicit_diff: Any = None,
        initializer: Any = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: float | str = 1.0,
        batch_size: int | None = None,
    ):
        _require_ott()
        self._lse_mode = lse_mode
        self._threshold = threshold
        self._norm_error = norm_error
        self._inner_iterations = inner_iterations
        self._min_iterations = min_iterations
        self._max_iterations = max_iterations
        self._momentum = momentum
        self._anderson = anderson
        self._parallel_dual_updates = parallel_dual_updates
        self._recenter_potentials = recenter_potentials
        self._use_danskin = use_danskin
        self._implicit_diff = implicit_diff
        self._initializer = initializer
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.scale_cost = scale_cost
        self.batch_size = batch_size

    def solve(  # type: ignore[override]
        self,
        linear_problem: Any,
        *,
        epsilon: float = 1e-2,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.linear.sinkhorn import Sinkhorn
        from ott.solvers.linear.implicit_differentiation import ImplicitDiff

        implicit_diff = self._implicit_diff
        if implicit_diff is None and self._anderson is None:
            implicit_diff = ImplicitDiff()

        solver = Sinkhorn(
            lse_mode=self._lse_mode,
            threshold=self._threshold,
            norm_error=self._norm_error,
            inner_iterations=self._inner_iterations,
            min_iterations=self._min_iterations,
            max_iterations=self._max_iterations,
            momentum=self._momentum,
            anderson=self._anderson,
            parallel_dual_updates=self._parallel_dual_updates,
            recenter_potentials=self._recenter_potentials,
            use_danskin=self._use_danskin,
            implicit_diff=implicit_diff,
            initializer=self._initializer,
        )
        out = solver(linear_problem)
        return _outputs.from_sinkhorn_output(out)


class OTTLRSinkhornSolver(BaseSolver):
    """OTT-JAX Low-Rank Sinkhorn solver wrapped as a uot BaseSolver.

    Stores the transport plan as low-rank factors in ``low_rank_plan``
    rather than materialising the full coupling matrix.
    """

    requires_squared_euclidean = False
    input_kind = "ott_linear"

    def __init__(
        self,
        rank: int = 10,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        lse_mode: bool = True,
        inner_iterations: int = 10,
        use_danskin: bool = True,
        initializer: Any = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: float | str = 1.0,
        batch_size: int | None = None,
    ):
        _require_ott()
        self._rank = rank
        self._gamma = gamma
        self._gamma_rescale = gamma_rescale
        self._lse_mode = lse_mode
        self._inner_iterations = inner_iterations
        self._use_danskin = use_danskin
        self._initializer = initializer
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.scale_cost = scale_cost
        self.batch_size = batch_size

    def solve(  # type: ignore[override]
        self,
        linear_problem: Any,
        *,
        epsilon: float = 1e-2,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.linear.sinkhorn_lr import LRSinkhorn

        solver = LRSinkhorn(
            rank=self._rank,
            gamma=self._gamma,
            gamma_rescale=self._gamma_rescale,
            lse_mode=self._lse_mode,
            inner_iterations=self._inner_iterations,
            use_danskin=self._use_danskin,
            initializer=self._initializer,
        )
        out = solver(linear_problem)
        return _outputs.from_lr_sinkhorn_output(out)


class OTTGromovWassersteinSolver(BaseSolver):
    """OTT-JAX Gromov–Wasserstein solver wrapped as a uot BaseSolver.

    The intra-space geometries are pre-built from each measure's self-cost
    by the runner.  Set ``fused_penalty > 0`` to run Fused GW.
    """

    requires_squared_euclidean = False
    input_kind = "ott_quadratic"

    def __init__(
        self,
        epsilon: float = 1e-2,
        relative_epsilon: Literal["mean", "std"] | None = None,
        warm_start: bool = False,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        fused_penalty: float = 0.0,
        scale_cost: float | str = 1.0,
        gw_unbalanced_correction: bool = True,
        ranks: int | tuple[int, ...] = -1,
        tolerances: float | tuple[float, ...] = 1e-2,
        # Inner Sinkhorn options
        linear_solver_threshold: float = 1e-3,
        linear_solver_max_iterations: int = 2000,
        linear_solver_lse_mode: bool = True,
    ):
        _require_ott()
        self._epsilon = epsilon
        self._relative_epsilon = relative_epsilon
        self._warm_start = warm_start
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.fused_penalty = fused_penalty
        self.scale_cost = scale_cost
        self.gw_unbalanced_correction = gw_unbalanced_correction
        self.ranks = ranks
        self.tolerances = tolerances
        self._lin_threshold = linear_solver_threshold
        self._lin_max_iter = linear_solver_max_iterations
        self._lin_lse_mode = linear_solver_lse_mode

    def solve(  # type: ignore[override]
        self,
        quadratic_problem: Any,
        *,
        epsilon: float | None = None,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.linear.sinkhorn import Sinkhorn
        from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein

        eps = epsilon if epsilon is not None else self._epsilon
        linear_solver = Sinkhorn(
            lse_mode=self._lin_lse_mode,
            threshold=self._lin_threshold,
            max_iterations=self._lin_max_iter,
        )
        solver = GromovWasserstein(
            linear_solver=linear_solver,
            epsilon=eps,
            relative_epsilon=self._relative_epsilon,
            warm_start=self._warm_start,
        )
        out = solver(quadratic_problem)
        return _outputs.from_gw_output(out)


class OTTLRGromovWassersteinSolver(BaseSolver):
    """OTT-JAX Low-Rank Gromov–Wasserstein solver wrapped as a uot BaseSolver."""

    requires_squared_euclidean = False
    input_kind = "ott_quadratic"

    def __init__(
        self,
        rank: int = 10,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        epsilon: float = 1e-2,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: float | str = 1.0,
        gw_unbalanced_correction: bool = True,
    ):
        _require_ott()
        self._rank = rank
        self._gamma = gamma
        self._gamma_rescale = gamma_rescale
        self._epsilon = epsilon
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.scale_cost = scale_cost
        self.gw_unbalanced_correction = gw_unbalanced_correction

    def solve(  # type: ignore[override]
        self,
        quadratic_problem: Any,
        *,
        epsilon: float | None = None,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein

        solver = LRGromovWasserstein(
            rank=self._rank,
            gamma=self._gamma,
            gamma_rescale=self._gamma_rescale,
        )
        out = solver(quadratic_problem)
        return _outputs.from_gw_output(out)


class OTTSinkhornDivergence(BaseSolver):
    """Sinkhorn divergence S_ε(μ,ν) = OT_ε(μ,ν) - ½(OT_ε(μ,μ) + OT_ε(ν,ν)).

    Uses ``input_kind = "marginals_costs"`` because sinkhorn_divergence builds
    three internal geometries (μ–ν, μ–μ, ν–ν) from raw point arrays.
    """

    requires_squared_euclidean = False
    input_kind = "marginals_costs"

    def __init__(
        self,
        lse_mode: bool = True,
        max_iterations: int = 2000,
        threshold: float = 1e-3,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        scale_cost: float | str = 1.0,
    ):
        _require_ott()
        self._lse_mode = lse_mode
        self._max_iterations = max_iterations
        self._threshold = threshold
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.scale_cost = scale_cost

    def solve(
        self,
        marginals: Any,
        costs: Any,
        *,
        epsilon: float = 1e-2,
        **kw: Any,
    ) -> SolverOutput:
        from ott.geometry.pointcloud import PointCloud
        from ott.tools.sinkhorn_divergence import sinkhorn_divergence

        import jax.numpy as jnp
        from uot.interop.ott._costs import cost_fn_for_name

        mu, nu = marginals[0], marginals[1]
        x_pts, x_wts = mu.as_point_cloud()
        y_pts, y_wts = nu.as_point_cloud()
        x_pts = jnp.asarray(x_pts)
        y_pts = jnp.asarray(y_pts)
        x_wts = jnp.asarray(x_wts)
        y_wts = jnp.asarray(y_wts)

        cost_fn = cost_fn_for_name(kw.get("cost_name"))

        divergence_val, out = sinkhorn_divergence(
            PointCloud,
            x_pts,
            y_pts,
            a=x_wts,
            b=y_wts,
            epsilon=epsilon,
            cost_fn=cost_fn,
            scale_cost=self.scale_cost,
            solve_kwargs={
                "lse_mode": self._lse_mode,
                "max_iterations": self._max_iterations,
                "threshold": self._threshold,
                "tau_a": self.tau_a,
                "tau_b": self.tau_b,
            },
        )
        return _outputs.from_sinkhorn_divergence_output(divergence_val, out)


class OTTDiscreteBarycenterSolver(BaseSolver):
    """OTT-JAX free-support discrete Wasserstein barycenter."""

    requires_squared_euclidean = False
    input_kind = "ott_barycenter"

    def __init__(
        self,
        max_iterations: int = 100,
        threshold: float = 1e-3,
        lse_mode: bool = True,
        scale_cost: float | str = 1.0,
    ):
        _require_ott()
        self._max_iterations = max_iterations
        self._threshold = threshold
        self._lse_mode = lse_mode
        self.scale_cost = scale_cost

    def solve(  # type: ignore[override]
        self,
        barycenter_problem: Any,
        *,
        epsilon: float = 1e-2,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.linear.continuous_barycenter import FreeWassersteinBarycenter

        solver = FreeWassersteinBarycenter(threshold=self._threshold)
        out = solver(barycenter_problem)
        return _outputs.from_discrete_barycenter_output(out)  # type: ignore[arg-type]


class OTTGWBarycenterSolver(BaseSolver):
    """OTT-JAX Gromov–Wasserstein barycenter solver."""

    requires_squared_euclidean = False
    input_kind = "ott_gw_barycenter"

    def __init__(
        self,
        max_iterations: int = 50,
        epsilon: float = 1e-2,
        scale_cost: float | str = 1.0,
        rank: int = -1,
    ):
        _require_ott()
        self._max_iterations = max_iterations
        self._epsilon = epsilon
        self.scale_cost = scale_cost
        self._rank = rank

    def solve(  # type: ignore[override]
        self,
        gw_barycenter_problem: Any,
        *,
        epsilon: float | None = None,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.quadratic.gw_barycenter import GromovWassersteinBarycenter

        eps = epsilon if epsilon is not None else self._epsilon
        solver = GromovWassersteinBarycenter(epsilon=eps)
        out = solver(gw_barycenter_problem)
        return _outputs.from_discrete_barycenter_output(out)


class OTTUnivariateSolver(BaseSolver):
    """OTT-JAX closed-form 1D optimal transport (sorting).

    Uses ``input_kind = "marginals_costs"`` because it requires raw 1D arrays
    that cannot be derived from a generic LinearProblem geometry.
    """

    requires_squared_euclidean = False
    input_kind = "marginals_costs"

    def __init__(self, cost_fn: Any = None):
        _require_ott()
        self._cost_fn = cost_fn

    def solve(
        self,
        marginals: Any,
        costs: Any,
        **kw: Any,
    ) -> SolverOutput:
        from ott.solvers.linear.univariate import UnivariateSolver

        import jax.numpy as jnp
        from uot.interop.ott._costs import cost_fn_for_name

        mu, nu = marginals[0], marginals[1]
        x_pts, x_wts = mu.as_point_cloud()
        y_pts, y_wts = nu.as_point_cloud()
        x_pts = jnp.asarray(x_pts).ravel()
        y_pts = jnp.asarray(y_pts).ravel()
        x_wts = jnp.asarray(x_wts)
        y_wts = jnp.asarray(y_wts)

        cost_fn = self._cost_fn or cost_fn_for_name(kw.get("cost_name", "cost_euclid"))
        solver = UnivariateSolver()
        out = solver(x_pts, y_pts, a=x_wts, b=y_wts, cost_fn=cost_fn)
        return {
            "cost": jnp.asarray(float(out.ot_cost)),
            "converged": True,
            "iterations": 0,
        }
