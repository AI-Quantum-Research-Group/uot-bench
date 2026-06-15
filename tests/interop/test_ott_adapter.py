"""Tests for the OTT-JAX interoperability adapter.

All tests are gated on ott-jax being installed; they are silently skipped
otherwise so the base test suite passes without the optional dependency.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

ott = pytest.importorskip("ott", reason="ott-jax is not installed (pip install uot-bench[ott])")

from uot.data.measure import GridMeasure, PointCloudMeasure
from uot.experiments.measurement import invoke_solver
from uot.experiments.representations import build_representation
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.costs import cost_euclid_squared


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_measures(n: int = 20, d: int = 2, seed: int = 0):
    """Return two small PointCloudMeasures for testing."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal((n, d)).astype(np.float32)
    a = np.ones(n, dtype=np.float32) / n
    b = np.ones(n, dtype=np.float32) / n
    mu = PointCloudMeasure(jnp.asarray(x), jnp.asarray(a))
    nu = PointCloudMeasure(jnp.asarray(y), jnp.asarray(b))
    return mu, nu


def _make_problem(n: int = 20, d: int = 2) -> TwoMarginalProblem:
    mu, nu = _make_measures(n=n, d=d)
    return TwoMarginalProblem("test", mu, nu, cost_euclid_squared)


def _solve(solver, prob, **kwargs):
    """Build a representation view and call invoke_solver — the standard pipeline path."""
    view = build_representation(prob, solver.input_kind, **kwargs)
    return invoke_solver(solver, view, **kwargs)


# ---------------------------------------------------------------------------
# Adapter primitives
# ---------------------------------------------------------------------------


class TestCostAdapter:
    def test_sq_euclidean(self):
        from uot.interop.ott._costs import cost_fn_for_name
        from ott.geometry.costs import SqEuclidean

        fn = cost_fn_for_name("cost_euclid_squared")
        assert isinstance(fn, SqEuclidean)

    def test_euclidean(self):
        from uot.interop.ott._costs import cost_fn_for_name
        from ott.geometry.costs import Euclidean

        fn = cost_fn_for_name("cost_euclid")
        assert isinstance(fn, Euclidean)

    def test_unknown_falls_back_to_sq_euclidean(self):
        from uot.interop.ott._costs import cost_fn_for_name
        from ott.geometry.costs import SqEuclidean

        fn = cost_fn_for_name("some_unknown_cost")
        assert isinstance(fn, SqEuclidean)


class TestProblemAdapter:
    def test_pointcloud_geometry(self):
        from uot.interop.ott._problems import two_measures_to_pointcloud
        from ott.geometry.pointcloud import PointCloud

        mu, nu = _make_measures()
        geom = two_measures_to_pointcloud(mu, nu, cost_name="cost_euclid_squared")
        assert isinstance(geom, PointCloud)

    def test_self_pointcloud_geometry(self):
        from uot.interop.ott._problems import measure_to_pointcloud
        from ott.geometry.pointcloud import PointCloud

        mu, _ = _make_measures()
        geom = measure_to_pointcloud(mu, cost_name="cost_euclid_squared")
        assert isinstance(geom, PointCloud)

    def test_linear_problem(self):
        from uot.interop.ott._problems import two_marginal_to_linear_problem
        from ott.problems.linear.linear_problem import LinearProblem

        mu, nu = _make_measures()
        prob = two_marginal_to_linear_problem(mu, nu, epsilon=0.1)
        assert isinstance(prob, LinearProblem)
        assert prob.a is not None
        assert prob.b is not None

    def test_quadratic_problem(self):
        from uot.interop.ott._problems import two_marginal_to_quadratic_problem
        from ott.problems.quadratic.quadratic_problem import QuadraticProblem

        mu, nu = _make_measures()
        prob = two_marginal_to_quadratic_problem(mu, nu, epsilon=0.1)
        assert isinstance(prob, QuadraticProblem)
        assert not prob.is_fused

    def test_two_marginal_to_ott_linear_problem(self):
        from ott.problems.linear.linear_problem import LinearProblem

        prob = _make_problem()
        ott_prob = prob.to_ott_linear_problem(epsilon=0.1)
        assert isinstance(ott_prob, LinearProblem)

    def test_measure_to_ott_geometry(self):
        from ott.geometry.pointcloud import PointCloud

        mu, nu = _make_measures()
        geom = mu.to_ott_geometry(nu, cost_name="cost_euclid_squared", epsilon=0.1)
        assert isinstance(geom, PointCloud)


class TestRepresentationBuilders:
    """Verify the OTT representation builders registered in the framework."""

    def test_ott_linear_view_is_linear_problem(self):
        from ott.problems.linear.linear_problem import LinearProblem

        prob = _make_problem()
        view = build_representation(prob, "ott_linear", epsilon=0.1)
        assert isinstance(view, LinearProblem)

    def test_ott_linear_view_skips_dense_cost_matrix(self):
        """Building an ott_linear view must NOT trigger get_costs()."""
        prob = _make_problem()
        assert prob._C is None
        build_representation(prob, "ott_linear", epsilon=0.1)
        assert prob._C is None, "Dense cost matrix was built but OTT does not need it"

    def test_ott_linear_view_cached(self):
        prob = _make_problem()
        v1 = build_representation(prob, "ott_linear", epsilon=0.1)
        v2 = build_representation(prob, "ott_linear", epsilon=0.1)
        assert v1 is v2, "Representation was not cached"

    def test_ott_quadratic_view_is_quadratic_problem(self):
        from ott.problems.quadratic.quadratic_problem import QuadraticProblem

        prob = _make_problem(n=10)
        view = build_representation(prob, "ott_quadratic", epsilon=0.1)
        assert isinstance(view, QuadraticProblem)

    def test_marginals_costs_view_matches_solver_inputs(self):
        from uot.problems.base_problem import SolverInputs

        prob = _make_problem()
        view = build_representation(prob, "marginals_costs", include_cost=True)
        assert isinstance(view, SolverInputs)
        assert len(view.marginals) == 2
        assert len(view.costs) == 1


# ---------------------------------------------------------------------------
# OTT Sinkhorn solver
# ---------------------------------------------------------------------------


class TestOTTSinkhornSolver:
    def test_returns_finite_cost(self):
        from uot.interop.ott import OTTSinkhornSolver

        prob = _make_problem()
        solver = OTTSinkhornSolver(max_iterations=500, threshold=1e-3)
        out = _solve(solver, prob, epsilon=0.1)

        assert math.isfinite(float(out["cost"]))

    def test_converged(self):
        from uot.interop.ott import OTTSinkhornSolver

        prob = _make_problem()
        solver = OTTSinkhornSolver(max_iterations=2000, threshold=1e-4)
        out = _solve(solver, prob, epsilon=0.1)
        assert out["converged"]

    def test_has_potentials(self):
        from uot.interop.ott import OTTSinkhornSolver

        prob = _make_problem()
        solver = OTTSinkhornSolver(max_iterations=500)
        out = _solve(solver, prob, epsilon=0.1)
        assert "u_final" in out
        assert "v_final" in out
        assert out["u_final"].shape[0] == 20

    def test_cost_key_always_present(self):
        from uot.interop.ott import OTTSinkhornSolver

        prob = _make_problem()
        solver = OTTSinkhornSolver()
        out = _solve(solver, prob, epsilon=0.5)
        assert "cost" in out


# ---------------------------------------------------------------------------
# Low-Rank Sinkhorn solver
# ---------------------------------------------------------------------------


class TestOTTLRSinkhornSolver:
    def test_returns_low_rank_plan(self):
        from uot.interop.ott import OTTLRSinkhornSolver

        prob = _make_problem()
        solver = OTTLRSinkhornSolver(rank=5)
        out = _solve(solver, prob, epsilon=0.1)

        assert math.isfinite(float(out["cost"]))
        assert "low_rank_plan" in out
        q, r, g = out["low_rank_plan"]
        assert q.shape[0] == 20
        assert r.shape[0] == 20
        assert g.shape[0] == 5

    def test_no_dense_transport_plan(self):
        from uot.interop.ott import OTTLRSinkhornSolver

        prob = _make_problem()
        solver = OTTLRSinkhornSolver(rank=5)
        out = _solve(solver, prob, epsilon=0.1)
        # We must NOT materialise the full n×m matrix by default.
        assert "transport_plan" not in out


# ---------------------------------------------------------------------------
# Gromov–Wasserstein solver
# ---------------------------------------------------------------------------


class TestOTTGromovWassersteinSolver:
    def test_returns_finite_cost(self):
        from uot.interop.ott import OTTGromovWassersteinSolver

        prob = _make_problem(n=10)
        solver = OTTGromovWassersteinSolver(
            epsilon=0.1,
            linear_solver_max_iterations=200,
        )
        out = _solve(solver, prob, epsilon=0.1)

        assert math.isfinite(float(out["cost"]))

    def test_converged(self):
        from uot.interop.ott import OTTGromovWassersteinSolver

        prob = _make_problem(n=10)
        solver = OTTGromovWassersteinSolver(epsilon=0.1)
        out = _solve(solver, prob, epsilon=0.1)
        assert out["converged"]

    def test_has_transport_plan(self):
        from uot.interop.ott import OTTGromovWassersteinSolver

        prob = _make_problem(n=10)
        solver = OTTGromovWassersteinSolver(epsilon=0.1)
        out = _solve(solver, prob, epsilon=0.1)
        assert "transport_plan" in out


# ---------------------------------------------------------------------------
# Sinkhorn divergence
# ---------------------------------------------------------------------------


class TestOTTSinkhornDivergence:
    def test_divergence_is_nonnegative(self):
        from uot.interop.ott import OTTSinkhornDivergence

        prob = _make_problem(n=15)
        solver = OTTSinkhornDivergence(max_iterations=500)
        out = _solve(solver, prob, epsilon=0.1)

        assert math.isfinite(float(out["cost"]))
        # Sinkhorn divergence is non-negative (debiased).
        assert float(out["cost"]) >= -1e-4  # allow small numerical error


# ---------------------------------------------------------------------------
# Importability when ott-jax is absent is tested at the module level above
# (pytest.importorskip handles this).
# ---------------------------------------------------------------------------
