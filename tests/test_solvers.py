import numpy as np
import pytest
import jax.numpy as jnp

from uot.data.measure import GridMeasure
from uot.experiments.experiment import Experiment
from uot.problems.two_marginal import TwoMarginalProblem
from uot.solvers.back_and_forth.barycenter import (
    backnforth_barycenter_sqeuclidean_nd_jax,
    backnforth_barycenter_sqeuclidean_nd_optimized,
    solve_barycenter_back_and_forth,
)
from uot.solvers.back_and_forth.c_transform import c_transform_quadratic_fast
from uot.solvers.back_and_forth.forward_pushforward import cic_pushforward_nd
from uot.solvers.back_and_forth.method import backnforth_sqeuclidean_nd
from uot.solvers.back_and_forth.solver import BackNForthSqEuclideanSolver
from uot.utils.costs import cost_euclid_squared


def _grid_problem():
    axes = [np.array([0.25, 0.75])]
    mu = GridMeasure(axes, np.array([0.5, 0.5]), normalize=False)
    nu = GridMeasure(axes, np.array([0.5, 0.5]), normalize=False)
    return axes, mu, nu


def test_solver_resolves_c_transform_alias_import_string_and_pushforward_alias():
    alias_solver = BackNForthSqEuclideanSolver(
        pushforward_fn="cic",
        c_transform_fn="quadratic_fast",
    )
    assert alias_solver._pushforward_fn is cic_pushforward_nd
    assert alias_solver._c_transform_fn is c_transform_quadratic_fast

    import_solver = BackNForthSqEuclideanSolver(
        c_transform_fn="uot.solvers.back_and_forth.c_transform.c_transform_quadratic_fast"
    )
    assert import_solver._c_transform_fn is c_transform_quadratic_fast


def test_backnforth_sqeuclidean_nd_rejects_string_c_transform():
    with pytest.raises(TypeError, match="c_transform_fn must be callable"):
        backnforth_sqeuclidean_nd(
            mu=jnp.array([0.5, 0.5]),
            nu=jnp.array([0.5, 0.5]),
            coordinates=[jnp.array([0.25, 0.75])],
            stepsize=1.0,
            maxiterations=1,
            tolerance=1e-3,
            c_transform_fn="quadratic_fast",
        )


def test_backnforth_sqeuclidean_nd_rejects_string_pushforward():
    with pytest.raises(TypeError, match="pushforward_fn must be callable"):
        backnforth_sqeuclidean_nd(
            mu=jnp.array([0.5, 0.5]),
            nu=jnp.array([0.5, 0.5]),
            coordinates=[jnp.array([0.25, 0.75])],
            stepsize=1.0,
            maxiterations=1,
            tolerance=1e-3,
            pushforward_fn="cic",
        )


def test_backnforth_sqeuclidean_nd_uses_two_arg_c_transform():
    captured = {}
    coordinates = [jnp.array([0.25, 0.75])]

    def custom_c_transform(phi, coords_list):
        captured["coords"] = coords_list
        return c_transform_quadratic_fast(phi, coords_list)

    backnforth_sqeuclidean_nd(
        mu=jnp.array([0.5, 0.5]),
        nu=jnp.array([0.5, 0.5]),
        coordinates=coordinates,
        stepsize=1.0,
        maxiterations=1,
        tolerance=1e-3,
        c_transform_fn=custom_c_transform,
    )

    assert captured["coords"] is coordinates


def test_solver_threads_direct_two_arg_c_transform_into_pair_solver(monkeypatch):
    captured = {}

    def custom_c_transform(phi, coords_list):
        return phi + coords_list[0].sum()

    def fake_pair_solver(**kwargs):
        phi = jnp.zeros_like(kwargs["mu"])
        captured["output"] = np.asarray(kwargs["c_transform_fn"](phi, kwargs["coordinates"]))
        mu = kwargs["mu"]
        return (
            jnp.array(1, dtype=jnp.int32),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
        )

    monkeypatch.setattr(
        "uot.solvers.back_and_forth.solver.backnforth_sqeuclidean_nd",
        fake_pair_solver,
    )

    _, mu, nu = _grid_problem()
    solver = BackNForthSqEuclideanSolver(c_transform_fn=custom_c_transform)
    result = solver.solve(marginals=[mu, nu], costs=[])

    np.testing.assert_allclose(captured["output"], np.array([1.0, 1.0]))
    assert "cost" in result


def test_barycenter_jax_rejects_string_c_transform():
    measures = jnp.stack([jnp.array([0.5, 0.5]), jnp.array([0.25, 0.75])], axis=0)
    with pytest.raises(TypeError, match="c_transform_fn must be callable"):
        backnforth_barycenter_sqeuclidean_nd_jax(
            weights=jnp.array([0.5, 0.5]),
            measures=measures,
            coordinates=[jnp.array([0.25, 0.75])],
            c_transform_fn="quadratic_fast",
        )


def test_barycenter_jax_uses_two_arg_c_transform():
    captured = {}
    coordinates = [jnp.array([0.25, 0.75])]

    def custom_c_transform(phi, coords_list):
        captured["coords"] = coords_list
        return c_transform_quadratic_fast(phi, coords_list)

    measures = jnp.stack([jnp.array([0.5, 0.5]), jnp.array([0.25, 0.75])], axis=0)
    backnforth_barycenter_sqeuclidean_nd_jax(
        weights=jnp.array([0.5, 0.5]),
        measures=measures,
        coordinates=coordinates,
        transport_maxiter=1,
        outer_maxiter=1,
        c_transform_fn=custom_c_transform,
    )

    assert captured["coords"] is coordinates


def test_barycenter_optimized_resolves_aliases(monkeypatch):
    captured = {}

    def fake_barycenter(**kwargs):
        captured["pushforward_fn"] = kwargs["pushforward_fn"]
        captured["c_transform_fn"] = kwargs["c_transform_fn"]
        return kwargs["measures"][0], {"iterations": jnp.array(0)}

    monkeypatch.setattr(
        "uot.solvers.back_and_forth.barycenter.backnforth_barycenter_sqeuclidean_nd_jax",
        fake_barycenter,
    )

    backnforth_barycenter_sqeuclidean_nd_optimized(
        weights=jnp.array([0.5, 0.5]),
        measures_weights=[jnp.array([0.5, 0.5]), jnp.array([0.25, 0.75])],
        coordinates=[jnp.array([0.25, 0.75])],
        pushforward_fn="cic",
        c_transform_fn="quadratic_fast",
    )

    assert captured["pushforward_fn"] is cic_pushforward_nd
    assert captured["c_transform_fn"] is c_transform_quadratic_fast


def test_legacy_barycenter_uses_direct_two_arg_c_transform():
    captured = {}

    def custom_c_transform(phi, coords_list):
        captured["coords"] = coords_list
        return phi + 5.0

    def fake_pair_solver(**kwargs):
        phi = jnp.zeros_like(kwargs["mu"])
        captured["output"] = np.asarray(kwargs["c_transform_fn"](phi, kwargs["coordinates"]))
        mu = kwargs["mu"]
        return (
            jnp.array(1, dtype=jnp.int32),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
        )

    def identity_pushforward(mu, psi):
        return mu, psi

    coordinates = [jnp.array([0.25, 0.75])]
    solve_barycenter_back_and_forth(
        mu_list=[jnp.array([0.5, 0.5]), jnp.array([0.25, 0.75])],
        lambda_list=jnp.array([0.5, 0.5]),
        gamma=1.0,
        params={
            "coordinates": coordinates,
            "num_outer_iters": 1,
            "two_marginal_solver": fake_pair_solver,
            "pushforward_fn": identity_pushforward,
            "c_transform_fn": custom_c_transform,
        },
    )

    assert captured["coords"] is coordinates
    np.testing.assert_allclose(captured["output"], np.array([5.0, 5.0]))


def test_legacy_barycenter_resolves_string_c_transform():
    captured = {}

    def fake_pair_solver(**kwargs):
        captured["c_transform_fn"] = kwargs["c_transform_fn"]
        mu = kwargs["mu"]
        return (
            jnp.array(1, dtype=jnp.int32),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.zeros_like(mu),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
            jnp.ones((1,), dtype=mu.dtype),
        )

    def identity_pushforward(mu, psi):
        return mu, psi

    solve_barycenter_back_and_forth(
        mu_list=[jnp.array([0.5, 0.5]), jnp.array([0.25, 0.75])],
        lambda_list=jnp.array([0.5, 0.5]),
        gamma=1.0,
        params={
            "coordinates": [jnp.array([0.25, 0.75])],
            "num_outer_iters": 1,
            "two_marginal_solver": fake_pair_solver,
            "pushforward_fn": identity_pushforward,
            "c_transform_fn": "quadratic_fast",
        },
    )

    assert captured["c_transform_fn"] is c_transform_quadratic_fast


def test_experiment_run_on_problems_accepts_c_transform_init_kwarg(monkeypatch):
    def fake_solve(self, marginals, costs, **kwargs):
        return {"cost": 1.0}

    monkeypatch.setattr(BackNForthSqEuclideanSolver, "solve", fake_solve)

    _, mu, nu = _grid_problem()
    problem = TwoMarginalProblem("grid", mu, nu, cost_euclid_squared)
    experiment = Experiment(
        name="bfm",
        solve_fn=lambda problem, solver, marginals, costs, **kwargs: solver.solve(
            marginals=marginals, costs=costs, **kwargs
        ),
    )

    df = experiment.run_on_problems(
        [problem],
        solver=BackNForthSqEuclideanSolver,
        use_cost_matrix=False,
        c_transform_fn="quadratic_fast",
        pushforward_fn="cic",
    )

    assert df.iloc[0]["status"] == "success"
    assert df.iloc[0]["c_transform_fn"] == "quadratic_fast"
