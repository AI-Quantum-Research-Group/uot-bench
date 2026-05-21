import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uot.data.measure import PointCloudMeasure
from uot.solvers import SinkhornTwoMarginalSolver, SinkhornTwoMarginalLogJaxSolver

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from conftest import assert_is_jax_array


def _pc_measures(n=8, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    Y = rng.standard_normal((n, dim))
    w = np.ones(n) / n
    mu = PointCloudMeasure(jnp.array(X), jnp.array(w))
    nu = PointCloudMeasure(jnp.array(Y), jnp.array(w))
    C = jnp.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return [mu, nu], [jnp.array(C)]


@pytest.mark.parametrize("cls", [SinkhornTwoMarginalSolver, SinkhornTwoMarginalLogJaxSolver])
def test_sinkhorn_constructs(cls):
    solver = cls()
    assert isinstance(solver, cls)


@pytest.mark.parametrize("cls", [SinkhornTwoMarginalSolver, SinkhornTwoMarginalLogJaxSolver])
def test_sinkhorn_returns_finite_cost(cls):
    marginals, costs = _pc_measures()
    result = cls().solve(marginals=marginals, costs=costs, reg=0.1)
    assert_is_jax_array(result["cost"])
    assert jnp.isfinite(result["cost"])


@pytest.mark.parametrize("cls", [SinkhornTwoMarginalSolver, SinkhornTwoMarginalLogJaxSolver])
def test_sinkhorn_coupling_sums_to_one(cls):
    marginals, costs = _pc_measures()
    result = cls().solve(marginals=marginals, costs=costs, reg=0.1)
    if "coupling" in result and result["coupling"] is not None:
        coupling = result["coupling"]
        assert_is_jax_array(coupling)
        np.testing.assert_allclose(float(coupling.sum()), 1.0, atol=1e-4)
