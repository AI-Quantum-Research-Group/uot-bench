import jax.numpy as jnp
import numpy as np
import pytest

from uot.data.measure import PointCloudMeasure
from uot.solvers import LBFGSTwoMarginalSolver, LBFGSPureSolver

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from conftest import assert_is_jax_array


def _pc_marginals_costs(n=8, dim=2, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    Y = rng.standard_normal((n, dim))
    w = np.ones(n) / n
    mu = PointCloudMeasure(jnp.array(X), jnp.array(w))
    nu = PointCloudMeasure(jnp.array(Y), jnp.array(w))
    C = jnp.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return [mu, nu], [jnp.array(C)]


@pytest.mark.parametrize("cls", [LBFGSTwoMarginalSolver, LBFGSPureSolver])
def test_lbfgs_constructs(cls):
    assert isinstance(cls(), cls)


@pytest.mark.parametrize("cls", [LBFGSTwoMarginalSolver, LBFGSPureSolver])
def test_lbfgs_returns_finite_cost(cls):
    marginals, costs = _pc_marginals_costs()
    result = cls().solve(marginals=marginals, costs=costs, reg=0.1)
    assert_is_jax_array(result["cost"])
    assert jnp.isfinite(result["cost"])
