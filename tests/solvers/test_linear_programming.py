import jax.numpy as jnp
import numpy as np

from uot.data.measure import PointCloudMeasure
from uot.solvers import LinearProgrammingTwoMarginalSolver

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from conftest import assert_is_jax_array


def _pc_marginals_costs(n=4, dim=2, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    Y = rng.standard_normal((n, dim))
    w = np.ones(n) / n
    mu = PointCloudMeasure(jnp.array(X), jnp.array(w))
    nu = PointCloudMeasure(jnp.array(Y), jnp.array(w))
    C = jnp.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return [mu, nu], [jnp.array(C)]


def test_lp_constructs():
    assert isinstance(LinearProgrammingTwoMarginalSolver(), LinearProgrammingTwoMarginalSolver)


def test_lp_returns_finite_jax_cost():
    marginals, costs = _pc_marginals_costs()
    result = LinearProgrammingTwoMarginalSolver().solve(marginals=marginals, costs=costs)
    assert_is_jax_array(result["cost"])
    assert jnp.isfinite(result["cost"])


def test_lp_coupling_is_jax_array():
    marginals, costs = _pc_marginals_costs()
    result = LinearProgrammingTwoMarginalSolver().solve(marginals=marginals, costs=costs)
    if "coupling" in result and result["coupling"] is not None:
        assert_is_jax_array(result["coupling"])
