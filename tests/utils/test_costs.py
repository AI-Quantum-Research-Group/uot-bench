import jax.numpy as jnp
import numpy as np

from uot.utils.costs import cost_euclid, cost_euclid_squared
from uot.problems.base_problem import is_squared_euclidean_cost_fn

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from conftest import assert_is_jax_array


def _pts(n=5, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((n, dim)))


def test_cost_euclid_squared_shape():
    X, Y = _pts(), _pts(seed=1)
    C = cost_euclid_squared(X, Y)
    assert_is_jax_array(C)
    assert C.shape == (5, 5)


def test_cost_euclid_squared_nonnegative():
    X = _pts()
    C = cost_euclid_squared(X, X)
    assert jnp.all(C >= 0)


def test_cost_euclid_squared_zero_diagonal():
    X = _pts()
    C = cost_euclid_squared(X, X)
    np.testing.assert_allclose(np.asarray(jnp.diag(C)), 0.0, atol=1e-6)


def test_cost_euclid_shape():
    X, Y = _pts(), _pts(seed=1)
    C = cost_euclid(X, Y, use_jax=True)
    assert_is_jax_array(C)
    assert C.shape == (5, 5)


def test_cost_euclid_nonnegative():
    X, Y = _pts(), _pts(seed=1)
    C = cost_euclid(X, Y)
    assert np.all(np.asarray(C) >= 0)


def test_is_squared_euclidean_identifies_correctly():
    assert is_squared_euclidean_cost_fn(cost_euclid_squared)
    assert not is_squared_euclidean_cost_fn(cost_euclid)
