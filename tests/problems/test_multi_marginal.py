"""Tests for MultiMarginalProblem and a concrete 3-marginal example.

The concrete subclass here also serves as a reference for library users
writing their own multi-marginal problems.
"""
import jax
import jax.numpy as jnp
import numpy as np

from uot.data.measure import PointCloudMeasure
from uot.problems.multi_marginal import MultiMarginalProblem
from uot.utils.costs import cost_euclid_squared


class ThreeMarginalProblem(MultiMarginalProblem):
    """Concrete 3-marginal problem: pairwise squared-Euclidean costs."""

    def get_marginals(self):
        return self.measures

    def get_costs(self):
        pts = [m.as_point_cloud()[0] for m in self.measures]
        return [
            cost_euclid_squared(pts[0], pts[1]),
            cost_euclid_squared(pts[1], pts[2]),
            cost_euclid_squared(pts[0], pts[2]),
        ]

    def to_dict(self):
        sizes = [m.as_point_cloud()[1].shape[0] for m in self.measures]
        return {"dataset": self.name, "type": "three_marginal", "sizes": sizes}

    def free_memory(self):
        pass


def _make_problem(n=6, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    measures = []
    for _ in range(3):
        X = jnp.array(rng.standard_normal((n, dim)))
        w = jnp.ones(n) / n
        measures.append(PointCloudMeasure(X, w))
    cost_fns = [cost_euclid_squared] * 3
    return ThreeMarginalProblem("three_marginal_test", measures, cost_fns)


def test_multi_marginal_is_abstract():
    try:
        MultiMarginalProblem("x", [], [])
    except TypeError:
        pass  # expected — still abstract


def test_three_marginal_get_marginals():
    p = _make_problem()
    marginals = p.get_marginals()
    assert len(marginals) == 3
    for m in marginals:
        assert isinstance(m, PointCloudMeasure)


def test_three_marginal_get_costs_shape():
    p = _make_problem(n=6)
    costs = p.get_costs()
    assert len(costs) == 3
    for C in costs:
        assert isinstance(C, jax.Array)
        assert C.shape == (6, 6)


def test_three_marginal_to_dict():
    p = _make_problem()
    d = p.to_dict()
    assert d["dataset"] == "three_marginal_test"
    assert d["type"] == "three_marginal"
    assert d["sizes"] == [6, 6, 6]


def test_three_marginal_free_memory():
    p = _make_problem()
    p.free_memory()  # should not raise
