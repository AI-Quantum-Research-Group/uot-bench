import numpy as np
import pytest

from uot.data.measure import BaseMeasure, GridMeasure, PointCloudMeasure
import uot.problems.base_problem as base_problem_mod
from uot.problems.base_problem import MarginalProblem
from uot.problems.barycenter_problem import BarycenterProblem
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.costs import cost_euclid_squared


def test_two_marginal_problem_basic():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5]), name="mu")
    nu = PointCloudMeasure(np.array([[0.0], [2.0]]), np.array([0.4, 0.6]), name="nu")

    prob = TwoMarginalProblem("test2", mu, nu, cost_euclid_squared)

    costs = prob.get_costs()
    assert len(costs) == 1
    assert costs[0].shape == (2, 2)
    np.testing.assert_allclose(costs[0], np.array([[0.0, 4.0], [1.0, 1.0]]))

    inputs = prob.solver_inputs()
    assert inputs.cost_name == "cost_euclid_squared"
    assert inputs.is_squared_euclidean is True
    assert len(inputs.marginals) == 2
    assert len(inputs.costs) == 1

    d = prob.to_dict()
    assert d["dataset"] == "test2"
    assert d["type"] == "two_marginal"
    assert d["n_mu"] == 2
    assert d["n_nu"] == 2


def test_point_cloud_inputs_align_support_and_weights():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.4, 0.6]))
    prob = TwoMarginalProblem("aligned", mu, nu, cost_euclid_squared)

    inputs = prob.point_cloud_inputs(include_cost=True)

    np.testing.assert_allclose(inputs.support, np.array([[0.0], [1.0]]))
    np.testing.assert_allclose(inputs.weights[0], np.array([0.25, 0.75]))
    np.testing.assert_allclose(inputs.weights[1], np.array([0.4, 0.6]))
    assert inputs.cost.shape == (2, 2)


def test_point_cloud_inputs_union_remains_available():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(np.array([[1.0], [2.0]]), np.array([0.4, 0.6]))
    prob = TwoMarginalProblem("aligned", mu, nu, cost_euclid_squared)

    inputs = prob.point_cloud_inputs(shared_support="union", include_cost=True)

    np.testing.assert_allclose(inputs.support, np.array([[0.0], [1.0], [2.0]]))
    np.testing.assert_allclose(inputs.weights[0], np.array([0.25, 0.75, 0.0]))
    np.testing.assert_allclose(inputs.weights[1], np.array([0.0, 0.4, 0.6]))
    assert inputs.cost.shape == (3, 3)


def test_point_cloud_inputs_intersection_remains_available():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(np.array([[1.0], [2.0]]), np.array([0.4, 0.6]))
    prob = TwoMarginalProblem("intersection", mu, nu, cost_euclid_squared)

    inputs = prob.point_cloud_inputs(shared_support="intersection", include_cost=True)

    np.testing.assert_allclose(inputs.support, np.array([[1.0]]))
    np.testing.assert_allclose(inputs.weights[:, 0], np.array([0.75, 0.4]))
    assert inputs.cost.shape == (1, 1)


def test_same_support_mismatch_raises_clear_error():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(np.array([[1.0], [2.0]]), np.array([0.4, 0.6]))
    prob = TwoMarginalProblem("mismatch", mu, nu, cost_euclid_squared)

    with pytest.raises(ValueError, match="Use mode='union' or mode='intersection' explicitly"):
        prob.point_cloud_inputs()


def test_same_support_tolerance_accepts_near_equal_points():
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(
        np.array([[1e-7], [1.0 + 1e-7]]),
        np.array([0.4, 0.6]),
    )
    prob = TwoMarginalProblem("near-equal", mu, nu, cost_euclid_squared)

    inputs = prob.point_cloud_inputs(atol=1e-6)

    np.testing.assert_allclose(inputs.support, np.array([[0.0], [1.0]]))
    np.testing.assert_allclose(inputs.weights[1], np.array([0.4, 0.6]))


def test_grid_inputs_stack_grid_weights():
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    mu = GridMeasure(axes, np.array([[0.1, 0.2], [0.3, 0.4]]), normalize=False)
    nu = GridMeasure(axes, np.array([[0.4, 0.3], [0.2, 0.1]]), normalize=False)
    prob = TwoMarginalProblem("grid", mu, nu, cost_euclid_squared)

    inputs = prob.grid_inputs(include_cost=True)

    assert len(inputs.axes) == 2
    assert inputs.weights.shape == (2, 2, 2)
    assert inputs.cost.shape == (4, 4)
    assert inputs.is_squared_euclidean is True


def test_same_support_fast_path_handles_matching_grids():
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    mu = GridMeasure(axes, np.array([[0.1, 0.2], [0.3, 0.4]]), normalize=False)
    nu = GridMeasure(axes, np.array([[0.4, 0.3], [0.2, 0.1]]), normalize=False)
    prob = TwoMarginalProblem("grid-shared", mu, nu, cost_euclid_squared)

    inputs = prob.point_cloud_inputs(include_cost=False)

    np.testing.assert_allclose(
        inputs.weights,
        np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        ),
    )
    assert inputs.support.shape == (4, 2)


def test_same_support_grid_mismatch_raises():
    mu = GridMeasure(
        [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        normalize=False,
    )
    nu = GridMeasure(
        [np.array([0.0, 2.0]), np.array([0.0, 1.0])],
        np.array([[0.4, 0.3], [0.2, 0.1]]),
        normalize=False,
    )
    prob = TwoMarginalProblem("grid-mismatch", mu, nu, cost_euclid_squared)

    with pytest.raises(ValueError, match="Use mode='union' or mode='intersection' explicitly"):
        prob.point_cloud_inputs()


def test_barycenter_problem_point_cloud_inputs_include_lambdas():
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    measures = [
        GridMeasure(axes, np.array([[0.5, 0.0], [0.5, 0.0]]), normalize=False),
        GridMeasure(axes, np.array([[0.0, 0.5], [0.0, 0.5]]), normalize=False),
    ]
    lambdas = np.array([0.25, 0.75])
    problem = BarycenterProblem("bary", measures, cost_euclid_squared, lambdas=lambdas)

    inputs = problem.point_cloud_inputs(include_cost=True)

    np.testing.assert_allclose(inputs.lambdas, lambdas)
    assert inputs.weights.shape[0] == 2
    assert inputs.cost.shape[0] == inputs.support.shape[0]


class TrackingMeasure(BaseMeasure):
    def __init__(self, points, weights, *, fail_on_access: bool = False):
        self._points = np.asarray(points)
        self._weights = np.asarray(weights)
        self.fail_on_access = fail_on_access
        self.calls = 0

    def as_point_cloud(self, include_zeros: bool = True):
        self.calls += 1
        if self.fail_on_access:
            raise RuntimeError("unexpected support access")
        if include_zeros:
            return self._points, self._weights
        mask = self._weights > 0
        return self._points[mask], self._weights[mask]

    def get_jax(self):
        return self


def test_shared_support_first_does_not_materialize_other_supports():
    mu = TrackingMeasure([[0.0], [1.0]], [0.25, 0.75])
    nu = TrackingMeasure([[1.0], [2.0]], [0.4, 0.6], fail_on_access=True)
    prob = TwoMarginalProblem("first", mu, nu, cost_euclid_squared)

    support = prob.shared_support(mode="first")

    np.testing.assert_allclose(support, np.array([[0.0], [1.0]]))
    assert mu.calls == 1
    assert nu.calls == 0


def test_weights_on_shared_support_prepares_support_once_and_caches(monkeypatch):
    mu = PointCloudMeasure(np.array([[0.0], [1.0]]), np.array([0.25, 0.75]))
    nu = PointCloudMeasure(np.array([[1.0], [2.0]]), np.array([0.4, 0.6]))
    prob = TwoMarginalProblem("cache", mu, nu, cost_euclid_squared)

    calls = {"count": 0}
    original = base_problem_mod._prepare_alignment_support

    def counted_prepare(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(base_problem_mod, "_prepare_alignment_support", counted_prepare)

    first_support, first_weights = prob.weights_on_shared_support(mode="union")
    second_support, second_weights = prob.weights_on_shared_support(mode="union")

    assert calls["count"] == 1
    np.testing.assert_allclose(first_support, second_support)
    np.testing.assert_allclose(first_weights, second_weights)


def test_marginal_problem_is_abstract():
    with pytest.raises(TypeError):
        MarginalProblem("abstract", [], [])
