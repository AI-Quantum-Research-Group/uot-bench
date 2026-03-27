import numpy as np
import pytest

from uot.data.measure import DiscreteMeasure, GridMeasure, PointCloudMeasure


def test_point_cloud_measure_basic():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    weights = np.array([0.3, 0.3, 0.4])

    measure = PointCloudMeasure(points, weights)
    points_out, weights_out = measure.as_point_cloud()

    assert np.allclose(points, points_out)
    assert np.allclose(weights, weights_out)
    assert np.allclose(measure.support(), points)
    assert np.allclose(measure.weights, weights)


def test_discrete_measure_is_compatibility_alias():
    points = np.array([[0.0], [1.0]])
    weights = np.array([0.25, 0.75])

    measure = DiscreteMeasure(points, weights)

    assert isinstance(measure, PointCloudMeasure)
    pts, wts = measure.as_point_cloud()
    assert np.allclose(pts, points)
    assert np.allclose(wts, weights)


def test_grid_measure_conversions():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    weights = np.array([[0.25, 0.25], [0.25, 0.25]])

    measure = GridMeasure([x, y], weights)

    axes, weights_nd = measure.as_grid()
    assert len(axes) == 2
    assert np.allclose(weights_nd, weights)

    points_out, weights_out = measure.as_point_cloud()
    expected_points = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert set(map(tuple, points_out.tolist())) == set(map(tuple, expected_points.tolist()))
    assert weights_out.shape == (4,)
    assert np.allclose(weights_out, np.array([0.25, 0.25, 0.25, 0.25]))


def test_grid_measure_zero_filtering():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    weights = np.array([[0.0, 0.2], [0.3, 0.5]])

    measure = GridMeasure([x, y], weights, normalize=False)
    points_out, weights_out = measure.as_point_cloud(include_zeros=False)

    assert points_out.shape == (3, 2)
    assert np.allclose(weights_out, np.array([0.2, 0.3, 0.5]))


def test_invalid_measure_types_raise_value_error():
    with pytest.raises(ValueError):
        PointCloudMeasure(points=np.array([[0, 0], [1, 2]]), weights=np.array([0.4]))

    with pytest.raises(ValueError):
        PointCloudMeasure(points=np.array([[0, 0]]), weights=np.array([-1.0]))

    with pytest.raises(ValueError):
        GridMeasure([np.array([0, 1, 2]), np.array([1])], np.ones((2, 2)))
