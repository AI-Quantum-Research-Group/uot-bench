import numpy as np

from uot.data.measure import GridMeasure
from uot.problems.hdf5_store import HDF5ProblemStore
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.costs import cost_euclid_squared


def test_hdf5_roundtrip_preserves_grid_measure(tmp_path):
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    mu = GridMeasure(axes, np.array([[0.1, 0.2], [0.3, 0.4]]), normalize=False)
    nu = GridMeasure(axes, np.array([[0.4, 0.3], [0.2, 0.1]]), normalize=False)
    problem = TwoMarginalProblem("grid", mu, nu, cost_euclid_squared)

    store = HDF5ProblemStore(str(tmp_path / "problem_store.h5"))
    store.save(problem)
    key = store.all_problems()[0]

    loaded = store.load(key)
    store.close()

    loaded_inputs = loaded.grid_inputs(include_cost=False)
    assert loaded_inputs.weights.shape == (2, 2, 2)
    np.testing.assert_allclose(np.asarray(loaded_inputs.weights[0]), np.asarray(mu.weights_nd))
    np.testing.assert_allclose(np.asarray(loaded_inputs.weights[1]), np.asarray(nu.weights_nd))
