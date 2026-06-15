import numpy as np
import pytest

from uot.data.measure import PointCloudMeasure
from uot.problems.generators.exponential_generator import ExponentialGenerator
from uot.problems.two_marginal import TwoMarginalProblem


@pytest.mark.parametrize("use_jax", [False, True])
def test_generate_raises_on_dimension(use_jax):
    with pytest.raises(ValueError):
        ExponentialGenerator(
            name="exp",
            dim=2,
            n_points=5,
            num_datasets=1,
            borders=(0.0, 1.0),
            use_jax=use_jax,
            seed=0,
        )


@pytest.mark.parametrize("use_jax,num_datasets", [(False, 2), (True, 3)])
def test_generate_returns_point_cloud_problems(use_jax, num_datasets):
    gen = ExponentialGenerator(
        name="exp",
        dim=1,
        n_points=10,
        num_datasets=num_datasets,
        borders=(0.0, 5.0),
        use_jax=use_jax,
        seed=123,
        measure_mode="point_cloud",
    )

    problems = list(gen.generate())
    assert len(problems) == num_datasets

    for problem in problems:
        assert isinstance(problem, TwoMarginalProblem)
        assert problem.is_squared_euclidean is True
        mu, nu = problem.solver_inputs(include_cost=False).marginals
        assert isinstance(mu, PointCloudMeasure)
        assert isinstance(nu, PointCloudMeasure)
        pmu, wmu = mu.as_point_cloud()
        pnu, wnu = nu.as_point_cloud()
        assert pmu.shape == (10, 1)
        assert pnu.shape == (10, 1)
        np.testing.assert_allclose(np.sum(wmu), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.sum(wnu), 1.0, atol=1e-6)
