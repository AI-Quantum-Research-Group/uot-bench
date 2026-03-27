import numpy as np

from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator


def test_gaussian_mixture_produces_distinct_datasets():
    gen = GaussianMixtureGenerator(
        name="gmm",
        dim=1,
        num_components=2,
        n_points=10,
        num_datasets=2,
        borders=(-10, 10),
        use_jax=False,
        seed=52,
    )

    problems = list(gen.generate())
    problem0_marginals = problems[0].solver_inputs(include_cost=False).marginals
    problem1_marginals = problems[1].solver_inputs(include_cost=False).marginals
    assert not np.allclose(
        problem0_marginals[0].as_point_cloud()[1],
        problem1_marginals[0].as_point_cloud()[1],
    )
    assert not np.allclose(
        problem0_marginals[1].as_point_cloud()[1],
        problem1_marginals[1].as_point_cloud()[1],
    )


def test_gaussian_generator_one_and_grid_inputs_use_default_sqeuclidean():
    gen = GaussianMixtureGenerator(
        name="gmm",
        dim=2,
        num_components=1,
        n_points=8,
        num_datasets=1,
        borders=(-2.0, 2.0),
        use_jax=False,
        seed=0,
    )

    problem = gen.one()
    assert problem.is_squared_euclidean is True

    inputs = gen.grid_inputs(include_cost=False)
    assert inputs.weights.shape[0] == 2
    assert len(inputs.axes) == 2
