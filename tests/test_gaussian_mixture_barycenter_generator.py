import numpy as np
import pytest

from uot.problems.generators.gaussian_mixture_barycenter_generator import (
    compute_gaussian_barycenter_analytic,
    GaussianMixtureBarycenterGenerator,
)


def _spd(mat):
    return 0.5 * (mat + mat.T)


def _spd_sqrtm(mat):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def test_barycenter_1d_matches_linear_std():
    rng = np.random.default_rng(0)
    n = 4
    weights = rng.random(n)
    weights = weights / weights.sum()
    stds = rng.uniform(0.1, 2.0, size=n)
    means = rng.normal(size=(n, 1))
    covs = stds[:, None, None] ** 2

    mean_star, cov_star = compute_gaussian_barycenter_analytic(
        means, covs, weights, mode="auto"
    )

    expected_std = np.dot(weights, stds)
    assert np.allclose(mean_star, weights @ means, atol=1e-10)
    assert np.allclose(cov_star[0, 0], expected_std**2, rtol=1e-6, atol=1e-8)


def test_barycenter_commuting_diagonalizable():
    rng = np.random.default_rng(1)
    n, d = 3, 3
    weights = rng.random(n)
    weights = weights / weights.sum()

    # random orthogonal basis
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    spectra = rng.uniform(0.5, 3.0, size=(n, d))
    covs = np.array([Q @ np.diag(s) @ Q.T for s in spectra])
    means = rng.normal(size=(n, d))

    mean_star, cov_star = compute_gaussian_barycenter_analytic(
        means, covs, weights, mode="auto"
    )

    expected_diag = np.square(weights @ np.sqrt(spectra))
    cov_expected = Q @ np.diag(expected_diag) @ Q.T

    assert np.allclose(mean_star, weights @ means, atol=1e-8)
    assert np.allclose(cov_star, cov_expected, rtol=1e-6, atol=1e-8)
    # SPD check
    eigs = np.linalg.eigvalsh(cov_star)
    assert np.all(eigs > 0)


def test_barycenter_two_marginal_matches_geodesic():
    rng = np.random.default_rng(2)
    d = 2
    t = 0.3
    weights = np.array([1 - t, t])
    A = rng.normal(size=(d, d))
    B = rng.normal(size=(d, d))
    cov0 = _spd(A @ A.T + np.eye(d))
    cov1 = _spd(B @ B.T + 2 * np.eye(d))
    means = rng.normal(size=(2, d))

    mean_star, cov_star = compute_gaussian_barycenter_analytic(
        means, np.stack([cov0, cov1]), weights, mode="auto"
    )

    sqrt0 = _spd_sqrtm(cov0)
    invsqrt0 = np.linalg.inv(sqrt0)
    middle = sqrt0 @ cov1 @ sqrt0
    middle_sqrt = _spd_sqrtm(middle)
    A_map = invsqrt0 @ middle_sqrt @ invsqrt0
    At = (1 - t) * np.eye(d) + t * A_map
    cov_expected = _spd(At @ cov0 @ At.T)
    mean_expected = (1 - t) * means[0] + t * means[1]

    assert np.allclose(mean_star, mean_expected, atol=1e-8)
    assert np.allclose(cov_star, cov_expected, rtol=1e-6, atol=1e-8)


def test_barycenter_two_marginal_matches_geodesic_in_4d():
    weights = np.array([0.75, 0.25])
    means = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.7, 0.5, 0.3],
        ]
    )
    cov0 = np.diag([0.4, 0.9, 1.6, 2.5])
    cov1 = np.diag([0.1, 0.25, 0.49, 0.81])

    mean_star, cov_star = compute_gaussian_barycenter_analytic(
        means, np.stack([cov0, cov1]), weights, mode="auto"
    )

    expected_std = (
        weights[0] * np.sqrt(np.diag(cov0))
        + weights[1] * np.sqrt(np.diag(cov1))
    )
    cov_expected = np.diag(expected_std**2)
    mean_expected = weights @ means

    assert np.allclose(mean_star, mean_expected, atol=1e-8)
    assert np.allclose(cov_star, cov_expected, rtol=1e-6, atol=1e-8)


def test_gaussian_mixture_barycenter_generator_supports_4d_grid_generation():
    gen = GaussianMixtureBarycenterGenerator(
        name="gmm-bary-4d",
        dim=4,
        n_points=3,
        num_datasets=1,
        borders=(0.0, 1.0),
        num_components=1,
        num_marginals=3,
        use_jax=False,
        seed=7,
    )

    problem = next(gen.generate())
    measures = problem.get_marginals()

    assert len(measures) == 3
    assert np.allclose(np.asarray(problem.get_lambdas()), np.full((3,), 1 / 3))

    for measure in measures:
        axes, weights_nd = measure.as_grid()
        points, weights = measure.as_point_cloud()
        assert len(axes) == 4
        assert np.asarray(weights_nd).shape == (3, 3, 3, 3)
        assert np.asarray(points).shape == (3**4, 4)
        assert np.asarray(weights).shape == (3**4,)
        assert np.isclose(np.asarray(weights_nd).sum(), 1.0)
        assert np.isclose(np.asarray(weights).sum(), 1.0)


def test_gaussian_mixture_barycenter_generator_validates_wishart_inputs():
    with pytest.raises(ValueError, match="wishart_df must be > dim - 1"):
        GaussianMixtureBarycenterGenerator(
            name="bad-df",
            dim=4,
            n_points=3,
            num_datasets=1,
            borders=(0.0, 1.0),
            use_jax=False,
            wishart_df=3,
        )

    with pytest.raises(ValueError, match=r"wishart_scale must have shape \(4, 4\)"):
        GaussianMixtureBarycenterGenerator(
            name="bad-scale",
            dim=4,
            n_points=3,
            num_datasets=1,
            borders=(0.0, 1.0),
            use_jax=False,
            wishart_scale=np.eye(3),
        )


def test_barycenter_non_commuting_raises():
    covs = np.array(
        [
            np.diag([1.0, 2.0]),
            np.array([[2.0, 1.0], [1.0, 2.0]]),
            np.array([[3.0, 0.2], [0.2, 4.0]]),
        ]
    )
    means = np.zeros((3, 2))
    weights = np.array([0.2, 0.3, 0.5])

    with pytest.raises(ValueError):
        compute_gaussian_barycenter_analytic(
            means, covs, weights, mode="auto", atol=1e-9, rtol=1e-6
        )
