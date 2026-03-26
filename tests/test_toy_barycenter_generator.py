from importlib import resources

import numpy as np
import pytest

import uot.assets.toy_shapes
from uot.data.dataset_loader import load_image_as_binary_grid
from uot.data.measure import GridMeasure
from uot.problems.generators import (
    FixedListSelector,
    ToyBarycenterGenerator,
)
from uot.utils.costs import cost_euclid_squared
from uot.utils.generator_helpers.get_axes import get_axes
from uot.utils.generator_helpers.shapes import (
    get_packaged_image_name_map,
    get_shape_factories,
    get_shape_fields,
    get_toy_source_factories,
    get_xy_grid,
)


def _build_toy_source_factories(n_points: int = 16) -> dict[str, object]:
    axes = get_axes(
        dim=2,
        borders=(0.0, 1.0),
        n_points=n_points,
        cell_discretization="cell-centered",
        use_jax=False,
    )
    X, Y = get_xy_grid(axes)
    return get_toy_source_factories(X, Y, n_points=n_points, use_jax=False)


def test_packaged_image_names_are_stems_and_follow_analytic_shapes():
    factories = _build_toy_source_factories()
    axes = get_axes(
        dim=2,
        borders=(0.0, 1.0),
        n_points=16,
        cell_discretization="cell-centered",
        use_jax=False,
    )
    analytic_names = list(get_shape_factories(*get_xy_grid(axes)).keys())
    available_names = list(factories.keys())

    assert available_names[: len(analytic_names)] == analytic_names

    image_names = available_names[len(analytic_names) :]
    assert image_names == sorted(image_names)
    assert "cristo_redentor" in image_names
    assert "motherland_monument_kyiv" in image_names
    assert "toronto_tower" in image_names
    assert all("." not in name for name in image_names)


def test_packaged_assets_are_discoverable():
    assets = resources.files(uot.assets.toy_shapes)
    found_assets = sorted(
        child.name
        for child in assets.iterdir()
        if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    assert "cristo_redentor.jpg" in found_assets
    assert "motherland_monument_kyiv.jpg" in found_assets
    assert "toronto_tower.jpg" in found_assets


def test_fixed_selector_supports_image_names():
    gen = ToyBarycenterGenerator(
        selector=FixedListSelector(names=("cristo_redentor", "toronto_tower")),
        n_points=16,
        cost_fn=cost_euclid_squared,
        num_datasets=1,
        measure_mode="grid",
    )

    problem = next(gen.generate(num_marginals=2))

    assert len(problem.get_marginals()) == 2
    assert all(isinstance(measure, GridMeasure) for measure in problem.get_marginals())


def test_mixed_selection_works():
    gen = ToyBarycenterGenerator(
        selector=FixedListSelector(names=("Ring", "cristo_redentor", "Star")),
        n_points=16,
        cost_fn=cost_euclid_squared,
        num_datasets=1,
        measure_mode="grid",
    )

    problem = next(gen.generate(num_marginals=3))
    measures = problem.get_marginals()

    assert len(measures) == 3
    assert all(isinstance(measure, GridMeasure) for measure in measures)
    assert all(np.asarray(measure.weights_nd).shape == (16, 16) for measure in measures)


def test_image_source_matches_notebook_loader_defaults():
    n_points = 32
    factories = _build_toy_source_factories(n_points=n_points)
    image_name_map = get_packaged_image_name_map()
    public_name = "cristo_redentor"
    asset = resources.files(uot.assets.toy_shapes).joinpath(image_name_map[public_name])

    with resources.as_file(asset) as asset_path:
        expected_measure = load_image_as_binary_grid(
            str(asset_path),
            size=(n_points, n_points),
            threshold=0.5,
            invert=True,
            normalize=True,
            axes_mode="normalized",
            use_jax=False,
        )

    actual_field = np.asarray(factories[public_name]())
    expected_field = np.asarray(expected_measure.weights_nd)

    assert actual_field.shape == (n_points, n_points)
    assert np.allclose(actual_field, expected_field)


def test_duplicate_public_image_name_is_rejected():
    with pytest.raises(ValueError, match="Duplicate toy source name 'duplicate'"):
        get_packaged_image_name_map(asset_names=("duplicate.jpg", "duplicate.png"))

    with pytest.raises(ValueError, match="Duplicate toy source name 'Ring'"):
        get_packaged_image_name_map(asset_names=("Ring.png",), analytic_names=("Ring",))


def test_analytic_only_selection_still_matches_shape_fields():
    n_points = 16
    axes = get_axes(
        dim=2,
        borders=(0.0, 1.0),
        n_points=n_points,
        cell_discretization="cell-centered",
        use_jax=False,
    )
    X, Y = get_xy_grid(axes)
    expected_fields = get_shape_fields(X, Y, shape_names=["Ring", "Star"])

    gen = ToyBarycenterGenerator(
        selector=FixedListSelector(names=("Ring", "Star")),
        n_points=n_points,
        cost_fn=cost_euclid_squared,
        num_datasets=1,
        measure_mode="grid",
    )

    problem = next(gen.generate(num_marginals=2))
    measures = problem.get_marginals()

    assert np.allclose(np.asarray(measures[0].weights_nd), np.asarray(expected_fields["Ring"]))
    assert np.allclose(np.asarray(measures[1].weights_nd), np.asarray(expected_fields["Star"]))
