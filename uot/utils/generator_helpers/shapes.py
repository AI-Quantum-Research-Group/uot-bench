from collections.abc import Callable

import jax.numpy as jnp

from uot.utils.types import ArrayLike

DEFAULT_SHAPE_NAMES = ("Ring", "Crescent", "Checker")


def get_xy_grid(axes: list[ArrayLike]) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(axes) != 2:
        raise ValueError("get_xy_grid expects exactly 2 axes for 2D shapes")
    X, Y = jnp.meshgrid(axes[0], axes[1], indexing="ij")
    return X, Y


def normalize_field(field: ArrayLike, eps: float = 1e-9) -> jnp.ndarray:
    field = jnp.clip(jnp.asarray(field, dtype=jnp.float64), 0.0, None)
    total = field.sum()
    return field / total if total > eps else field


def diamond_mask(X: ArrayLike, Y: ArrayLike, center_x: float, center_y: float, radius: float) -> jnp.ndarray:
    return (jnp.abs(X - center_x) + jnp.abs(Y - center_y)) <= radius


def gaussian2d(
    X: ArrayLike,
    Y: ArrayLike,
    mx: float,
    my: float,
    sx: float,
    sy: float,
) -> jnp.ndarray:
    return jnp.exp(-0.5 * (((X - mx) / sx) ** 2 + ((Y - my) / sy) ** 2))


def triangle_sdf_band(
    X: ArrayLike,
    Y: ArrayLike,
    cx: float = 0.5,
    cy: float = 0.5,
    R: float = 0.30,
    theta0: float = jnp.pi / 2,
    thickness: float = 0.02,
) -> jnp.ndarray:
    """
    Thick outline (band) of an equilateral triangle.
    Returns boolean mask of shape (n,n).
    """
    p = jnp.stack([X - cx, Y - cy], axis=-1)  # (n,n,2)

    # triangle vertices (CCW) on circumcircle radius R
    angles = theta0 + jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
    V = jnp.stack([R * jnp.cos(angles), R * jnp.sin(angles)], axis=1)  # (3,2)

    # edges and outward normals for CCW polygon
    Vn = jnp.roll(V, shift=-1, axis=0)  # next vertex
    E = Vn - V  # (3,2)
    n_out = jnp.stack([E[:, 1], -E[:, 0]], axis=1)  # right normal = outward for CCW
    n_out = n_out / (jnp.linalg.norm(n_out, axis=1, keepdims=True) + 1e-12)

    # signed distance-ish: max over half-space constraints
    # inside => all dot(n_out, p - V[i]) <= 0
    # sdf = max_i dot(n_out[i], p - V[i])
    d = jnp.einsum("...j,ij->...i", p, n_out) - jnp.einsum("ij,ij->i", V, n_out)  # (...,3)
    sdf = jnp.max(d, axis=-1)  # (n,n)

    # band around boundary
    return jnp.abs(sdf) <= thickness


def david_star_density(
    X: ArrayLike,
    Y: ArrayLike,
    cx: float = 0.5,
    cy: float = 0.5,
    R: float = 0.30,
    thickness: float = 0.02,
) -> jnp.ndarray:
    up = triangle_sdf_band(X, Y, cx, cy, R, theta0=jnp.pi / 2, thickness=thickness)
    dn = triangle_sdf_band(X, Y, cx, cy, R, theta0=-jnp.pi / 2, thickness=thickness)
    star = up | dn
    return normalize_field(star.astype(jnp.float64))


def ring_density(
    X: ArrayLike,
    Y: ArrayLike,
    center_x: float = 0.5,
    center_y: float = 0.5,
    outer_radius: float = 0.28,
    inner_radius: float = 0.16,
) -> jnp.ndarray:
    outer = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= outer_radius ** 2
    inner = ((X - center_x) ** 2 + (Y - center_y) ** 2) >= inner_radius ** 2
    return normalize_field(outer & inner)


def gmm_density(
    X: ArrayLike,
    Y: ArrayLike,
    components: list[tuple[float, float, float, float, float]] | None = None,
) -> jnp.ndarray:
    if components is None:
        components = [
            (0.55, 0.35, 0.55, 0.06, 0.06),
            (0.45, 0.65, 0.45, 0.06, 0.06),
        ]
    density = jnp.zeros_like(X, dtype=jnp.float64)
    for weight, mx, my, sx, sy in components:
        density = density + weight * gaussian2d(X, Y, mx, my, sx, sy)
    return normalize_field(density)


def crescent_density(
    X: ArrayLike,
    Y: ArrayLike,
    outer_center_x: float = 0.5,
    outer_center_y: float = 0.5,
    outer_radius: float = 0.30,
    inner_center_x: float = 0.58,
    inner_center_y: float = 0.5,
    inner_radius: float = 0.22,
) -> jnp.ndarray:
    outer = ((X - outer_center_x) ** 2 + (Y - outer_center_y) ** 2) <= outer_radius ** 2
    inner = ((X - inner_center_x) ** 2 + (Y - inner_center_y) ** 2) <= inner_radius ** 2
    return normalize_field(outer & (~inner))


def plus_density(
    X: ArrayLike,
    Y: ArrayLike,
    v_center_x: float = 0.5,
    v_center_y: float = 0.5,
    v_half_width: float = 0.05,
    v_half_height: float = 0.3,
    h_center_x: float = 0.5,
    h_center_y: float = 0.5,
    h_half_width: float = 0.3,
    h_half_height: float = 0.05,
) -> jnp.ndarray:
    vertical = (jnp.abs(X - v_center_x) <= v_half_width) & (jnp.abs(Y - v_center_y) <= v_half_height)
    horizontal = (jnp.abs(Y - h_center_y) <= h_half_height) & (jnp.abs(X - h_center_x) <= h_half_width)
    return normalize_field(vertical | horizontal)


def diamond_density(
    X: ArrayLike,
    Y: ArrayLike,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius: float = 0.28,
) -> jnp.ndarray:
    return normalize_field(diamond_mask(X, Y, center_x, center_y, radius))


def square_density(
    X: ArrayLike,
    Y: ArrayLike,
    center_x: float = 0.5,
    center_y: float = 0.5,
    half_size: float = 0.2,
) -> jnp.ndarray:
    return normalize_field(
        (jnp.abs(X - center_x) <= half_size)
        & (jnp.abs(Y - center_y) <= half_size)
    )


def two_squares_density(
    X: ArrayLike,
    Y: ArrayLike,
    left_center_x: float = 0.35,
    right_center_x: float = 0.65,
    center_y: float = 0.5,
    half_size: float = 0.08,
) -> jnp.ndarray:
    left = (jnp.abs(X - left_center_x) <= half_size) & (jnp.abs(Y - center_y) <= half_size)
    right = (jnp.abs(X - right_center_x) <= half_size) & (jnp.abs(Y - center_y) <= half_size)
    return normalize_field(left | right)


def checker_density(
    X: ArrayLike,
    Y: ArrayLike,
    freq: int = 8,
) -> jnp.ndarray:
    return normalize_field(((jnp.floor(freq * X) + jnp.floor(freq * Y)) % 2) == 0)


def star_density(
    X: ArrayLike,
    Y: ArrayLike,
    R: float = 0.30,
    thickness: float = 0.018,
) -> jnp.ndarray:
    return david_star_density(X, Y, R=R, thickness=thickness)


def get_shape_factories(
    X: ArrayLike,
    Y: ArrayLike,
    checker_freq: int = 8,
    star_thickness: float = 0.018,
) -> dict[str, Callable[[], jnp.ndarray]]:
    # Build callables so densities are computed only when requested.
    return {
        "Ring": lambda: ring_density(X, Y),
        "GMM": lambda: gmm_density(X, Y),
        "Crescent": lambda: crescent_density(X, Y),
        "Plus": lambda: plus_density(X, Y),
        "Diamond": lambda: diamond_density(X, Y),
        "Square": lambda: square_density(X, Y),
        "Two Squares": lambda: two_squares_density(X, Y),
        "Checker": lambda: checker_density(X, Y, freq=checker_freq),
        "Star": lambda: star_density(X, Y, thickness=star_thickness),
    }


def get_all_shape_fields(
    X: ArrayLike,
    Y: ArrayLike,
    checker_freq: int = 8,
    star_thickness: float = 0.018,
) -> dict[str, jnp.ndarray]:
    factories = get_shape_factories(
        X,
        Y,
        checker_freq=checker_freq,
        star_thickness=star_thickness,
    )
    return {name: factory() for name, factory in factories.items()}


def get_shape_fields(
    X: ArrayLike,
    Y: ArrayLike,
    shape_names: list[str] | tuple[str, ...] | None = None,
    checker_freq: int = 8,
    star_thickness: float = 0.018,
) -> dict[str, jnp.ndarray]:
    factories = get_shape_factories(
        X,
        Y,
        checker_freq=checker_freq,
        star_thickness=star_thickness,
    )
    if shape_names is None:
        shape_names = DEFAULT_SHAPE_NAMES
    missing = [name for name in shape_names if name not in factories]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown shape name(s): {missing_str}")
    return {name: factories[name]() for name in shape_names}


def get_measures_weights(
    shape_fields: dict[str, jnp.ndarray],
    shape_names: list[str] | tuple[str, ...] | None = None,
) -> list[jnp.ndarray]:
    if shape_names is None:
        return list(shape_fields.values())
    missing = [name for name in shape_names if name not in shape_fields]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown shape name(s): {missing_str}")
    return [shape_fields[name] for name in shape_names]
