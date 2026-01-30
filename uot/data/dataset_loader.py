from uot.data.measure import DiscreteMeasure, GridMeasure
from uot.utils.types import ArrayLike
from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax.ops import segment_sum
from skimage.color import rgb2lab, lab2rgb


def _normalize_lab(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    l = np.clip(lab[..., 0], 0.0, 100.0) / 100.0
    a = (lab[..., 1] + 128.0) / 255.0
    b = (lab[..., 2] + 128.0) / 255.0
    return np.stack([l, a, b], axis=-1)


def _denormalize_lab(norm_lab: np.ndarray) -> np.ndarray:
    norm_lab = np.asarray(norm_lab, dtype=np.float64)
    l = np.clip(norm_lab[..., 0], 0.0, 1.0) * 100.0
    a = np.clip(norm_lab[..., 1], 0.0, 1.0) * 255.0 - 128.0
    b = np.clip(norm_lab[..., 2], 0.0, 1.0) * 255.0 - 128.0
    return np.stack([l, a, b], axis=-1)


def convert_rgb_to_color_space(rgb: np.ndarray, color_space: str) -> np.ndarray:
    space = color_space.strip().lower()
    if space == "rgb":
        return np.asarray(rgb, dtype=np.float64)
    if space in {"lab", "cielab"}:
        return _normalize_lab(rgb2lab(np.asarray(rgb, dtype=np.float64)))
    raise ValueError(f"Unsupported color space: {color_space}")


def convert_color_space_to_rgb(image: np.ndarray, color_space: str) -> np.ndarray:
    space = color_space.strip().lower()
    if space == "rgb":
        return np.asarray(image, dtype=np.float64)
    if space in {"lab", "cielab"}:
        return lab2rgb(_denormalize_lab(image))
    raise ValueError(f"Unsupported color space: {color_space}")

def load_csv_as_discrete(path: str) -> DiscreteMeasure:
    "Loads the discrete measure from the defined path to the data."
    raise NotImplementedError()


def load_matrix_as_color_grid(pixels: ArrayLike, num_channels: int, bins_per_channel: int = 32, use_jax: bool = False) -> GridMeasure:
    """
    Converts a matrix to a color grid measure.
    Fast and vectorized version for both NumPy and JAX.
    """
    bin_edges = np.linspace(0.0, 1.0, bins_per_channel + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = np.stack([
        np.clip(np.digitize(np.asarray(pixels[:, ch]), bin_edges) - 1, 0, bins_per_channel - 1)
        for ch in range(num_channels)
    ], axis=1)

    if use_jax:

        multipliers = np.array([bins_per_channel ** i for i in reversed(range(num_channels))])
        flat_bins = (bins @ multipliers).astype(np.int32)

        weights_1d = jnp.ones(flat_bins.shape[0], dtype=jnp.float64)
        flat_hist = segment_sum(weights_1d, flat_bins, bins_per_channel ** num_channels)
        weights_nd = flat_hist.reshape([bins_per_channel] * num_channels)
    else:
        weights_nd = np.zeros([bins_per_channel] * num_channels, dtype=np.float64)
        for idx in bins:
            weights_nd[tuple(idx)] += 1

    axes = [bin_centers for _ in range(num_channels)]

    return GridMeasure(axes=axes, weights_nd=weights_nd, normalize=True)


def load_image_as_color_grid(
    path: str,
    bins_per_channel: int = 32,
    use_jax: bool = False,
    *,
    color_space: str = "rgb",
    active_channels: list[int] | None = None,
) -> GridMeasure:
    """
    Loads the image at `path` and converts it to a color grid measure.
    Fast and vectorized version for both NumPy and JAX.
    """
    lib = jnp if use_jax else np

    image = Image.open(path)
    data = np.asarray(image)

    if data.ndim == 2:
        data = data[:, :, None]

    rgb = data.astype(np.float64) / 255.0
    color_data = convert_rgb_to_color_space(rgb, color_space)
    if active_channels is not None:
        color_data = color_data[..., active_channels]
    num_channels = color_data.shape[2]
    pixels = color_data.reshape(-1, num_channels).astype(lib.float64)

    return load_matrix_as_color_grid(
        pixels=pixels,
        num_channels=num_channels,
        bins_per_channel=bins_per_channel,
        use_jax=use_jax
    )


def load_image_as_binary_grid(
    path: str,
    *,
    size: tuple[int, int] | None = None,
    resample: int | None = None,
    threshold: float = 0.5,
    invert: bool = False,
    use_jax: bool = False,
    normalize: bool = True,
    axes_mode: str = "normalized",  # "normalized" | "pixel"
) -> GridMeasure:
    """
    Load a PNG/JPG image as a 2D binary GridMeasure (0/1), then optionally normalize.

    - threshold is applied on grayscale intensities in [0, 1].
    - axes_mode="normalized" creates axes in [0,1]; "pixel" uses 0..H-1 and 0..W-1.
    """
    image = Image.open(path).convert("L")
    if size is not None:
        resample_mode = Image.BILINEAR if resample is None else resample
        image = image.resize(size, resample=resample_mode)
    data = np.asarray(image, dtype=np.float64) / 255.0

    if invert:
        data = 1.0 - data

    binary = (data >= threshold).astype(np.float64)

    if normalize:
        total = binary.sum()
        if total > 0:
            binary = binary / total

    if axes_mode == "normalized":
        ax0 = np.linspace(0.0, 1.0, binary.shape[0])
        ax1 = np.linspace(0.0, 1.0, binary.shape[1])
    elif axes_mode == "pixel":
        ax0 = np.arange(binary.shape[0])
        ax1 = np.arange(binary.shape[1])
    else:
        raise ValueError("axes_mode must be 'normalized' or 'pixel'")

    if use_jax:
        axes = [jnp.asarray(ax0), jnp.asarray(ax1)]
        weights_nd = jnp.asarray(binary)
    else:
        axes = [ax0, ax1]
        weights_nd = binary

    return GridMeasure(axes=axes, weights_nd=weights_nd, normalize=False)
