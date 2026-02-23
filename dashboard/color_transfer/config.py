from __future__ import annotations

import os
from pathlib import Path


# NOTE:
# The dashboard is typically run with the working directory set to the
# `dashboard/` folder. These paths are therefore interpreted relative to
# that directory unless absolute paths are provided via environment
# variables.


def _as_path(value: str | os.PathLike[str]) -> Path:
    """Return a `Path` instance for the given value."""
    return value if isinstance(value, Path) else Path(value)


# Root directory that contains per‑run subfolders with
# `color_transfer_results.csv` and an `images/` directory.
#
# - Primary source is the COLOR_TRANSFER_DATA_FOLDER environment variable.
# - Fallback is the current working directory (historical behaviour).
COLOR_TRANSFER_DATA_FOLDER: Path = _as_path(
    os.getenv("COLOR_TRANSFER_DATA_FOLDER", ".")
)


# Directory that stores the original (untransported) images.
#
# Historically the default folder name was `colour_transfer_images`
# (British spelling). The actual data on disk typically uses
# `color_transfer_images`. We prefer the American spelling by default
# while still honouring an explicit environment override.
COLOR_TRANSFER_SOURCE_FOLDER_DEFAULT = "color_transfer_images"

ORIGINAL_IMAGES_FOLDER: Path = _as_path(
    os.getenv("COLOR_TRANSFER_SOURCE_FOLDER", COLOR_TRANSFER_SOURCE_FOLDER_DEFAULT)
)

