import logging
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from PIL import Image


COLOR_TRANSFER_DATA_FOLDER_IGNORE = [".DS_Store"]
DATA_FILE_NAME = "color_transfer_results.csv"
DATA_DIRECTORY_NAME = "images"


@lru_cache(maxsize=256)
def _read_image_cached(path_str: str) -> np.ndarray:
    """Read an image from disk and cache the RGB array."""
    path = Path(path_str)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def _read_image(path: Path) -> np.ndarray:
    return _read_image_cached(str(path))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def load_color_transfer_data(
    path: str | Path,
) -> tuple[pd.DataFrame, Callable[[str, str, Mapping[str, Any]], np.ndarray | None]]:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists() or not path.is_dir():
        logging.warning(
            "Color transfer data directory %s does not exist or is not a directory",
            path,
        )
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    try:
        contents = set(os.listdir(path)) - set(COLOR_TRANSFER_DATA_FOLDER_IGNORE)
    except OSError as exc:  # pragma: no cover - very rare, but safe-guarded
        logging.warning("Could not list contents of %s: %s", path, exc)
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    collections = []
    for subpath in contents:
        run_path = path / Path(subpath)
        # Only attempt to load valid directories – ignore stray files.
        if not run_path.is_dir():
            continue
        df_run, loader = load_run_data(run_path)
        if not df_run.empty:
            collections.append((df_run, loader))

    if not collections:
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    df = pd.concat([entry[0] for entry in collections], ignore_index=True)
    df = preprocess(df)

    solver_loaders: dict[str, Callable[[str, Mapping[str, Any] | None], np.ndarray | None]] = {}
    run_loaders: dict[str, Callable[[str, Mapping[str, Any] | None], np.ndarray | None]] = {}
    for df_run, loader in collections:
        solver_name = df_run["name"].unique()[0]
        solver_loaders[solver_name] = loader
        if "run_folder" in df_run.columns:
            run_folder = df_run["run_folder"].iloc[0]
            if isinstance(run_folder, str):
                run_loaders[run_folder] = loader

    def load_image(image_name: str, solver: str, params: Mapping[str, Any]) -> np.ndarray | None:
        run_folder = None
        if params is not None:
            run_folder = params.get("run_folder")
        loader = run_loaders.get(run_folder) if run_folder else None
        if loader is None:
            loader = solver_loaders.get(solver)
        if loader is None:
            logging.warning("No image loader found for solver %s", solver)
            return None
        return loader(image_name, params)

    return df, load_image


def load_original_image_loader(
    path: str | Path,
) -> Callable[[str], np.ndarray | None]:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        logging.warning("Original images directory %s does not exist", path)

    def _load_original(image_name: str) -> np.ndarray | None:
        candidate = path / str(image_name)
        if not candidate.exists():
            logging.warning("Original image %s not found in %s", image_name, path)
            return None
        return _read_image(candidate)

    return _load_original


def load_run_data(
    path: str | Path,
) -> tuple[pd.DataFrame, Callable[[str, Mapping[str, Any] | None], np.ndarray | None]]:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists() or not path.is_dir():
        logging.warning("Run directory %s does not exist or is not a directory", path)
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    try:
        entries = list(path.iterdir())
    except OSError as exc:  # pragma: no cover - defensive
        logging.warning("Could not list run directory %s: %s", path, exc)
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    files = [p for p in entries if p.is_file()]
    folders = [f for f in entries if f.is_dir()]
    if DATA_FILE_NAME not in map(lambda x: x.name, files):
        logging.warning("%s has no data file %s", path, DATA_FILE_NAME)
        return pd.DataFrame(), lambda *_args, **_kwargs: None
    if DATA_DIRECTORY_NAME not in map(lambda x: x.name, folders):
        logging.warning("%s has no data directory %s", path, DATA_DIRECTORY_NAME)
        return pd.DataFrame(), lambda *_args, **_kwargs: None

    df = pd.read_csv(path / DATA_FILE_NAME)
    df.loc[df["name"] == "sinkhor-log", "name"] = "sinkhorn-log"
    df["run_folder"] = path.name

    images_dir = path / DATA_DIRECTORY_NAME

    def _load_image(image_name: str, params: Mapping[str, Any] | None = None) -> np.ndarray | None:
        explicit = None
        if params is not None:
            explicit = params.get("result_image_filename")
        if not explicit or _is_missing(explicit):
            logging.warning(
                "Missing result_image_filename for %s in %s", image_name, images_dir
            )
            return None
        candidate = images_dir / str(explicit)
        if not candidate.exists():
            logging.warning("Listed image %s not found in %s", explicit, images_dir)
            return None
        return _read_image(candidate)

    return df, _load_image


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={
        "name": "solver",
    }, inplace=True)
    if "numItermax" in df.columns:
        df["maxiter"] = df.get("numItermax", 0).fillna(0) + df.get("maxiter", 0).fillna(0)
        df.drop("numItermax", axis=1, inplace=True, errors="ignore")
    df.fillna({
        "reg": 0.0,
        "tol": 1e-12,
    }, inplace=True)
    df.loc[df["solver"] == "sinkhor-log", "solver"] = "sinkhorn-log"
    # Normalise a few commonly used parameter columns so that the
    # dashboard filters can rely on predictable types.
    if "bins_per_channel" in df.columns:
        df["bins_per_channel"] = pd.to_numeric(df["bins_per_channel"], errors="coerce")
    if "displacement_alpha" in df.columns:
        df["displacement_alpha"] = pd.to_numeric(
            df["displacement_alpha"],
            errors="coerce",
        )
    if "color_space" in df.columns:
        df["color_space"] = df["color_space"].astype(str)
    for reg in df['reg'].unique():
        for sol in df.loc[df['reg'] == reg, 'solver'].unique():
            df.loc[(df["solver"] == sol) & (df['reg'] == reg), 'solver'] = f"{sol} (reg={reg})"
    return df


def extract_image_params(row: Mapping[str, Any]) -> dict[str, Any]:
    filename = None
    run_folder = None
    if hasattr(row, "get"):
        filename = row.get("result_image_filename")
        run_folder = row.get("run_folder")
    else:
        filename = row["result_image_filename"] if "result_image_filename" in row else None
        run_folder = row["run_folder"] if "run_folder" in row else None
    if filename is None or _is_missing(filename):
        return {}
    params = {"result_image_filename": filename}
    if run_folder is not None and not _is_missing(run_folder):
        params["run_folder"] = run_folder
    return params
