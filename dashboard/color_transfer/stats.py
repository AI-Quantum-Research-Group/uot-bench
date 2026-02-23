from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from color_transfer.config import COLOR_TRANSFER_DATA_FOLDER
from color_transfer_data_loader import load_color_transfer_data

METRIC_GROUPS = {
    "runtime": ["iterations", "error", "time", "cost"],
    "distance": ["sinkhorn_divergence", "kl_divergence"],
    "map": ["ma_residual_L1", "ma_residual_Linf", "map_diffuseness", "tv_mu_to_nu"],
    "image_quality": ["ssim", "colorfulness_diff", "gradient_correlation", "laplacian_sharpness_diff"],
}

LOG_SCALE_METRICS = {
    "time",
    "error",
    "iterations",
    "sinkhorn_divergence",
    "kl_divergence",
    "ma_residual_L1",
    "ma_residual_Linf",
    "tv_mu_to_nu",
    "marginal_error",
}


def coerce_solver_base(value: str) -> str:
    if not isinstance(value, str):
        return value
    return re.sub(r"\s*\(reg=.*\)$", "", value)


def with_solver_base(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "solver" in data.columns:
        data["solver_base"] = data["solver"].map(coerce_solver_base)
    else:
        data["solver_base"] = None
    return data


def sorted_unique_values(frame: pd.DataFrame, column: str) -> list:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().unique().tolist())


def extract_stats_options(frame: pd.DataFrame) -> dict[str, list]:
    return {
        "solvers": sorted_unique_values(frame, "solver_base"),
        "alphas": sorted_unique_values(frame, "displacement_alpha"),
        "bins": sorted_unique_values(frame, "bins_per_channel"),
        "regs": sorted_unique_values(frame, "reg"),
        "color_spaces": sorted_unique_values(frame, "color_space"),
    }


def filter_stats_dataframe(
    frame: pd.DataFrame,
    solvers,
    alphas,
    bins,
    regs,
    color_spaces,
) -> pd.DataFrame:
    filtered = frame.copy()
    if solvers and "solver_base" in filtered.columns:
        filtered = filtered[filtered["solver_base"].isin(solvers)]
    if alphas and "displacement_alpha" in filtered.columns:
        filtered = filtered[filtered["displacement_alpha"].isin(alphas)]
    if bins and "bins_per_channel" in filtered.columns:
        filtered = filtered[filtered["bins_per_channel"].isin(bins)]
    if regs and "reg" in filtered.columns:
        filtered = filtered[filtered["reg"].isin(regs)]
    if color_spaces and "color_space" in filtered.columns:
        filtered = filtered[filtered["color_space"].isin(color_spaces)]
    return filtered


@lru_cache(maxsize=1)
def get_stats_dataframe():
    """Load and cache the color transfer statistics dataframe.

    Returns
    -------
    df:
        Preprocessed dataframe with an extra ``solver_base`` column.
    status:
        Optional human‑readable status message describing problems
        encountered while loading the data. ``None`` when everything
        looks fine.
    """
    df, _ = load_color_transfer_data(COLOR_TRANSFER_DATA_FOLDER)
    if df.empty:
        return df, f"No color transfer statistics found in {COLOR_TRANSFER_DATA_FOLDER}"
    return with_solver_base(df), None
