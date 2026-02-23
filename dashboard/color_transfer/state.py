from __future__ import annotations

import pandas as pd


def filter_showcase_dataframe(
    frame: pd.DataFrame,
    selected_bins,
    selected_color_space,
    selected_solver,
) -> pd.DataFrame:
    filtered = frame.copy()
    if selected_bins is not None and "bins_per_channel" in filtered.columns:
        try:
            bins_value = int(selected_bins)
        except (TypeError, ValueError):
            return filtered.iloc[0:0]
        filtered = filtered[filtered["bins_per_channel"] == bins_value]

    if selected_color_space and "color_space" in filtered.columns:
        # Normalise both sides to string to avoid subtle dtype mismatches.
        selected_color_space_str = str(selected_color_space)
        filtered = filtered[
            filtered["color_space"].astype(str) == selected_color_space_str
        ]

    if selected_solver and "solver" in filtered.columns:
        selected_solver_str = str(selected_solver)
        filtered = filtered[filtered["solver"].astype(str) == selected_solver_str]

    return filtered


def build_problem_list(frame: pd.DataFrame) -> list[tuple[str, str]]:
    required = {"source_image_name", "target_image_name"}
    if not required.issubset(frame.columns):
        return []

    group_cols = ["source_image_name", "target_image_name"]
    grouped = (
        frame[group_cols]
        .dropna(subset=["source_image_name", "target_image_name"])
        .drop_duplicates()
        .sort_values(["source_image_name", "target_image_name"])
    )
    return list(grouped.itertuples(index=False, name=None))


def step_problem_index(current_index, triggered_id, max_index: int) -> int:
    index = current_index or 0
    if triggered_id == "ct-prev-btn":
        index -= 1
    elif triggered_id == "ct-next-btn":
        index += 1
    return clamp_index(index, max_index)


def clamp_index(index: int, max_index: int) -> int:
    return max(0, min(index, max_index))


def format_problem_label(index: int, total: int, source_name: str, target_name: str) -> str:
    return f"{index + 1}/{total}: {source_name} -> {target_name}"
