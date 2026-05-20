from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShowcaseOptions:
    bins: list[int]
    color_spaces: list[str]
    solvers: list[str]


def sorted_unique_values(frame: pd.DataFrame, column: str, *, as_str: bool = False) -> list:
    if column not in frame.columns:
        return []
    values = frame[column].dropna()
    if as_str:
        values = values.astype(str)
    return sorted(values.unique().tolist())


def build_showcase_options(frame: pd.DataFrame) -> ShowcaseOptions:
    return ShowcaseOptions(
        bins=sorted_unique_values(frame, "bins_per_channel"),
        color_spaces=sorted_unique_values(frame, "color_space", as_str=True),
        solvers=sorted_unique_values(frame, "solver", as_str=True),
    )
