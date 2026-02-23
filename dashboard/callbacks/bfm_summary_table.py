import dash
from dash import Output, Input, html, dash_table
import pandas as pd

import numpy as np

from utils.bfm import (
    filtered_bfm_dataframe,
    BFM_PERFORMANCE_COLUMNS,
    BFM_MONGE_COLUMNS,
)


def _as_scalar(value):
    if isinstance(value, pd.Series):
        return value.iloc[0] if not value.empty else None
    if isinstance(value, (pd.Index, list, tuple)):
        return value[0] if len(value) else None
    if hasattr(value, "item") and isinstance(value, (np.generic,)):
        return value.item()
    return value


def _is_missing(value) -> bool:
    result = pd.isna(value)
    if isinstance(result, (pd.Series, np.ndarray)):
        return bool(result.all())
    if isinstance(result, (list, tuple)):
        return all(result)
    return bool(result)


@dash.callback(
    Output("bfm-summary-table", "children"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_summary_table(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if (
        len(dims) == 0
        or len(dists) == 0
        or len(stepsizes) == 0
        or len(sizes) == 0
        or len(pushforwards) == 0
    ):
        return html.Div("Select filters to populate the summary table.", className="text-muted")
    if df.empty:
        return html.Div("No data for these filters.", className="text-muted")

    metrics = [m for m in BFM_PERFORMANCE_COLUMNS + BFM_MONGE_COLUMNS if m in df.columns]
    group_cols = ["dim", "distribution", "stepsize"]
    if "pushforward_fn" in df.columns:
        group_cols.append("pushforward_fn")

    summary = (
        df.groupby(group_cols)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    columns = [
        {"name": "Dimension", "id": "dim"},
        {"name": "Distribution", "id": "distribution"},
        {"name": "Stepsize", "id": "stepsize"},
    ]
    if "pushforward_fn" in summary.columns:
        columns.append({"name": "Pushforward", "id": "pushforward_fn"})
    for metric in metrics:
        columns.append(
            {"name": f"{metric} (mean ± std)", "id": metric, "presentation": "markdown"}
        )

    data = []
    for _, row in summary.iterrows():
        dist_val = _as_scalar(row["distribution"])
        dist_display = "–" if _is_missing(dist_val) else str(dist_val)
        dim_val = _as_scalar(row["dim"])
        dim_display = "–" if _is_missing(dim_val) else int(dim_val)
        push_val = _as_scalar(row["pushforward_fn"]) if "pushforward_fn" in summary.columns else None
        push_display = "–" if _is_missing(push_val) else str(push_val)
        stepsize_val = _as_scalar(row["stepsize"])
        if _is_missing(stepsize_val):
            step_display = "–"
        else:
            try:
                step_display = float(stepsize_val)
            except (TypeError, ValueError):
                step_display = str(stepsize_val)

        record = {
            "dim": dim_display,
            "distribution": dist_display,
            "stepsize": step_display,
        }
        if "pushforward_fn" in summary.columns:
            record["pushforward_fn"] = push_display
        for metric in metrics:
            mean = row[(metric, "mean")]
            std = row[(metric, "std")]
            if pd.isna(mean):
                record[metric] = "–"
            elif pd.isna(std):
                record[metric] = f"{mean:.4g}"
            else:
                record[metric] = f"{mean:.4g} ± {std:.2g}"
        data.append(record)

    return dash_table.DataTable(
        columns=columns,
        data=data,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "monospace"},
        page_size=15,
    )
